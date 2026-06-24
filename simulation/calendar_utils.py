"""
真实日历模块 — 2025年日历 + PVGIS 8760h实测气象

v6.7: PVGIS实测8760h逐时GHI+温度+风速替代Markov合成天气.
  晴天概率不再是硬编码转移矩阵, 而是2020年真实气象数据.

统一PVGenerator和MicrogridOptimizer共用同一份日历和天气序列.
"""
import json
import os
import numpy as np
from config import (
    MONTHLY_WEATHER_DAYS, get_season,
    WEATHER_TRANSITION_MATRIX, WEATHER_SEASONAL_PERSISTENCE,
    SECOND_ORDER_ALPHA,
    TMY_GHI_CLEAR,
)

# 2025年1月1日 = 周三 (0=Mon, 6=Sun)
CALENDAR_FIRST_WEEKDAY = 2

# 2025年中国法定节假日 — 仅含法定放假日期 (不含调休延长的假期)
# 法定节假日共11天: 元旦1 + 春节3 + 清明1 + 劳动1 + 端午1 + 中秋1 + 国庆3
# 其中春节3天标注为 spring_festival, 其余8天标注为 holiday
# 2025年所有11天法定假日均落在工作日上 (不含周末), 因此无需补休处理
SPRING_FESTIVAL_DATES = [
    (1, 28),    # 除夕 (周二)
    (1, 29),    # 正月初一 (周三)
    (1, 30),    # 正月初二 (周四)
]

STATUTORY_HOLIDAY_DATES = [
    (1, 1),     # 元旦 (周三)
    (4, 4),     # 清明节 (周五)
    (5, 1),     # 劳动节 (周四)
    (6, 2),     # 端午节 (周一 — 法定日5月31为周六, 补休至此)
    (10, 1),    # 国庆节 (周三)
    (10, 2),    # 国庆节 (周四)
    (10, 3),    # 国庆节 (周五)
    (10, 6),    # 中秋节 (周一 — 农历八月十五)
]

# 天气Markov链持续性参数 (0~1, 越高天气越持续)
# v6.3: 保留作为回退值, 推荐使用transition_matrix模式
WEATHER_PERSISTENCE = 0.65

# v6.7: PVGIS数据路径
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def _load_pvgis_hourly(location='wuhan'):
    """加载PVGIS 8760h逐时实测数据

    Returns
    -------
    dict with keys: hourly_ghi, hourly_temp, hourly_wind, hourly_count
    或 None (文件不存在时)
    """
    fpath = os.path.join(_DATA_DIR, f'pvgis_tmy_{location}.json')
    if not os.path.exists(fpath):
        return None
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        'hourly_ghi': np.array(data['hourly_ghi'], dtype=float),
        'hourly_temp': np.array(data['hourly_temp'], dtype=float),
        'hourly_wind': np.array(data['hourly_wind'], dtype=float),
        'hourly_count': data.get('hourly_count', 8760),
    }


def _classify_weather_from_ghi(ghi_hourly, ghi_clear_hourly):
    """从逐时GHI/晴空GHI比值分类天气类型

    使用Kasten-Czeplak逆推: 云量 = 8 * ((1 - GHI/GHI_clear) / 0.75)^(1/3.4)
    聚合到天级别: 取08-17时(日照时段)的平均云量分类.

    Returns
    -------
    str: 'clear' | 'partly_cloudy' | 'cloudy' | 'overcast' | 'rain'
    """
    # 只评估08-17时的日照时段
    daylight = slice(8, 18)
    ghi_day = ghi_hourly[daylight]
    ghi_clr = ghi_clear_hourly[daylight]

    ratios = []
    for g, gc in zip(ghi_day, ghi_clr):
        if gc > 10:
            ratios.append(np.clip(g / gc, 0.0, 1.0))

    if not ratios:
        return 'cloudy'

    avg_ratio = np.mean(ratios)
    if avg_ratio > 0.75:
        return 'clear'
    elif avg_ratio > 0.55:
        return 'partly_cloudy'
    elif avg_ratio > 0.30:
        return 'cloudy'
    elif avg_ratio > 0.12:
        return 'overcast'
    else:
        return 'rain'


def _utc_to_local_roll(arr_8760, utc_offset=8):
    """将UTC时间数组滚动到本地时间 (UTC→UTC+8: 每个24h块内roll +8)"""
    n_hours = len(arr_8760)
    n_days = n_hours // 24
    result = np.zeros_like(arr_8760)
    for d in range(n_days):
        day_utc = arr_8760[d * 24:(d + 1) * 24]
        result[d * 24:(d + 1) * 24] = np.roll(day_utc, utc_offset)
    return result


def _build_weather_from_pvgis(hourly_ghi, hourly_count=8760):
    """从PVGIS逐时GHI数据构建全年365天天气序列

    PVGIS数据为UTC时间, 武汉为UTC+8, 需要将每天24h块内roll(+8).
    然后取日照时段(08-17h)的GHI/晴空GHI比值,
    按Kasten-Czeplak逆推云量分类为5种天气类型.

    Parameters
    ----------
    hourly_ghi : np.ndarray (8760+,)
    hourly_count : int

    Returns
    -------
    weather_seq : list[str] (365,)
    hourly_ghi_8760 : np.ndarray (8760,) — 本地时间
    hourly_ghi_raw : np.ndarray — 原始UTC (前8760h)
    """
    n_days = 365
    n_hours = n_days * 24  # 8760
    # 截断到8760h
    ghi_utc = hourly_ghi[:n_hours]
    # UTC→本地时间 (UTC+8)
    ghi_local = _utc_to_local_roll(ghi_utc, 8)

    # 构建晴空GHI参考 (每月逐时, 本地时间)
    ghi_clear_hourly = np.zeros(n_hours)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_start = 0
    for month in range(1, 13):
        ndays = days_in_month[month - 1]
        ghi_clr_24 = np.array(TMY_GHI_CLEAR[month], dtype=float)
        for d in range(ndays):
            ghi_clear_hourly[(day_start + d) * 24:(day_start + d + 1) * 24] = ghi_clr_24
        day_start += ndays

    weather_seq = []
    for d in range(n_days):
        ghi_24 = ghi_local[d * 24:(d + 1) * 24]
        ghi_clr_24 = ghi_clear_hourly[d * 24:(d + 1) * 24]
        weather_seq.append(_classify_weather_from_ghi(ghi_24, ghi_clr_24))

    return weather_seq, ghi_local, ghi_utc


def _is_spring_festival(month, day):
    """检查是否为春节法定日"""
    return (month, day) in SPRING_FESTIVAL_DATES


def _is_statutory_holiday(month, day):
    """检查是否为法定节假日 (不含春节)"""
    return (month, day) in STATUTORY_HOLIDAY_DATES


def build_calendar():
    """
    构建2025年365天真实日历.

    Returns
    -------
    day_types : list[str] (365,)
        每天的类型: workday / weekend / holiday / spring_festival
    day_of_week : list[int] (365,)
        每周几: 0=Mon, 6=Sun
    month_of_day : list[int] (365,)
        每天所属月份 1-12
    day_of_month : list[int] (365,)
        每天在月份中的日期
    """
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    day_types = []
    day_of_week = []
    month_of_day = []
    day_of_month_list = []

    for month in range(1, 13):
        for dom in range(1, days_in_month[month - 1] + 1):
            day_idx = len(day_types)
            dow = (CALENDAR_FIRST_WEEKDAY + day_idx) % 7

            month_of_day.append(month)
            day_of_week.append(dow)
            day_of_month_list.append(dom)

            # 优先级: spring_festival > holiday > weekend > workday
            if _is_spring_festival(month, dom):
                day_types.append('spring_festival')
            elif _is_statutory_holiday(month, dom):
                day_types.append('holiday')
            elif dow >= 5:  # Saturday=5, Sunday=6
                day_types.append('weekend')
            else:
                day_types.append('workday')

    return day_types, day_of_week, month_of_day, day_of_month_list


def print_calendar_summary(day_types):
    """打印日历统计, 验证日类型计数"""
    from collections import Counter
    counts = Counter(day_types)
    print("日历统计 (2025年):")
    for dt in ['workday', 'weekend', 'holiday', 'spring_festival']:
        print(f"  {dt}: {counts.get(dt, 0)} 天")
    print(f"  总计: {sum(counts.values())} 天")


def _compute_persistence(wdays, weather_type, n_days_total):
    """基于月度天气天数反推类型特定持续性 (文件10 月度统计)

    原理: 天气类型出现天数越多, 说明该月该类型越"稳定"。
    P(stay|w) ∝ N_w / M, 下限0.35确保所有类型都有最小持续性。
    """
    n_w = wdays.get(weather_type, 0)
    if n_w <= 1:
        return 0.35
    # 持续性 = N_w占全月比例, 限定在 [0.35, 0.85]
    return np.clip(n_w / n_days_total * 1.3, 0.35, 0.85)


def generate_weather_markov(rng, month, days_in_month_list, persistence=None,
                            use_transition_matrix=True, prev_weather=None):
    """
    使用一阶Markov链生成单月天气序列.

    v6.3: 支持两种模式:
    - use_transition_matrix=True: 使用研究数据集转移矩阵 (文件24 §2.4),
      含季节性修正. 矩阵行和为1.0, 对角线=持续概率.
    - use_transition_matrix=False: 回退到v6.2类型特定持续性模型.

    二阶近似 (文件24 §2.4):
      P(w_t | w_{t-1}, w_{t-2}) ≈ α * P(w_t | w_{t-1}) + (1-α) * P(w_t | w_{t-2})
    当 prev_weather 提供时, 使用二阶近似修正一阶转移概率.

    Parameters
    ----------
    rng : np.random.RandomState
    month : int (1-12)
    days_in_month_list : list[int]
    persistence : float or None
        仅 use_transition_matrix=False 时使用
    use_transition_matrix : bool
        是否使用研究数据集转移矩阵
    prev_weather : str or None
        上个月的最后一个天气类型, 用于二阶近似 (可选)

    Returns
    -------
    weather_seq : list[str]
    """
    wdays = MONTHLY_WEATHER_DAYS[month]
    weather_types = list(wdays.keys())
    n_days = days_in_month_list[month - 1]

    # 月度目标分布 (用于第一天抽样)
    total = sum(wdays.values())
    target_probs = np.array([wdays[t] / total for t in weather_types])

    if use_transition_matrix:
        # v6.3: 使用研究数据集转移矩阵 + 季节性修正
        season = get_season(month)
        season_corr = WEATHER_SEASONAL_PERSISTENCE.get(season, {})

        # 构建月度转移矩阵 (从年度平均矩阵出发, 应用季节修正)
        trans = {}
        for w_from in weather_types:
            base_row = WEATHER_TRANSITION_MATRIX.get(w_from, {})
            row = {}
            for w_to in weather_types:
                p = base_row.get(w_to, 0.0)
                # 季节修正: 调整对角线 (持续概率)
                if w_to == w_from:
                    if w_from == 'clear':
                        p = season_corr.get('clear_stay', p)
                    elif w_from == 'rain':
                        p = season_corr.get('rain_stay', p)
                row[w_to] = p
            # 重新归一化行和=1.0
            row_sum = sum(row.values())
            if row_sum > 0:
                for w_to in row:
                    row[w_to] /= row_sum
            trans[w_from] = row
    else:
        # 回退到v6.2类型特定持续性模型
        if persistence is not None:
            persist_values = {t: persistence for t in weather_types}
        else:
            persist_values = {t: _compute_persistence(wdays, t, n_days)
                            for t in weather_types}

    seq = []
    # 第一天从目标分布抽样
    first_idx = rng.choice(len(weather_types), p=target_probs)
    prev_type = weather_types[first_idx]
    seq.append(prev_type)
    prev_prev_type = prev_weather if prev_weather is not None else prev_type

    for _ in range(1, n_days):
        if use_transition_matrix:
            # 一阶转移概率
            first_order = trans.get(prev_type, {})
            probs_1st = np.array([first_order.get(t, 0.0) for t in weather_types])

            # 二阶近似: α * P(w_t | w_{t-1}) + (1-α) * P(w_t | w_{t-2})
            if prev_prev_type is not None and prev_prev_type != prev_type:
                second_order_row = trans.get(prev_prev_type, {})
                probs_2nd = np.array([second_order_row.get(t, 0.0) for t in weather_types])
                probs = SECOND_ORDER_ALPHA * probs_1st + (1 - SECOND_ORDER_ALPHA) * probs_2nd
            else:
                probs = probs_1st

            # 归一化
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                probs = target_probs

            # 混合目标分布 (30%), 确保月度校准
            TARGET_BLEND = 0.30
            probs = (1 - TARGET_BLEND) * probs + TARGET_BLEND * target_probs
            probs = probs / probs.sum()

            curr_idx = rng.choice(len(weather_types), p=probs)
        else:
            p_stay = persist_values[prev_type]
            if rng.random() < p_stay:
                curr_idx = weather_types.index(prev_type)
            else:
                curr_idx = rng.choice(len(weather_types), p=target_probs)

        prev_prev_type = prev_type
        prev_type = weather_types[curr_idx]
        seq.append(prev_type)

    return seq


class CalendarContext:
    """
    日历上下文 — 统一管理日历和天气序列.

    v6.3: 支持 transition_matrix 模式 (文件24 §2.4), 含二阶Markov近似+季节性修正.

    用法:
        ctx = CalendarContext(seed=42)
        ctx.day_types[day]        # 第day天的类型
        ctx.weather_seq[day]      # 第day天的天气
        ctx.season_of_day[day]    # 第day天的季节
        ctx.day_of_week[day]      # 第day天是周几
    """

    def __init__(self, seed=None, weather_persistence=None, use_transition_matrix=True,
                 pvgis_location='wuhan'):
        self.rng = np.random.RandomState(seed)
        self.persistence = (weather_persistence if weather_persistence is not None
                           else WEATHER_PERSISTENCE)
        self.use_transition_matrix = use_transition_matrix
        self.pvgis_location = pvgis_location

        # 构建日历
        (self.day_types, self.day_of_week,
         self.month_of_day, self.day_of_month) = build_calendar()

        # v6.7: 优先加载PVGIS 8760h实测数据, 回退Markov合成
        pvgis = _load_pvgis_hourly(pvgis_location)
        if pvgis is not None:
            self._pvgis_ghi = pvgis['hourly_ghi']
            self._pvgis_temp = pvgis['hourly_temp']
            self._pvgis_wind = pvgis['hourly_wind']
            self._pvgis_count = pvgis['hourly_count']
            weather_seq, self.hourly_ghi, _ = _build_weather_from_pvgis(
                self._pvgis_ghi, self._pvgis_count)
            self.weather_seq = weather_seq
            # v6.7: UTC→本地时间 (UTC+8), 截断到8760h
            self.hourly_temp = _utc_to_local_roll(self._pvgis_temp[:8760], 8)
            self.hourly_wind = _utc_to_local_roll(self._pvgis_wind[:8760], 8)
            self._has_pvgis = True
        else:
            self._has_pvgis = False
            self.hourly_ghi = None
            self.hourly_temp = None
            self.hourly_wind = None
            # 旧版: Markov天气序列
            self.weather_seq = self._build_weather()

        # 预计算每天的季节
        self.season_of_day = [get_season(self.month_of_day[d]) for d in range(365)]

    def _build_weather(self):
        """使用Markov链生成全年365天天气序列 (v6.3: 转移矩阵+二阶近似)"""
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        weather_seq = []
        prev_month_last = None
        for month in range(1, 13):
            month_weather = generate_weather_markov(
                self.rng, month, days_in_month,
                persistence=self.persistence,
                use_transition_matrix=self.use_transition_matrix,
                prev_weather=prev_month_last,
            )
            weather_seq.extend(month_weather)
            prev_month_last = month_weather[-1]
        return weather_seq

    def get_day_weather(self, day_idx):
        """获取第day_idx天 (0-364) 的天气类型"""
        return self.weather_seq[day_idx]

    def get_day_type(self, day_idx):
        """获取第day_idx天的日类型"""
        return self.day_types[day_idx]

    def get_season(self, day_idx):
        """获取第day_idx天的季节"""
        return self.season_of_day[day_idx]

    def get_month(self, day_idx):
        """获取第day_idx天的月份"""
        return self.month_of_day[day_idx]

    def print_summary(self):
        """打印日历和天气统计"""
        from collections import Counter
        print("=" * 55)
        print("日历上下文统计 (CalendarContext)")
        print("=" * 55)

        # 日类型统计
        dt_counts = Counter(self.day_types)
        print("\n日类型分布:")
        for dt in ['workday', 'weekend', 'holiday', 'spring_festival']:
            bar = '█' * dt_counts.get(dt, 0)
            print(f"  {dt:20s}: {dt_counts.get(dt, 0):3d} 天 {bar}")

        # 天气统计 (按月)
        mode_label = "转移矩阵+二阶Markov" if self.use_transition_matrix else f"persistence={self.persistence:.2f}"
        print(f"\n月度天气分布 (Markov链, {mode_label}):")
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        day_start = 0
        for month in range(1, 13):
            month_end = day_start + days_in_month[month - 1]
            month_weather = self.weather_seq[day_start:month_end]
            w_counts = Counter(month_weather)
            parts = [f"{w}:{w_counts.get(w, 0)}" for w in
                     ['clear', 'partly_cloudy', 'cloudy', 'overcast', 'rain']]
            target_w = MONTHLY_WEATHER_DAYS[month]
            t_parts = [f"{w}={target_w[w]}" for w in
                       ['clear', 'partly_cloudy', 'cloudy', 'overcast', 'rain']]
            print(f"  {month:2d}月: {' | '.join(parts):45s}  (目标: {' | '.join(t_parts)})")
            day_start = month_end

        # 天气持续性统计 (平均连续天数)
        run_lengths = []
        current_run = 1
        for d in range(1, 365):
            if self.weather_seq[d] == self.weather_seq[d - 1]:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)
        print(f"\n天气持续性: 平均连续 {np.mean(run_lengths):.2f} 天, "
              f"最长连续 {max(run_lengths)} 天, "
              f"中位连续 {np.median(run_lengths):.1f} 天")
        print(f"  (对比: 纯随机shuffle期望连续 ~1.0 天)")


if __name__ == '__main__':
    ctx = CalendarContext(seed=42)
    ctx.print_summary()
