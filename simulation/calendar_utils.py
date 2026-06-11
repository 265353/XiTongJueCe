"""
真实日历模块 — 2025年日历 + Markov天气序列

解决两个关键简化:
1. 日类型不再随机分配, 而是基于2025年真实日历 (含中国法定节假日和调休)
2. 天气序列使用Markov链生成, 具有时间持续性 (晴天更可能连续出现)

统一PVGenerator和MicrogridOptimizer共用同一份日历和天气序列.
"""
import numpy as np
from config import MONTHLY_WEATHER_DAYS, get_season

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
WEATHER_PERSISTENCE = 0.65


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


def generate_weather_markov(rng, month, days_in_month_list, persistence=None):
    """
    使用一阶Markov链生成单月天气序列.

    转移概率: P(w'|w) = persistence * δ(w',w) + (1-persistence) * P_target(w')
    这保证了天气的时间持续性, 同时长期频率趋近于月度目标分布.

    Parameters
    ----------
    rng : np.random.RandomState
    month : int (1-12)
    days_in_month_list : list[int]
        每月天数列表
    persistence : float or None
        持续性参数, None则使用默认值

    Returns
    -------
    weather_seq : list[str] 该月每天天气类型
    """
    if persistence is None:
        persistence = WEATHER_PERSISTENCE

    wdays = MONTHLY_WEATHER_DAYS[month]
    weather_types = list(wdays.keys())
    n_days = days_in_month_list[month - 1]

    # 月度目标分布
    total = sum(wdays.values())
    target_probs = np.array([wdays[t] / total for t in weather_types])

    seq = []
    # 第一天从目标分布抽样
    prev_idx = rng.choice(len(weather_types), p=target_probs)
    seq.append(weather_types[prev_idx])

    for _ in range(1, n_days):
        if rng.random() < persistence:
            curr_idx = prev_idx  # 保持相同天气
        else:
            # 按目标分布切换
            curr_idx = rng.choice(len(weather_types), p=target_probs)
        seq.append(weather_types[curr_idx])
        prev_idx = curr_idx

    return seq


class CalendarContext:
    """
    日历上下文 — 统一管理日历和天气序列.

    用法:
        ctx = CalendarContext(seed=42)
        ctx.day_types[day]        # 第day天的类型
        ctx.weather_seq[day]      # 第day天的天气
        ctx.season_of_day[day]    # 第day天的季节
        ctx.day_of_week[day]      # 第day天是周几
    """

    def __init__(self, seed=None, weather_persistence=None):
        self.rng = np.random.RandomState(seed)
        self.persistence = (weather_persistence if weather_persistence is not None
                           else WEATHER_PERSISTENCE)

        # 构建日历
        (self.day_types, self.day_of_week,
         self.month_of_day, self.day_of_month) = build_calendar()

        # 构建天气序列 (Markov链)
        self.weather_seq = self._build_weather()

        # 预计算每天的季节
        self.season_of_day = [get_season(self.month_of_day[d]) for d in range(365)]

    def _build_weather(self):
        """使用Markov链生成全年365天天气序列"""
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        weather_seq = []
        for month in range(1, 13):
            month_weather = generate_weather_markov(
                self.rng, month, days_in_month, self.persistence
            )
            weather_seq.extend(month_weather)
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
        print("\n月度天气分布 (Markov链, persistence={:.2f}):".format(self.persistence))
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
