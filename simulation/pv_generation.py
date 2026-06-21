"""
光伏出力合成模块
基于数据集逐时归一化系数 + Markov天气序列, 生成8760h年度光伏出力序列

v4: 支持CalendarContext统一日历, 天气序列由Markov链生成(具有时间持续性)
v6.3: TMY逐时温度+GHI数据集成 (文件24 §1.1-1.2), Kasten-Czeplak天气修正 (文件24 §1.3)
"""
import numpy as np
from config import (
    PV_COEFF, WEATHER_COEFF, MONTHLY_WEATHER_DAYS, get_season,
    PV_TEMP_COEFF, PV_NOCT, PV_STC_TEMP, MONTHLY_AMBIENT_TEMP,
    TMY_HOURLY_TEMP, TMY_GHI_CLEAR, kasten_czeplak_ghi,
)

# 季节→代表月份映射 (用于典型日温度估算)
SEASON_REPRESENTATIVE_MONTH = {
    'spring': 4,    # 4月: 温和
    'summer': 7,    # 7月: 最热
    'autumn': 10,   # 10月: 凉爽
    'winter': 1,    # 1月: 最冷
}


class PVGenerator:
    """光伏出力合成器

    Parameters
    ----------
    pv_capacity_kwp : float
        光伏装机容量 (kWp)
    seed : int or None
        随机种子 (仅在未传入calendar_ctx时使用)
    calendar_ctx : CalendarContext or None
        统一日历上下文, 提供天气序列. 若为None则使用旧版随机shuffle模式.
    """

    def __init__(self, pv_capacity_kwp=500, seed=None, calendar_ctx=None):
        self.pv_capacity = pv_capacity_kwp
        self.rng = np.random.RandomState(seed)
        self.calendar_ctx = calendar_ctx

        if calendar_ctx is not None:
            self.weather_seq = calendar_ctx.weather_seq
        else:
            self._build_weather_sequence()

    def _build_weather_sequence(self):
        """旧版: 基于各月典型天气天数随机shuffle (无时间持续性)"""
        weather_seq = []
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        for month in range(1, 13):
            wdays = MONTHLY_WEATHER_DAYS[month]
            types = []
            for wtype, count in wdays.items():
                types.extend([wtype] * count)
            remaining = days_in_month[month - 1] - len(types)
            if remaining > 0:
                fill = self.rng.choice(list(wdays.keys()), remaining,
                                       p=[wdays[t]/sum(wdays.values()) for t in wdays])
                types.extend(fill)
            elif remaining < 0:
                types = types[:days_in_month[month - 1]]

            self.rng.shuffle(types)
            weather_seq.extend(types)

        self.weather_seq = weather_seq

    def _get_cell_temperature(self, coeff_array, ambient_temp, ghi_array=None):
        """Sandia热模型: 计算逐时电池温度

        模型: T_cell = T_ambient + (NOCT - 20) * G_effective / 800

        v6.3: 支持TMY逐时温度和GHI输入 (文件24 §1.1-1.2)
        - ambient_temp: 可为float (月均温, 向后兼容) 或 24元素array (TMY逐时)
        - ghi_array: 若提供则使用TMY真实GHI, 否则从PV系数反推
        """
        if ghi_array is not None:
            g_effective = ghi_array
        else:
            g_effective = coeff_array / max(coeff_array.max(), 0.01) * 1000.0
        t_cell = ambient_temp + (PV_NOCT - 20.0) * g_effective / 800.0
        return t_cell

    def _apply_temperature_derate(self, power, t_cell):
        """温度功率修正: P = P_stc * (1 + γ * (T_cell - T_stc))

        数据来源: microgrid_architecture.md, TOPCon γ=-0.30%/℃
        """
        return power * (1.0 + PV_TEMP_COEFF * (t_cell - PV_STC_TEMP))

    def generate_daily_profile(self, season, weather_type, month=None, use_tmy=True):
        """生成某一天的24h光伏出力 (kW), 含温度效应修正

        Parameters
        ----------
        season : str
            季节: spring/summer/autumn/winter
        weather_type : str
            天气类型: clear/partly_cloudy/cloudy/overcast/rain
        month : int or None
            月份(1-12). 若为None则使用季节代表月份.
        use_tmy : bool
            是否使用TMY逐时温度+GHI (默认True). False则回退到月均温+系数反推.
        """
        coeff = np.array(PV_COEFF[season])
        weather_factor = WEATHER_COEFF.get(weather_type, 1.0)

        # GHI: TMY真实数据 或 系数反推
        if use_tmy and month is not None:
            ghi_clear = np.array(TMY_GHI_CLEAR[month])
            # 根据云量使用Kasten-Czeplak衰减
            cloud_oktas_map = {'clear': 0, 'partly_cloudy': 3, 'cloudy': 5.5,
                              'overcast': 7.5, 'rain': 8}
            cloud_oktas = cloud_oktas_map.get(weather_type, 0)
            ghi = kasten_czeplak_ghi(ghi_clear, cloud_oktas)
            ideal_output = ghi / 1000.0 * self.pv_capacity
        else:
            ideal_output = coeff * self.pv_capacity * weather_factor
            ghi = None

        # 温度修正 (Sandia热模型)
        if month is None:
            month = SEASON_REPRESENTATIVE_MONTH.get(season, 4)

        if use_tmy and month is not None:
            ambient_temp = np.array(TMY_HOURLY_TEMP[month])
        else:
            ambient_temp = MONTHLY_AMBIENT_TEMP.get(month, 20.0)

        t_cell = self._get_cell_temperature(
            coeff * weather_factor if ghi is None else ghi / 1000.0,
            ambient_temp,
            ghi_array=ghi
        )
        temp_corrected = self._apply_temperature_derate(ideal_output, t_cell)

        # 添加小幅随机扰动 (±5%)
        noise = self.rng.uniform(0.95, 1.05, 24)
        return temp_corrected * noise

    def generate_annual(self, use_tmy=True):
        """生成8760h年度光伏出力序列 (含TMY温度+GHI效应)"""
        hourly = np.zeros(8760)

        for day in range(365):
            month = self._day_to_month(day)
            season = get_season(month)
            weather = self.weather_seq[day]
            profile = self.generate_daily_profile(season, weather, month=month, use_tmy=use_tmy)
            hourly[day * 24:(day + 1) * 24] = profile

        return hourly

    def generate_typical_days(self):
        """生成典型日光伏出力 (四季 × 晴天), 用于优化"""
        profiles = {}
        for season in ['spring', 'summer', 'autumn', 'winter']:
            profiles[season] = {
                'clear': self.generate_daily_profile(season, 'clear'),
                'partly_cloudy': self.generate_daily_profile(season, 'partly_cloudy'),
                'overcast': self.generate_daily_profile(season, 'overcast'),
                'rain': self.generate_daily_profile(season, 'rain'),
            }
        return profiles

    @staticmethod
    def _day_to_month(day):
        """将日序号(0-364)转换为月份(1-12)"""
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        cumsum = 0
        for i, d in enumerate(days_in_month):
            cumsum += d
            if day < cumsum:
                return i + 1
        return 12

    def compute_annual_metrics(self, hourly_output):
        """计算年度光伏指标"""
        total_gen = hourly_output.sum()  # kWh
        peak_power = hourly_output.max()
        eq_hours = total_gen / self.pv_capacity
        return {
            'annual_generation_kwh': total_gen,
            'peak_power_kw': peak_power,
            'equivalent_hours': eq_hours,
            'capacity_factor': eq_hours / 8760,
        }

    def get_monthly_generation(self, hourly_output):
        """月度发电量统计"""
        monthly = np.zeros(12)
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        day_start = 0
        for m in range(12):
            day_end = day_start + days_in_month[m]
            monthly[m] = hourly_output[day_start * 24:day_end * 24].sum()
            day_start = day_end
        return monthly


if __name__ == '__main__':
    pv = PVGenerator(pv_capacity_kwp=500, seed=42)

    print("=" * 60)
    print("v6.3: TMY逐时温度+GHI模式 (文件24 §1.1-1.2)")
    print("=" * 60)
    annual_tmy = pv.generate_annual(use_tmy=True)
    metrics_tmy = pv.compute_annual_metrics(annual_tmy)
    print(f"年发电量: {metrics_tmy['annual_generation_kwh']:.0f} kWh")
    print(f"等效利用小时: {metrics_tmy['equivalent_hours']:.1f} h")
    print(f"容量因子: {metrics_tmy['capacity_factor']:.3f}")

    print(f"\n温度效应对比 (晴天, 正午12时):")
    print(f"{'季节':<10s} {'月':>3s} {'TMY温(℃)':>10s} {'月均温(℃)':>10s} {'TMY出力(kW)':>12s} {'旧出力(kW)':>12s}")
    for season in ['winter', 'spring', 'summer']:
        month = SEASON_REPRESENTATIVE_MONTH[season]
        tmy_temp = TMY_HOURLY_TEMP[month][12]
        monthly_temp = MONTHLY_AMBIENT_TEMP[month]
        profile_tmy = pv.generate_daily_profile(season, 'clear', month=month, use_tmy=True)
        profile_old = pv.generate_daily_profile(season, 'clear', month=month, use_tmy=False)
        print(f"  {season:<8s} {month:>3d} {tmy_temp:>10.1f} {monthly_temp:>10.1f} {profile_tmy[12]:>12.0f} {profile_old[12]:>12.0f}")

    print(f"\n天气修正系数对比 (Kasten-Czeplak v6.3 vs 旧值):")
    print(f"{'天气':<18s} {'旧值':>6s} {'v6.3':>6s} {'变化':>8s}")
    old_vals = {'clear': 1.00, 'partly_cloudy': 0.80, 'cloudy': 0.55, 'overcast': 0.30, 'rain': 0.15}
    for wtype in ['clear', 'partly_cloudy', 'cloudy', 'overcast', 'rain']:
        old_v = old_vals[wtype]
        new_v = WEATHER_COEFF[wtype]
        print(f"  {wtype:<16s} {old_v:>6.2f} {new_v:>6.2f} {new_v-old_v:>+7.2f}")
