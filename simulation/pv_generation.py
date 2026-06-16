"""
光伏出力合成模块
基于数据集逐时归一化系数 + Markov天气序列, 生成8760h年度光伏出力序列

v4: 支持CalendarContext统一日历, 天气序列由Markov链生成(具有时间持续性)
"""
import numpy as np
from config import (
    PV_COEFF, WEATHER_COEFF, MONTHLY_WEATHER_DAYS, get_season,
    PV_TEMP_COEFF, PV_NOCT, PV_STC_TEMP, MONTHLY_AMBIENT_TEMP,
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

    def _get_cell_temperature(self, coeff_array, ambient_temp):
        """Sandia热模型: 计算逐时电池温度

        模型: T_cell = T_ambient + (NOCT - 20) * G_effective / 800
        其中 G_effective 由归一化PV系数反推 (峰值系数≈1000W/m² STC)

        数据来源: microgrid_architecture.md §2.1 TOPCon温度系数 -0.30%/℃
                 NOCT=43℃ (IEC 61215)
        """
        # 估算逐时有效辐照度: PV系数峰值对应1000W/m² STC条件
        g_effective = coeff_array / max(coeff_array.max(), 0.01) * 1000.0
        # Sandia电池温度模型
        t_cell = ambient_temp + (PV_NOCT - 20.0) * g_effective / 800.0
        return t_cell

    def _apply_temperature_derate(self, power, t_cell):
        """温度功率修正: P = P_stc * (1 + γ * (T_cell - T_stc))

        数据来源: microgrid_architecture.md, TOPCon γ=-0.30%/℃
        """
        return power * (1.0 + PV_TEMP_COEFF * (t_cell - PV_STC_TEMP))

    def generate_daily_profile(self, season, weather_type, month=None):
        """生成某一天的24h光伏出力 (kW), 含温度效应修正

        Parameters
        ----------
        season : str
            季节: spring/summer/autumn/winter
        weather_type : str
            天气类型: clear/partly_cloudy/cloudy/overcast/rain
        month : int or None
            月份(1-12), 用于查询环境温度. 若为None则使用季节代表月份.
        """
        coeff = np.array(PV_COEFF[season])
        weather_factor = WEATHER_COEFF.get(weather_type, 1.0)
        ideal_output = coeff * self.pv_capacity * weather_factor

        # 温度修正 (Sandia热模型)
        if month is None:
            month = SEASON_REPRESENTATIVE_MONTH.get(season, 4)
        ambient_temp = MONTHLY_AMBIENT_TEMP.get(month, 20.0)
        t_cell = self._get_cell_temperature(coeff * weather_factor, ambient_temp)
        temp_corrected = self._apply_temperature_derate(ideal_output, t_cell)

        # 添加小幅随机扰动 (±5%)
        noise = self.rng.uniform(0.95, 1.05, 24)
        return temp_corrected * noise

    def generate_annual(self):
        """生成8760h年度光伏出力序列 (含温度效应)"""
        hourly = np.zeros(8760)

        for day in range(365):
            month = self._day_to_month(day)
            season = get_season(month)
            weather = self.weather_seq[day]
            profile = self.generate_daily_profile(season, weather, month=month)
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
    annual = pv.generate_annual()
    metrics = pv.compute_annual_metrics(annual)
    monthly = pv.get_monthly_generation(annual)

    print(f"光伏装机: 500 kWp")
    print(f"年发电量: {metrics['annual_generation_kwh']:.0f} kWh")
    print(f"等效利用小时: {metrics['equivalent_hours']:.1f} h")
    print(f"容量因子: {metrics['capacity_factor']:.3f}")
    print(f"\n温度效应示例 (晴天, 不同季节正午12时):")
    for season in ['winter', 'spring', 'summer']:
        month = SEASON_REPRESENTATIVE_MONTH[season]
        ambient = MONTHLY_AMBIENT_TEMP[month]
        profile = pv.generate_daily_profile(season, 'clear', month=month)
        ideal = np.array(PV_COEFF[season]) * pv.pv_capacity
        temp_loss_pct = (1 - profile[12] / ideal[12]) * 100
        print(f"  {season} ({month}月): 环境温度≈{ambient}°C, "
              f"正午出力={profile[12]:.0f}kW (理想={ideal[12]:.0f}kW, 温度损失{temp_loss_pct:.1f}%)")
    print(f"\n月度发电量 (kWh):")
    for m, gen in enumerate(monthly):
        print(f"  {m+1:2d}月: {gen:.0f}")
