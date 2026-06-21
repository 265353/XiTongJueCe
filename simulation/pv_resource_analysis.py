"""
光伏资源评估模块 — 全国资源分区 / 服务区可用面积 / 发电潜力 / 按气候分区装机测算

理论来源: 文件04_分布式光伏发电潜力评估.md
         文件10_细粒度建模数据_光伏出力.md (四类资源区逐时归一化系数)

运行:
    python pv_resource_analysis.py       # 模块自测
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# ============================================================
# 全国太阳能资源分区数据
# 来源: 文件04 §1 (2024年全国太阳能资源概况)
# ============================================================

# 四类资源区逐时归一化系数 (来源: 文件10)
# 结构: zone -> season -> [24h coefficients]
PV_ZONE_COEFFICIENTS = {
    'I': {   # 一类: 青藏/川西, 年利用 1600-1800h
        'spring': [0,0,0,0,0,0,0.010,0.080,0.200,0.380,0.550,0.700,
                   0.800,0.820,0.780,0.680,0.500,0.300,0.120,0.020,0,0,0,0],
        'summer': [0,0,0,0,0,0,0.020,0.100,0.250,0.430,0.600,0.750,
                   0.850,0.880,0.830,0.720,0.550,0.350,0.160,0.040,0,0,0,0],
        'autumn': [0,0,0,0,0,0,0.005,0.060,0.180,0.350,0.520,0.670,
                   0.760,0.780,0.740,0.620,0.450,0.260,0.100,0.010,0,0,0,0],
        'winter': [0,0,0,0,0,0,0,0.030,0.120,0.280,0.440,0.560,
                   0.620,0.600,0.540,0.420,0.280,0.140,0.040,0,0,0,0,0],
    },
    'II': {  # 二类: 新疆/内蒙古/华北, 年利用 1400-1600h
        'spring': [0,0,0,0,0,0,0.005,0.060,0.180,0.350,0.510,0.640,
                   0.720,0.740,0.700,0.600,0.440,0.260,0.100,0.015,0,0,0,0],
        'summer': [0,0,0,0,0,0,0.020,0.100,0.240,0.410,0.570,0.710,
                   0.800,0.820,0.770,0.660,0.500,0.310,0.140,0.035,0,0,0,0],
        'autumn': [0,0,0,0,0,0,0.005,0.050,0.160,0.320,0.480,0.610,
                   0.690,0.710,0.670,0.560,0.410,0.230,0.080,0.008,0,0,0,0],
        'winter': [0,0,0,0,0,0,0,0.020,0.100,0.240,0.380,0.480,
                   0.540,0.520,0.460,0.350,0.230,0.110,0.030,0,0,0,0,0],
    },
    'III': {  # 三类: 华中/华东/华南, 年利用 1100-1400h (与原config.py一致)
        'spring': [0,0,0,0,0,0,0.003,0.040,0.140,0.280,0.420,0.530,
                   0.600,0.620,0.580,0.490,0.350,0.200,0.070,0.010,0,0,0,0],
        'summer': [0,0,0,0,0,0,0.015,0.080,0.200,0.350,0.500,0.630,
                   0.710,0.730,0.680,0.570,0.420,0.260,0.110,0.025,0,0,0,0],
        'autumn': [0,0,0,0,0,0,0.003,0.035,0.120,0.250,0.380,0.490,
                   0.560,0.570,0.530,0.440,0.310,0.170,0.060,0.005,0,0,0,0],
        'winter': [0,0,0,0,0,0,0,0.015,0.075,0.180,0.300,0.380,
                   0.430,0.410,0.360,0.270,0.180,0.080,0.020,0,0,0,0,0],
    },
    'IV': {  # 四类: 川渝黔, 年利用 900-1100h
        'spring': [0,0,0,0,0,0,0.002,0.030,0.100,0.220,0.340,0.440,
                   0.500,0.520,0.480,0.400,0.280,0.160,0.050,0.008,0,0,0,0],
        'summer': [0,0,0,0,0,0,0.010,0.060,0.160,0.280,0.390,0.500,
                   0.570,0.580,0.540,0.450,0.340,0.210,0.090,0.020,0,0,0,0],
        'autumn': [0,0,0,0,0,0,0.002,0.025,0.090,0.190,0.300,0.390,
                   0.440,0.450,0.420,0.350,0.240,0.130,0.045,0.004,0,0,0,0],
        'winter': [0,0,0,0,0,0,0,0.010,0.050,0.120,0.210,0.280,
                   0.310,0.300,0.260,0.190,0.130,0.060,0.015,0,0,0,0,0],
    },
}

# 各省水平面年总辐射量 (kWh/m2) — 来源: 文件04 §1.3
PROVINCE_RADIATION = {
    '西藏': 1819.8, '青海': 1747.2, '甘肃': 1627.7, '宁夏': 1611.0,
    '新疆': 1588.6, '内蒙古': 1571.6, '河北': 1537.5, '北京': 1527.6,
    '四川': 1499.9, '山西': 1488.3, '陕西': 1475.2, '云南': 1469.8,
    '山东': 1462.3, '河南': 1451.8, '江苏': 1438.2, '湖北': 1416.7,
    '安徽': 1409.5, '浙江': 1393.2, '湖南': 1373.6, '江西': 1357.6,
    '福建': 1349.3, '广东': 1329.4, '广西': 1317.1, '重庆': 1311.0,
    '贵州': 1289.9,
}

# 资源分区映射
RADIATION_TO_ZONE = {
    'I': (1750, float('inf')),    # >1750
    'II': (1400, 1750),            # 1400-1750
    'III': (1050, 1400),           # 1050-1400
    'IV': (0, 1050),               # <1050
}

# 服务区可用敷设区域参数 — 来源: 文件04 §2.3
SERVICE_AREA_AREA_PARAMS = {
    'small': {
        'building_roof_m2': (800, 1200),
        'parking_carport_m2': (2000, 3000),
        'slope_vacant_m2': (1500, 2500),
        'ramp_m2': (500, 1000),
        'total_available_m2': (5000, 7000),
        'pv_area_ratio': 0.65,      # 有效敷设比例
    },
    'medium': {
        'building_roof_m2': (1200, 2000),
        'parking_carport_m2': (3000, 5000),
        'slope_vacant_m2': (2500, 4000),
        'ramp_m2': (1000, 2000),
        'total_available_m2': (8000, 12000),
        'pv_area_ratio': 0.65,
    },
    'large': {
        'building_roof_m2': (2000, 3500),
        'parking_carport_m2': (5000, 8000),
        'slope_vacant_m2': (4000, 6000),
        'ramp_m2': (1500, 2500),
        'total_available_m2': (12000, 18000),
        'pv_area_ratio': 0.60,
    },
}

# 按气候分区的推荐配置 — 来源: 文件04 §3.1
ZONE_TYPICAL_CONFIG = {
    'I':   {'equiv_hours': (1600, 1800), 'pv_range_kwp': (500, 800),
            'annual_gen_wan': (80, 144)},
    'II':  {'equiv_hours': (1400, 1600), 'pv_range_kwp': (300, 600),
            'annual_gen_wan': (42, 96)},
    'III': {'equiv_hours': (1100, 1400), 'pv_range_kwp': (200, 500),
            'annual_gen_wan': (22, 70)},
    'IV':  {'equiv_hours': (900, 1100), 'pv_range_kwp': (150, 300),
            'annual_gen_wan': (13.5, 33)},
}

# 天气修正系数 (文件10)
WEATHER_TYPES = ['clear', 'partly_cloudy', 'cloudy', 'overcast', 'rain']
WEATHER_COEFF_MAP = {
    'clear': 1.00, 'partly_cloudy': 0.80,
    'cloudy': 0.55, 'overcast': 0.30, 'rain': 0.15,
}

# 各月典型天气天数 (三类资源区/华中)
MONTHLY_WEATHER_DAYS_BASE = {
    1:  {'clear': 10, 'partly_cloudy': 6, 'cloudy': 5, 'overcast': 5, 'rain': 5},
    2:  {'clear': 8,  'partly_cloudy': 7, 'cloudy': 5, 'overcast': 5, 'rain': 3},
    3:  {'clear': 10, 'partly_cloudy': 7, 'cloudy': 6, 'overcast': 5, 'rain': 3},
    4:  {'clear': 11, 'partly_cloudy': 7, 'cloudy': 6, 'overcast': 4, 'rain': 2},
    5:  {'clear': 12, 'partly_cloudy': 8, 'cloudy': 5, 'overcast': 4, 'rain': 2},
    6:  {'clear': 10, 'partly_cloudy': 7, 'cloudy': 6, 'overcast': 5, 'rain': 2},
    7:  {'clear': 14, 'partly_cloudy': 8, 'cloudy': 4, 'overcast': 3, 'rain': 2},
    8:  {'clear': 13, 'partly_cloudy': 7, 'cloudy': 5, 'overcast': 4, 'rain': 2},
    9:  {'clear': 12, 'partly_cloudy': 7, 'cloudy': 5, 'overcast': 4, 'rain': 2},
    10: {'clear': 14, 'partly_cloudy': 6, 'cloudy': 5, 'overcast': 4, 'rain': 2},
    11: {'clear': 11, 'partly_cloudy': 6, 'cloudy': 5, 'overcast': 5, 'rain': 3},
    12: {'clear': 10, 'partly_cloudy': 6, 'cloudy': 5, 'overcast': 5, 'rain': 5},
}


def get_zone_by_province(province):
    """根据省份获取资源区"""
    rad = PROVINCE_RADIATION.get(province, 1400)
    for zone, (lo, hi) in RADIATION_TO_ZONE.items():
        if lo <= rad < hi:
            return zone
    return 'III'


def get_zone_by_radiation(annual_radiation_kwh_m2):
    """根据年辐射量获取资源区"""
    for zone, (lo, hi) in RADIATION_TO_ZONE.items():
        if lo <= annual_radiation_kwh_m2 < hi:
            return zone
    return 'III'


def get_pv_coeff_annual(zone='III', weather_data=None):
    """获取指定资源区的8760h归一化出力系数

    Parameters
    ----------
    zone : str
        'I', 'II', 'III', 'IV'
    weather_data : list of str, optional
        每日天气类型 (len=365), 若None则使用默认天气分布

    Returns
    -------
    coeff : np.ndarray (8760,)
        逐时归一化系数
    """
    coeffs = PV_ZONE_COEFFICIENTS[zone]
    seasons = {1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring',
               5: 'spring', 6: 'summer', 7: 'summer', 8: 'summer',
               9: 'autumn', 10: 'autumn', 11: 'autumn', 12: 'winter'}
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if weather_data is None:
        weather_data = _generate_default_weather(len(coeffs) // 24)

    hourly = np.zeros(8760)
    hour_idx = 0

    for m in range(1, 13):
        season = seasons[m]
        days = days_in_month[m - 1]
        for d in range(days):
            day_idx = sum(days_in_month[:m-1]) + d
            weather = weather_data[day_idx] if day_idx < len(weather_data) else 'clear'
            wc = WEATHER_COEFF_MAP.get(weather, 1.0)
            for h in range(24):
                if hour_idx < 8760:
                    hourly[hour_idx] = coeffs[season][h] * wc
                    hour_idx += 1

    return hourly


def _generate_default_weather(n_days=365):
    """生成默认天气序列 (基于月度天气分布比例)"""
    rng = np.random.RandomState(42)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    weather_seq = []
    for m in range(1, 13):
        wd = MONTHLY_WEATHER_DAYS_BASE[m]
        types = list(wd.keys())
        probs = np.array([wd[t] for t in types], dtype=float)
        probs /= probs.sum()
        for _ in range(days_in_month[m - 1]):
            weather_seq.append(types[rng.choice(len(types), p=probs)])
    return weather_seq


def compute_service_area_pv_potential(area_size='medium', zone='III'):
    """计算服务区光伏装机潜力和年发电量

    返回详细评估结果字典
    """
    area_params = SERVICE_AREA_AREA_PARAMS[area_size]
    zone_config = ZONE_TYPICAL_CONFIG[zone]

    total_area = area_params['total_available_m2'][1]  # 上限
    effective_area = total_area * area_params['pv_area_ratio']

    # 单位面积功率估算 (215 Wp/m2, 考虑间距取150 Wp/m2)
    wp_per_m2 = 0.150  # kWp/m2
    max_pv_kwp = effective_area * wp_per_m2

    # 推荐装机范围
    equiv_hours_mid = np.mean(zone_config['equiv_hours'])
    recommended_pv_kwp = zone_config['pv_range_kwp']

    # 年发电量估算
    annual_gen_lo = recommended_pv_kwp[0] * equiv_hours_mid  # kWh
    annual_gen_hi = recommended_pv_kwp[1] * equiv_hours_mid

    # 日等效发电小时 (四季平均)
    coeffs = PV_ZONE_COEFFICIENTS[zone]
    daily_hours = {}
    for season in ['spring', 'summer', 'autumn', 'winter']:
        daily_hours[season] = sum(coeffs[season])

    return {
        'area_size': area_size,
        'zone': zone,
        'equiv_hours_range': zone_config['equiv_hours'],
        'equiv_hours_mean': equiv_hours_mid,
        'total_area_m2': total_area,
        'effective_area_m2': effective_area,
        'max_pv_kwp': max_pv_kwp,
        'recommended_pv_kwp_range': recommended_pv_kwp,
        'annual_generation_range_kwh': (annual_gen_lo, annual_gen_hi),
        'daily_equivalent_hours': daily_hours,
        'zone_config': zone_config,
        'area_params': area_params,
    }


def compare_all_zones(area_size='medium'):
    """四类资源区横向对比"""
    print("\n" + "=" * 65)
    print(f"四类资源区光伏潜力对比 — {area_size}服务区")
    print("=" * 65)
    print(f"{'资源区':<6} {'年利用h':>10} {'推荐装机kWp':>14} {'年发电万kWh':>14} "
          f"{'有效面积m2':>12}")

    for zone in ['I', 'II', 'III', 'IV']:
        pot = compute_service_area_pv_potential(area_size, zone)
        print(f"  {zone}类    {pot['equiv_hours_mean']:6.0f}-{pot['equiv_hours_range'][1]:4.0f}"
              f"   {pot['recommended_pv_kwp_range'][0]:4.0f}-{pot['recommended_pv_kwp_range'][1]:4.0f}"
              f"       {pot['annual_generation_range_kwh'][0]/1e4:4.0f}-{pot['annual_generation_range_kwh'][1]/1e4:4.0f}"
              f"       {pot['effective_area_m2']:6.0f}")

    # 按省份归类
    print(f"\n{'省份':<6} {'辐射量':>8} {'资源区':>6}")
    print("-" * 25)
    for prov, rad in sorted(PROVINCE_RADIATION.items(), key=lambda x: -x[1]):
        zone = get_zone_by_province(prov)
        print(f"  {prov:<4}  {rad:6.1f}    {zone}类")


class PVResourceAnalyzer:
    """光伏资源综合分析器

    用于给定坐标/省份/资源区下, 评估服务区光伏发电潜力,
    并生成可与仿真链对接的出力系数.
    """

    def __init__(self, zone='III', area_size='medium',
                 weather_data=None, pv_capacity_kwp=None):
        self.zone = zone
        self.area_size = area_size
        self.weather_data = weather_data
        self.coeffs = PV_ZONE_COEFFICIENTS[zone]
        self.potential = compute_service_area_pv_potential(area_size, zone)

        if pv_capacity_kwp is None:
            pv_capacity_kwp = np.mean(self.potential['recommended_pv_kwp_range'])
        self.pv_capacity_kwp = pv_capacity_kwp

    def get_hourly_coeff(self, month, season, weather='clear'):
        """获取指定月/季/天气的逐时系数"""
        wc = WEATHER_COEFF_MAP.get(weather, 1.0)
        return np.array(self.coeffs[season]) * wc

    def get_daily_generation(self, month, season, weather='clear'):
        """日发电量估算 (kWh)"""
        coeff = self.get_hourly_coeff(month, season, weather)
        return float(np.sum(coeff) * self.pv_capacity_kwp)

    def get_annual_generation(self):
        """年发电量估算 (kWh)"""
        coeff_8760 = get_pv_coeff_annual(self.zone, self.weather_data)
        return float(np.sum(coeff_8760) * self.pv_capacity_kwp / 365)

    def get_equivalent_hours(self):
        """年等效利用小时数"""
        annual_kwh = self.get_annual_generation()
        return annual_kwh / self.pv_capacity_kwp if self.pv_capacity_kwp > 0 else 0

    def print_potential_summary(self):
        """打印光伏潜力评估摘要"""
        p = self.potential
        print("\n" + "=" * 55)
        print(f"光伏发电潜力评估 [{self.area_size}服务区 / {self.zone}类资源区]")
        print("=" * 55)
        print(f"  有效敷设面积: {p['effective_area_m2']:.0f} m2")
        print(f"  最大可装机: {p['max_pv_kwp']:.0f} kWp")
        print(f"  推荐装机范围: {p['recommended_pv_kwp_range'][0]:.0f} - "
              f"{p['recommended_pv_kwp_range'][1]:.0f} kWp")
        print(f"  当前装机: {self.pv_capacity_kwp:.0f} kWp")
        print(f"  年等效利用小时: {p['equiv_hours_mean']:.0f} h")
        print(f"  年发电量预测: {self.get_annual_generation()/1e4:.1f} 万kWh")
        print(f"  日等效发电小时:")
        for season, hours in p['daily_equivalent_hours'].items():
            print(f"    {season}: {hours:.2f} h")

        # CO2减排
        carbon = self.get_annual_generation() / 1000 * 0.5703
        print(f"  年碳减排估算: {carbon:.1f} tCO2")


def self_test():
    """模块自测"""
    print("=" * 55)
    print("光伏资源分析模块 — 自测")
    print("=" * 55)

    # 1. 全国辐射量概览
    print("\n[1] 全国太阳能资源分布:")
    rads = list(PROVINCE_RADIATION.values())
    print(f"  最高: 西藏 {PROVINCE_RADIATION['西藏']:.1f} kWh/m2")
    print(f"  最低: 贵州 {PROVINCE_RADIATION['贵州']:.1f} kWh/m2")
    print(f"  全国均值: {np.mean(rads):.1f} kWh/m2")

    # 2. 四类资源区对比
    print("\n[2] 四类资源区服务区光伏潜力:")
    compare_all_zones('medium')

    # 3. 三类资源区详细分析 (当前项目区域)
    print("\n[3] 三类资源区中型服务区详细评估:")
    analyzer = PVResourceAnalyzer(zone='III', area_size='medium', pv_capacity_kwp=1231)
    analyzer.print_potential_summary()

    # 4. 不同天气日出力曲线对比
    print("\n[4] 三类资源区夏季不同天气日出力 (1231 kWp):")
    for weather in ['clear', 'partly_cloudy', 'cloudy', 'overcast', 'rain']:
        coeff = analyzer.get_hourly_coeff(7, 'summer', weather)
        daily = float(np.sum(coeff) * 1231)
        print(f"  {weather:>14s}: {daily:.0f} kWh/天 ({WEATHER_COEFF_MAP[weather]:.0%}×基准)")

    # 5. 高速公路光伏潜力宏观数据
    print("\n[5] 高速公路光伏潜力宏观数据 (文件04):")
    print(f"  全国高速公路总里程: 143,684 km")
    print(f"  高速总面积: 3,957 km2")
    print(f"  服务区数量: ~3,225个")
    print(f"  高速路域光伏装机潜力: 700.85 GW")
    print(f"  高速路域年发电潜力: 629.06 TWh")
    print(f"  服务区年光伏潜力: 66.37 TWh")
    print(f"  服务区可安装容量: 51.85 GW")


if __name__ == '__main__':
    self_test()
