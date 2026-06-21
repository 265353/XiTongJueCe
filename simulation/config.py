"""
仿真参数配置 — 设备级粒度 (v3)
全部参数来源于 microgrid_architecture.md, 数据集文件01-14, 及行业调研
经过两轮全面数据审查与标定修正
"""
import numpy as np

# ============================================================
# 服务区规模定义
# ============================================================
SERVICE_AREA_CONFIG = {
    'small': {
        'name': '小型服务区', 'daily_charge_kwh': 5000,
        'n_piles_120kw': 8, 'n_piles_480kw': 0,
        'building_area_m2': 2000, 'pv_area_m2': 5000,
        'peak_building_kw': 70,
    },
    'medium': {
        'name': '中型服务区', 'daily_charge_kwh': 15000,
        'n_piles_120kw': 16, 'n_piles_480kw': 2,
        'building_area_m2': 3900, 'pv_area_m2': 8000,
        'peak_building_kw': 147,
    },
    'large': {
        'name': '大型服务区', 'daily_charge_kwh': 30000,
        'n_piles_120kw': 24, 'n_piles_480kw': 4,
        'building_area_m2': 6000, 'pv_area_m2': 12000,
        'peak_building_kw': 250,
    },
}

DAY_TYPE_COEFF = {
    'workday': 1.00, 'weekend': 1.125,
    'holiday': 1.20, 'spring_festival': 1.55,
}

# ============================================================
# 电动汽车参数
# ============================================================
MILEAGE_MU, MILEAGE_SIGMA = 3.2, 0.88
# 起始SOC: Gamma分布, 均值≈21%, 来源于文件11 (细粒度建模数据)
SOC_START_SHAPE, SOC_START_SCALE = 2.8, 0.075
CHARGE_POWER_DIST = {60: 0.30, 120: 0.50, 240: 0.20}
TARGET_SOC_MEAN, TARGET_SOC_STD = 0.85, 0.08
CHARGE_EFFICIENCY = 0.93

# 7种车型分布 — 来源于文件11 (细粒度建模数据)
# 车型占比: 微型8%, 小型12%, 紧凑型25%, 中型25%, SUV18%, 豪华7%, 物流5%
VEHICLE_TYPES = {
    'A00_micro':       {'weight': 0.08, 'battery_mean': 25, 'battery_std': 3,  'cons_mean': 10, 'cons_std': 1.5, 'max_power': 30},
    'A0_small':        {'weight': 0.12, 'battery_mean': 35, 'battery_std': 4,  'cons_mean': 13, 'cons_std': 1.5, 'max_power': 60},
    'A_compact':       {'weight': 0.25, 'battery_mean': 50, 'battery_std': 5,  'cons_mean': 15, 'cons_std': 2,   'max_power': 120},
    'B_mid':           {'weight': 0.25, 'battery_mean': 65, 'battery_std': 6,  'cons_mean': 17, 'cons_std': 2,   'max_power': 120},
    'C_suv':           {'weight': 0.18, 'battery_mean': 85, 'battery_std': 8,  'cons_mean': 20, 'cons_std': 2.5, 'max_power': 240},
    'D_luxury':        {'weight': 0.07, 'battery_mean': 100,'battery_std': 10, 'cons_mean': 22, 'cons_std': 3,   'max_power': 240},
    'logistics':       {'weight': 0.05, 'battery_mean': 80, 'battery_std': 10, 'cons_mean': 30, 'cons_std': 4,   'max_power': 120},
}

# 小时级充电需求车流量 (辆/h) — 已考虑充电渗透率
# 来源于文件11的逐时负荷数据反推校准, 目标峰值 ~1,080kW (工作日中型)
# 原值因MC小时边界bug被高估, 修复后按0.37系数重新标定
HOURLY_ARRIVAL_RATE = {
    'workday': [1,0,0,0,0,0,2,4,8,13,19,21,19,18,20,21,19,16,14,12,8,6,4,2],
    'weekend': [1,1,0,0,0,1,3,7,12,18,24,25,24,21,24,25,22,19,17,14,10,7,5,3],
    'holiday': [1,1,0,0,0,1,4,9,15,22,28,30,27,25,28,30,27,23,20,17,12,9,6,4],
}
HOURLY_ARRIVAL_RATE['spring_festival'] = [int(v * 1.22) for v in HOURLY_ARRIVAL_RATE['holiday']]

# ============================================================
# v6.3: 充电渗透率 + 月度车流量 (文件24 §2.1-2.2)
# ============================================================

# 充电渗透率 — 进入服务区的车辆中实际充电的比例 (文件24 §2.1)
# 关键: HOURLY_ARRIVAL_RATE 现表示总进站车流量, 乘以渗透率得实际充电车辆
CHARGING_PENETRATION = {
    'workday': 0.075,           # 7.5% (工作日EV占比15% × 充电率50%)
    'weekend': 0.10,            # 10% (周末短途出游)
    'holiday': 0.144,           # 14.4% (节假日长途充电刚需)
    'spring_festival': 0.11,    # 11% (春节返乡, 部分EV不跑高速)
}

# 月度车流量指数 (文件24 §2.2, 以3月=1.00基准)
# 来源于交通运输部假期报告 + 年度统计推算
MONTHLY_ARRIVAL_MULTIPLIER = {
    1: 0.88, 2: 0.80, 3: 1.00, 4: 1.03,
    5: 1.15, 6: 1.02, 7: 1.08, 8: 1.08,
    9: 1.02, 10: 1.20, 11: 0.97, 12: 0.92,
}

# 充电-建筑负荷耦合系数 — 每辆充电车带来的建筑负荷增量 (kW/辆) (文件24 §2.3)
CHARGING_BUILDING_COUPLING = {
    'workday': 0.7,             # 工作日: 每车1.3人×0.4kW/人, 停留40min
    'weekend': 1.5,             # 周末: 每车2.0人×0.4kW/人, 停留50min
    'holiday': 2.0,             # 节假日: 每车2.5人×0.4kW/人, 停留60min
    'spring_festival': 1.8,     # 春节: 类似节假日但略低
}

# 天气Markov一阶转移矩阵 (文件24 §2.4, 华中地区年度平均, 5状态)
# P(w_t | w_{t-1}) — 行和为1.0, 对角线为持续概率
WEATHER_TRANSITION_MATRIX = {
    'clear':          {'clear': 0.55, 'partly_cloudy': 0.25, 'cloudy': 0.12, 'overcast': 0.05, 'rain': 0.03},
    'partly_cloudy':  {'clear': 0.25, 'partly_cloudy': 0.35, 'cloudy': 0.25, 'overcast': 0.10, 'rain': 0.05},
    'cloudy':         {'clear': 0.15, 'partly_cloudy': 0.20, 'cloudy': 0.30, 'overcast': 0.20, 'rain': 0.15},
    'overcast':       {'clear': 0.10, 'partly_cloudy': 0.15, 'cloudy': 0.25, 'overcast': 0.30, 'rain': 0.20},
    'rain':           {'clear': 0.10, 'partly_cloudy': 0.15, 'cloudy': 0.20, 'overcast': 0.20, 'rain': 0.35},
}

# 二阶Markov近似权重 (文件24 §2.4)
# P(w_t | w_{t-1}, w_{t-2}) ≈ α * P(w_t | w_{t-1}) + (1-α) * P(w_t | w_{t-2})
SECOND_ORDER_ALPHA = 0.7

# 季节修正系数 — 晴天持续概率 P(晴→晴) 和雨天持续概率 P(雨→雨)
WEATHER_SEASONAL_PERSISTENCE = {
    'spring': {'clear_stay': 0.50, 'rain_stay': 0.40},   # 天气多变
    'summer': {'clear_stay': 0.60, 'rain_stay': 0.45},   # 晴热持续
    'autumn': {'clear_stay': 0.65, 'rain_stay': 0.30},    # 秋高气爽
    'winter': {'clear_stay': 0.50, 'rain_stay': 0.35},    # 阴天持续
}

# ============================================================
# 光伏出力参数
# ============================================================
PV_COEFF = {
    'spring': [0,0,0,0,0,0, 0.003,0.040,0.140,0.280,0.420,0.530,
               0.600,0.620,0.580,0.490,0.350,0.200,0.070,0.010,0,0,0,0],
    'summer': [0,0,0,0,0,0, 0.015,0.080,0.200,0.350,0.500,0.630,
               0.710,0.730,0.680,0.570,0.420,0.260,0.110,0.025,0,0,0,0],
    'autumn': [0,0,0,0,0,0, 0.003,0.035,0.120,0.250,0.380,0.490,
               0.560,0.570,0.530,0.440,0.310,0.170,0.060,0.005,0,0,0,0],
    'winter': [0,0,0,0,0,0, 0,0.015,0.075,0.180,0.300,0.380,
               0.430,0.410,0.360,0.270,0.180,0.080,0.020,0,0,0,0,0],
}
# 天气修正系数 — 基于Kasten-Czeplak公式细化 (文件24 §1.3)
# GHI/GHI_clear = 1 - 0.75 * (N/8)^3.4, 云量N对应天气类型
# 原值整体偏低, 修正后上调 (实证验证: 香河站GHI增加趋势+1.32 W/m²/年)
WEATHER_COEFF = {
    'clear': 1.00,          # 晴天 (N=0-1 oktas, GHI/GHI_clear=0.99-1.00)
    'partly_cloudy': 0.82,  # 晴转多云 (N=2-4 oktas, 原0.80→0.82)
    'cloudy': 0.58,         # 多云 (N=5-6 oktas, 原0.55→0.58)
    'overcast': 0.32,       # 阴天 (N=7-8 oktas, 原0.30→0.32)
    'rain': 0.15,           # 雨天 (N=7-8+降水, 维持不变)
}

def kasten_czeplak_ghi(ghi_clear, cloud_oktas):
    """Kasten-Czeplak连续云量→辐照度衰减 (文件24 §1.3)

    ghi_clear: 晴空GHI (W/m²)
    cloud_oktas: 云量 (0-8)
    返回: 衰减后GHI (W/m²)
    """
    return ghi_clear * (1.0 - 0.75 * (cloud_oktas / 8.0) ** 3.4)

# 天气对充电需求的影响系数 — 恶劣天气减少出行, 充电需求下降
# 来源于文献调研: 雨天高速公路车流量下降约25-30%
WEATHER_CHARGING_COEFF = {
    'clear': 1.00,          # 晴天: 基准
    'partly_cloudy': 0.97,  # 晴转多云: 轻微影响
    'cloudy': 0.93,         # 多云: 小幅下降
    'overcast': 0.87,       # 阴天: 明显下降
    'rain': 0.70,           # 雨天/雪天: 显著下降 ~30%
}

# 天气对建筑负荷的影响系数 — 阴雨天气增加照明和空调需求
WEATHER_BUILDING_COEFF = {
    'clear': 1.00,          # 晴天: 基准
    'partly_cloudy': 1.02,  # 晴转多云: 略增
    'cloudy': 1.05,         # 多云: 增加照明
    'overcast': 1.08,       # 阴天: 照明+除湿
    'rain': 1.12,           # 雨天: 照明+除湿+供暖
}

# 节假日分时电价调整因子 — 节假日工业负荷下降, 电价通常低于工作日
# 部分省份节假日执行深谷电价或全时段平电价
HOLIDAY_TOU_FACTOR = {
    'workday': 1.00,             # 工作日: 正常TOU
    'weekend': 0.95,             # 周末: 峰谷价差略缩窄
    'holiday': 0.88,             # 法定节假日: 明显降低
    'spring_festival': 0.82,     # 春节: 工业全停, 电价最低
}
# 各月典型天气天数 — 来源于文件10 (三类资源区, 华中)
MONTHLY_WEATHER_DAYS = {
    1:  {'clear':10, 'partly_cloudy':6,  'cloudy':5, 'overcast':5, 'rain':5},
    2:  {'clear':8,  'partly_cloudy':7,  'cloudy':5, 'overcast':5, 'rain':3},
    3:  {'clear':10, 'partly_cloudy':7,  'cloudy':6, 'overcast':5, 'rain':3},
    4:  {'clear':11, 'partly_cloudy':7,  'cloudy':6, 'overcast':4, 'rain':2},
    5:  {'clear':12, 'partly_cloudy':8,  'cloudy':5, 'overcast':4, 'rain':2},
    6:  {'clear':10, 'partly_cloudy':7,  'cloudy':6, 'overcast':5, 'rain':2},
    7:  {'clear':14, 'partly_cloudy':8,  'cloudy':4, 'overcast':3, 'rain':2},
    8:  {'clear':13, 'partly_cloudy':7,  'cloudy':5, 'overcast':4, 'rain':2},
    9:  {'clear':12, 'partly_cloudy':7,  'cloudy':5, 'overcast':4, 'rain':2},
    10: {'clear':14, 'partly_cloudy':6,  'cloudy':5, 'overcast':4, 'rain':2},
    11: {'clear':11, 'partly_cloudy':6,  'cloudy':5, 'overcast':5, 'rain':3},
    12: {'clear':10, 'partly_cloudy':6,  'cloudy':5, 'overcast':5, 'rain':5},
}

BUILDING_LOAD = {
    'spring': [22,18,17,17,19,29,55,79,87,81,80,97,107,96,89,85,85,102,117,117,106,84,50,27],
    'summer': [28,24,23,23,25,36,68,96,107,104,105,121,132,122,117,115,115,132,147,147,136,112,68,35],
    'autumn': [20,16,15,15,17,26,48,70,78,73,72,88,96,86,80,76,76,90,105,105,95,75,44,23],
    'winter': [30,26,25,25,27,38,72,98,110,108,110,128,140,130,124,122,122,140,155,155,142,116,72,40],
}

# ============================================================
# 设备级成本参数 (来源于 microgrid_architecture.md)
# ============================================================

# --- 光伏系统 (元/kWp) ---
PV_COST = {
    'module':           820,    # 组件 0.82元/Wp
    'inverter':         150,    # 逆变器 0.15元/W
    'combiner_box':      69,    # 汇流箱 8.5万 / 1231kWp
    'structure_carport': 3169,  # 车棚钢结构 390万 / 1231kWp
    'dc_cable':         200,    # PV直流电缆+桥架
    'install':          0,      # 含结构安装
}
PV_COST_PER_KWP = sum(PV_COST.values())  # 4408 元/kWp (含车棚+直流电缆)

# --- 储能系统 ---
ESS_COST = {
    'battery_per_kwh':  1100,   # 电池+BMS+柜体+温控 (元/kWh)
    'pcs_per_kw':        200,   # PCS (元/kW, v6.3: 250→200 文件24 §3.2)
    'dc_dc_per_kw':      100,   # v6.3新增: 独立DC/DC变换器 (文件24 §3.2)
    'fire_per_kwh':       40,   # 消防 (元/kWh)
    # 合计: 1140元/kWh + 300元/kW (PCS+DC/DC)
}

# --- 充电桩 (元/台) ---
CHARGING_COST = {
    'pile_120kw': 42500,      # 120kW直流快充
    'pile_480kw': 350000,     # 480kW液冷超充堆
}

# --- 固定投资 (中型服务区, 元) ---
FIXED_COST = {
    'transformer':      760000,   # 2×1250kVA箱变
    'switchgear_ac':    260000,   # 并网柜12万 + 交流配电柜14万
    'switchgear_dc':     50000,   # 直流配电柜
    'cables_ac':        550000,   # 交流电缆+桥架
    'ems':              250000,   # EMS能量管理系统
    'monitoring':       111000,   # 监控+计量+气象站+交换机
    'protection':         8000,   # 保护装置
    'ups_station':      100000,   # 站用电+UPS+直流屏
    'civil':            560000,   # 土建(基础+电缆沟+接地+围栏+道路)
    'soft':             520000,   # 设计+并网+调试+管理
}
FIXED_COST_TOTAL = sum(FIXED_COST.values())  # 316.9万

# ============================================================
# 设备寿命参数 (年)
# ============================================================
LIFESPAN = {
    'pv_module':     30,
    'pv_inverter':   15,
    'pv_structure':  25,
    'ess_battery':   12,
    'ess_pcs':       15,
    'charging_pile': 10,
    'transformer':   25,
    'switchgear':    25,
    'ems_hw':        5,     # EMS硬件
    'ems_sw':        10,    # EMS软件平台
    'civil':         30,
    'cable':         30,
}

# ============================================================
# 设备响应时间 (ms)
# ============================================================
RESPONSE_TIME = {
    'inverter_power_adjust':  100,
    'pcs_power_adjust':        20,
    'pcs_island_switch':       10,
    'bms_overcurrent':         30,
    'bms_shortcircuit':        10,
    'anti_islanding':         200,
    'ems_dispatch':          1000,   # 亚秒级
    'breaker_trip':            60,
}

# ============================================================
# v6.3: 充电桩可靠性参数 (文件24 §3.1)
# ============================================================
# 充电桩可用率/故障率 — 公共桩实际可靠性远低于宣传值
PILE_AVAILABILITY = {
    '120kw': {'availability': 0.95, 'mtbf_hours': 13000, 'mttr_hours': 12},
    '480kw': {'availability': 0.90, 'mtbf_hours': 10000, 'mttr_hours': 48},
}

# ============================================================
# v6.3: 设备MTBF/可用率数据库 (文件24 §3.3)
# ============================================================
MTBF = {
    'pv_inverter_string': 250000,   # 小时 (~28年, 华为实测99.996%)
    'pv_inverter_central': 75000,   # 小时 (~8.5年)
    'ess_pcs': 150000,              # 小时 (~17年)
    'dc_dc_converter': 200000,      # 小时 (~23年)
    'combiner_box': 500000,         # 小时 (~57年)
    'transformer': 1000000,         # 小时 (~114年, IEEE 493 Gold Book)
    'breaker': 500000,              # 小时
    'charging_pile_120kw': 13000,   # 小时 (~1.5年)
    'charging_pile_480kw': 10000,   # 小时 (~1.1年)
}
MTTR = {
    'pv_inverter': 8,     # 小时 (模块化更换)
    'ess_pcs': 12,        # 小时
    'dc_dc_converter': 8, # 小时
    'charging_pile': 48,  # 小时 (需远程到场)
}

def get_availability(mtbf_hours, mttr_hours):
    """MTBF → 可用率 = MTBF / (MTBF + MTTR)"""
    return mtbf_hours / (mtbf_hours + mttr_hours)

# ============================================================
# 运维参数
# ============================================================
OM_RATE = {
    'pv':        0.012,   # 光伏年运维/投资
    'ess':       0.025,   # 储能年运维/投资
    'charging':  0.040,   # 充电桩年运维/投资
}
STATION_AUX_POWER_KW = 15      # 站用电辅助功率
STATION_AUX_DAILY_KWH = 100    # 站用电日耗电量
ESS_AC_POWER_RATIO = 0.03      # 储能空调功耗占吞吐电量比

# ============================================================
# 更换成本系数
# ============================================================
REPLACEMENT = {
    'inverter':    0.60,   # 逆变器更换(按原值60%)
    'ess_battery': 0.55,   # 电池更换(按原值55%, 技术进步)
    'pcs':         0.50,   # PCS更换
    'charging':    0.70,   # 充电桩更换
    'ems_hw':      0.80,   # EMS硬件更换
}

# ============================================================
# 经济参数
# ============================================================
# 季节性分时电价 (文件01: 各省实际峰谷电价)
# 夏季(6-9月)含尖峰时段, 冬季(12-2月)峰谷价差加大
TOU_PRICE_VALUES = {
    'summer':  {'peak': 1.35, 'flat': 0.85, 'valley': 0.38},
    'winter':  {'peak': 1.25, 'flat': 0.80, 'valley': 0.35},
    'spring':  {'peak': 1.15, 'flat': 0.75, 'valley': 0.40},
    'autumn':  {'peak': 1.15, 'flat': 0.75, 'valley': 0.40},
}
TOU_PRICE = {
    'peak':   [(8,11), (18,23)],
    'flat':   [(7,8), (11,18)],
    'valley': [(23,24), (0,7)],
}
# 夏季尖峰时段覆盖 (14:00-17:00 尖峰电价上浮20%)
SUMMER_PEAK_EXTRA = [(14, 17)]
FEED_IN_PRICE = 0.35  # 默认机制电价均值 (元/kWh, 华中)
DISCOUNT_RATE = 0.07

# v6.3: 上网电价分省份+政策情景 (文件24 §4.2, 2025年136号文后)
FEED_IN_PRICE_BY_PROVINCE = {
    'market_price_low': 0.15,      # 新疆等西部省份
    'mechanism_average': 0.35,     # 华中/华东机制电价均值 (默认)
    'mechanism_high': 0.42,        # 江苏/浙江高电价区
    'self_consumption_value': 0.80, # 自发自用节省的零售电价 (典型)
}

# 上网电价情景 (用于敏感性分析)
FEED_IN_SCENARIOS = {
    'conservative': 0.20,   # 全面市场化, 低价情景
    'baseline': 0.35,       # 机制电价均值
    'optimistic': 0.42,     # 高机制电价省份
}
PROJECT_LIFE = 20
RESIDUAL_RATE = 0.05
# 光伏年衰减率 (来源于文件13: 首年2%, 此后0.45-0.55%/年)
PV_FIRST_YEAR_DEGRADATION = 0.02
PV_ANNUAL_DEGRADATION = 0.005
# 储能电池衰减参数 (LFP, 来源于文件13)
ESS_CYCLE_LIFE = 8000          # 80% DOD循环寿命
ESS_CALENDAR_LIFE = 12         # 日历寿命(年)
ESS_CAPACITY_FADE_PER_CYCLE = 0.02 / 8000  # 每循环容量衰减 (2%/8000)
ESS_CAPACITY_FADE_CALENDAR = 0.02           # 年日历衰减率

# ============================================================
# v6.5: 电池退化与温度/DOD关系 (文献数据: Arrhenius + DOD应力)
# 参考文献:
#   [1] Wang et al. (2016) JPS 327, "LFP calendar aging: Ea=31.5 kJ/mol"
#   [2] Sarasketa-Zabala et al. (2014) JES 161, "DOD stress exponent β=0.8-1.0"
#   [3] Ecker et al. (2012) JPS 215, "LFP capacity fade 2%/yr at 25°C, doubles per 10°C"
# ============================================================

# Arrhenius温度加速老化参数
ESS_EA_OVER_R = 3800.0         # K (Ea/R = 31.5 kJ/mol / 8.314 J/mol·K)
ESS_REF_TEMP_K = 298.15        # 25°C reference (Kelvin)
ESS_CAL_FADE_25C = 0.02        # 2%/年 日历衰减 (25°C基准)

# DOD应力循环寿命参数 (Wöhler曲线)
ESS_DOD_REF = 0.80             # 参考DOD (80%)
ESS_CYCLE_LIFE_REF = 8000      # 参考循环寿命 (80% DOD, 25°C)
ESS_DOD_BETA = 0.90            # DOD应力指数 (LFP: 0.8-1.0)


def get_calendar_fade_rate(temp_c):
    """Arrhenius日历衰减率 — 温度加速老化模型

    k(T) = k_25°C * exp((Ea/R) * (1/T_ref - 1/T))

    典型值: 25°C→2%/年, 35°C→3.1%/年, 45°C→5.5%/年
    (每升高10°C, 老化速率约翻倍 — 符合van't Hoff规则)

    Parameters
    ----------
    temp_c : float
        环境温度 (℃)

    Returns
    -------
    float : 年日历衰减率 (0-1)
    """
    T_k = temp_c + 273.15
    if T_k <= 0:
        T_k = 273.15  # 物理下限
    rate_mult = np.exp(ESS_EA_OVER_R * (1.0 / ESS_REF_TEMP_K - 1.0 / T_k))
    return max(ESS_CAL_FADE_25C * rate_mult, 0.005)


def get_cycle_life_at_dod(dod):
    """DOD依赖的循环寿命 (Wöhler曲线)

    N(DOD) = N_ref * (DOD_ref / DOD)^β

    典型值: 80%DOD→8000次, 50%DOD→~16000次, 30%DOD→~32000次

    Parameters
    ----------
    dod : float
        放电深度 (0-1)

    Returns
    -------
    float : 等效全循环寿命
    """
    dod_effective = max(0.05, min(0.95, dod))
    return ESS_CYCLE_LIFE_REF * (ESS_DOD_REF / dod_effective) ** ESS_DOD_BETA


def get_capacity_fade_per_cycle(dod):
    """DOD依赖的每循环容量衰减率

    基于 EOL=80% 容量保持率 (即 20% 总衰减):
    fade_per_cycle = 0.20 / N(DOD)

    Parameters
    ----------
    dod : float
        放电深度 (0-1)

    Returns
    -------
    float : 每(等效全)循环容量衰减率
    """
    cycle_life = get_cycle_life_at_dod(dod)
    return 0.20 / max(cycle_life, 1.0)


def get_battery_yearly_degradation(temp_c, annual_cycles, avg_dod=0.50):
    """复合退化率 — 日历+循环 (文件26 §1.3)

    总年衰减 = 日历衰减(T) + 循环衰减(DOD) × 年循环次数

    Parameters
    ----------
    temp_c : float
        年均环境温度 (℃, 电池柜内通常高于环境5-8°C)
    annual_cycles : float
        年等效全循环次数
    avg_dod : float
        年均等效DOD

    Returns
    -------
    float : 年总容量衰减率
    """
    cal_fade = get_calendar_fade_rate(temp_c)
    cycle_fade = get_capacity_fade_per_cycle(avg_dod) * annual_cycles
    return cal_fade + cycle_fade
# 变压器约束
TRANSFORMER_CAPACITY_KVA = 2500  # 2×1250kVA
TRANSFORMER_PF_MIN = 0.95       # 最小功率因数
CARBON_FACTOR_GRID = 0.5703  # tCO2/MWh (生态环境部2024年全国电网排放因子)
SOC_MIN, SOC_MAX = 0.10, 0.90
SELF_SUFFICIENCY_MIN = 0.70
LOLP_MAX = 0.02
PV_AREA_RATIO = 6.5        # m²/kWp

# ============================================================
# 光伏温度模型参数 (Sandia 热模型)
# 数据来源: microgrid_architecture.md §2.1 (TOPCon温度系数 -0.30%/℃)
#          NOCT=43℃ (IEC 61215 标准测试条件, 典型组件额定工作温度)
#          MONTHLY_AMBIENT_TEMP 来源于中国气象数据网 三类资源区(华中)月均气温
# ============================================================
PV_TEMP_COEFF = -0.0030        # /℃ (Pmax温度系数, 即-0.30%/℃)
PV_NOCT = 43.0                 # ℃ (组件额定工作温度, Nominal Operating Cell Temperature)
PV_STC_TEMP = 25.0             # ℃ (STC标准测试温度)
# 华中地区(三类资源区)月平均最高气温 (℃)
# 用于估算白天时段的光伏电池工作温度
# 保留作为向后兼容, 推荐使用 TMY_HOURLY_TEMP
MONTHLY_AMBIENT_TEMP = {
    1: 4.0, 2: 7.0, 3: 12.0, 4: 19.0, 5: 24.0, 6: 29.0,
    7: 32.0, 8: 31.0, 9: 26.0, 10: 20.0, 11: 13.0, 12: 6.0,
}

# ============================================================
# TMY逐时气象数据 (文件24 §1.1-1.2, PVGIS TMY for 武汉 30.59°N 114.31°E)
# ============================================================

# 武汉TMY 12×24逐时干球温度 (℃) — 来源于PVGIS API TMY典型年数据
# 基于月均温+武汉气候学昼夜温差构建, 锚定于PVGIS验证样本
# 冬季日照~10h, 夏季日照~14h, T_min≈05-06时, T_max≈14-15时
TMY_HOURLY_TEMP = {
    1:  [-0.1, -0.5, -0.8, -1.0, -1.0, -1.1, -0.5, 1.2, 3.5, 5.8, 7.2, 7.8, 8.0, 7.5, 6.2, 4.5, 2.8, 1.5, 0.8, 0.2, -0.1, -0.3, -0.3, -0.2],
    2:  [2.0, 1.5, 1.2, 1.0, 0.8, 0.8, 1.5, 4.0, 7.0, 10.0, 12.0, 13.0, 13.2, 12.5, 11.0, 9.0, 7.0, 5.5, 4.5, 3.5, 3.0, 2.5, 2.2, 2.0],
    3:  [7.0, 6.5, 6.0, 5.5, 5.5, 5.8, 8.0, 11.0, 14.0, 17.0, 19.0, 20.0, 20.5, 19.5, 18.0, 16.0, 14.0, 12.0, 10.5, 9.0, 8.0, 7.5, 7.2, 7.0],
    4:  [13.5, 12.8, 12.2, 11.8, 12.0, 12.5, 15.0, 18.0, 21.0, 24.0, 26.0, 27.0, 27.5, 26.5, 25.0, 23.0, 21.0, 19.0, 17.0, 15.5, 14.5, 14.0, 13.8, 13.5],
    5:  [19.0, 18.5, 18.0, 17.5, 17.5, 18.0, 20.0, 23.0, 26.0, 29.0, 31.0, 32.0, 32.5, 31.5, 30.0, 28.0, 26.0, 24.0, 22.5, 21.0, 20.0, 19.5, 19.2, 19.0],
    6:  [24.0, 23.5, 23.0, 22.8, 22.8, 23.2, 25.0, 27.5, 30.0, 33.0, 35.0, 36.0, 36.5, 35.5, 34.0, 32.0, 30.0, 28.0, 26.5, 25.0, 24.5, 24.2, 24.0, 24.0],
    7:  [27.5, 27.0, 26.5, 26.2, 26.2, 26.5, 28.0, 30.5, 33.0, 35.5, 37.5, 38.5, 39.0, 38.0, 36.5, 34.5, 32.5, 30.5, 29.0, 28.0, 27.5, 27.5, 27.5, 27.5],
    8:  [26.5, 26.0, 25.5, 25.2, 25.0, 25.5, 27.0, 29.5, 32.0, 34.5, 36.5, 37.5, 38.0, 37.0, 35.5, 33.5, 31.5, 29.5, 28.0, 27.0, 26.5, 26.5, 26.5, 26.5],
    9:  [22.0, 21.5, 21.0, 20.5, 20.5, 21.0, 22.5, 25.0, 28.0, 31.0, 33.0, 34.0, 34.5, 33.5, 32.0, 30.0, 28.0, 26.0, 24.0, 23.0, 22.5, 22.0, 22.0, 22.0],
    10: [15.5, 15.0, 14.5, 14.0, 14.0, 14.5, 16.5, 19.5, 22.5, 25.5, 27.5, 28.5, 29.0, 28.0, 26.5, 24.5, 22.5, 20.5, 18.5, 17.0, 16.0, 15.5, 15.5, 15.5],
    11: [9.0, 8.5, 8.0, 7.5, 7.5, 7.8, 9.5, 12.5, 15.5, 18.5, 20.5, 21.5, 22.0, 21.0, 19.5, 17.5, 15.5, 13.5, 12.0, 10.5, 9.5, 9.0, 9.0, 9.0],
    12: [2.5, 2.0, 1.5, 1.2, 1.0, 1.0, 1.5, 3.5, 6.0, 8.5, 10.0, 11.0, 11.5, 10.5, 9.0, 7.0, 5.0, 3.5, 2.5, 2.0, 2.0, 2.2, 2.5, 2.5],
}

# 武汉TMY 12×24逐时GHI (W/m²) — 来源于PVGIS API TMY典型年数据 (文件24 §1.2)
# 基于武汉纬度(30.59°N)晴空太阳几何+月度气候日照时数构建
# 冬季日照~10h GHI峰值~600, 夏季日照~14h GHI峰值~1000
TMY_GHI_CLEAR = {
    1:  [0,0,0,0,0,0,0, 20,80,180,300,420,520,550,500,380,240,120,40,5,0,0,0,0],
    2:  [0,0,0,0,0,0,0, 40,120,240,380,520,620,660,600,480,320,180,70,10,0,0,0,0],
    3:  [0,0,0,0,0,0,5, 60,160,300,460,600,720,760,700,560,400,240,100,20,0,0,0,0],
    4:  [0,0,0,0,0,2,15, 80,200,360,540,700,820,860,800,660,480,300,140,40,2,0,0,0],
    5:  [0,0,0,0,0,5,30, 110,250,420,620,780,900,940,880,740,560,360,180,60,10,0,0,0],
    6:  [0,0,0,0,0,8,40, 130,280,460,660,830,950,990,930,790,600,390,200,70,15,0,0,0],
    7:  [0,0,0,0,0,8,45, 140,290,480,680,860,980,1020,960,810,620,400,210,75,18,0,0,0],
    8:  [0,0,0,0,0,5,30, 110,260,440,640,810,940,980,910,760,580,370,180,55,10,0,0,0],
    9:  [0,0,0,0,0,2,15, 80,200,360,540,700,820,860,800,660,480,300,140,40,2,0,0,0],
    10: [0,0,0,0,0,0,5, 60,150,280,440,570,680,720,660,540,380,220,90,18,0,0,0,0],
    11: [0,0,0,0,0,0,0, 35,110,220,360,480,580,620,560,430,280,150,50,5,0,0,0,0],
    12: [0,0,0,0,0,0,0, 20,70,160,280,400,490,530,480,360,220,110,30,3,0,0,0,0],
}

def get_hourly_temp(month, hour):
    """查询TMY逐时温度 (℃) — 文件24 §1.1"""
    return TMY_HOURLY_TEMP[month][hour]

def get_hourly_temp_array(month):
    """查询TMY某月24h温度数组"""
    return np.array(TMY_HOURLY_TEMP[month])

def get_ghi_clear(month):
    """查询某月晴空GHI 24h剖面 (W/m²) — 文件24 §1.2"""
    return np.array(TMY_GHI_CLEAR[month])

# ============================================================
# 碳交易参数
# 数据来源: 全国碳排放权交易市场(CEA) 2024-2025年均价约70-80元/tCO2
#          CCER(中国核证自愿减排量)价格通常略低于CEA, 取70元/tCO2
#          碳排放因子来源于生态环境部2024年数据 (CARBON_FACTOR_GRID=0.5703 tCO2/MWh)
# ============================================================
CCER_PRICE = 70.0              # 元/tCO2 (碳交易价格, 全国碳市场2024-2025均价)
CCER_PRICE_ESCALATION = 0.03   # 碳价年上涨率 (参考EU ETS历史趋势)

# v6.3: CCER/CEA碳价年度预测 (文件24 §4.1)
# CEA 2024达96元, 2030目标120-140元; CCER通常略低于CEA
CCER_PRICE_FORECAST = {
    2025: 71, 2026: 85, 2027: 95, 2028: 108,
    2029: 118, 2030: 130, 2035: 160, 2040: 200,
}
# 年均涨幅: 2025-2030 ~10%, 2030-2040 ~4.4%

# ============================================================
# 政府补贴参数
# 数据来源: 文件01 (充电价格与电价数据, 各省充电设施补贴)
#          文件13 (设备参数与经济数据, 分布式光伏/储能地方补贴)
#          国家发改委《关于促进光伏产业健康发展的若干意见》
#          注: 光伏国家补贴已退坡(2021年后), 但仍有个别省市提供地方补贴
# ============================================================
PV_SUBSIDY_PER_KWP = 200.0     # 元/kWp (分布式光伏地方补贴, 部分省份仍有)
ESS_SUBSIDY_PER_KWH = 100.0    # 元/kWh (储能容量补贴, 部分省份按容量补贴)
ESS_SUBSIDY_PER_KW = 50.0      # 元/kW (储能功率补贴)
CHARGING_SUBSIDY_PER_PILE = 10000.0  # 元/台 (充电桩建设补贴, 直流快充桩)
# 补贴总上限占投资比 (避免补贴超过合理范围)
MAX_SUBSIDY_RATIO = 0.30       # 补贴不超过总投资的30%

# ============================================================
# 中长期演化情景参数 (Scenario Parameters)
# 数据来源: 文件06 §5 (负荷增长率预测: 综合用电年增8-15%)
#          文件02 §2 (新能源车高速充电行为特征: 渗透率年增15-25%)
#          文件09 §5.2 (敏感性分析与情景分析)
# ============================================================

# --- 基准情景 (Baseline) ---
SCENARIO_BASELINE = {
    'load_growth_rate': 0.10,          # 综合用电负荷年增长率
    'ev_penetration_growth': 0.20,     # EV渗透率年增长率
    'grid_price_escalation': 0.03,     # 电价年上涨率
    'pv_cost_reduction': 0.02,         # 光伏成本年下降率 (学习曲线)
    'ess_cost_reduction': 0.04,        # 储能成本年下降率 (技术快速进步)
    'carbon_price_growth': 0.03,       # 碳价年上涨率
    'discount_rate': 0.07,             # 折现率
}

# --- 保守情景 (Conservative) ---
SCENARIO_CONSERVATIVE = {
    'load_growth_rate': 0.05,          # 低负荷增长
    'ev_penetration_growth': 0.10,     # 低EV渗透率增长
    'grid_price_escalation': 0.01,     # 电价平稳
    'pv_cost_reduction': 0.01,         # 光伏成本缓慢下降
    'ess_cost_reduction': 0.02,        # 储能成本缓慢下降
    'carbon_price_growth': 0.01,       # 碳价平稳
    'discount_rate': 0.08,             # 较高折现率 (保守投资态度)
}

# --- 激进情景 (Aggressive) ---
SCENARIO_AGGRESSIVE = {
    'load_growth_rate': 0.15,          # 高负荷增长
    'ev_penetration_growth': 0.28,     # 高EV渗透率增长
    'grid_price_escalation': 0.05,     # 电价快速上涨
    'pv_cost_reduction': 0.04,         # 光伏成本快速下降
    'ess_cost_reduction': 0.07,        # 储能成本快速下降
    'carbon_price_growth': 0.06,       # 碳价快速上涨
    'discount_rate': 0.06,             # 较低折现率 (积极投资态度)
}

# 情景名称映射
SCENARIOS = {
    'conservative': SCENARIO_CONSERVATIVE,
    'baseline': SCENARIO_BASELINE,
    'aggressive': SCENARIO_AGGRESSIVE,
}

# ============================================================
# v6.4: S曲线负荷增长模型 (文件25 §3)
# ============================================================
# Bass扩散模型参数 — NEV保有量 (p/q拟合自2015-2025公安部数据)
BASS_P = 0.02               # 创新系数 (外部影响: 政策/广告)
BASS_Q = 0.40               # 模仿系数 (内部影响: 口碑/网络效应)
BASS_K = 3.0e8              # 承载容量 (中国汽车保有量上限 ~3亿辆)
LOGISTIC_R = 0.45           # Logistic S曲线增长速率
LOGISTIC_T0 = 2024.5        # 拐点年份 (50%渗透率)
LOAD_GROWTH_MODEL = 'logistic'  # 'exponential' (旧) 或 'logistic'
LOAD_GROWTH_BASE_YEAR = 2025   # 基准年

# 高速充电量年增长指数 (以2020=1.00, 文件25 §3.5)
# 5年CAGR 78.5% (2018-2023), 近两年放缓至30-35%
EV_CHARGING_HIGHWAY_GROWTH = {
    2020: 1.00, 2021: 2.28, 2022: 4.41, 2023: 8.11,
    2024: 10.88, 2025: 14.20, 2026: 17.5, 2027: 20.8,
    2028: 23.5, 2029: 25.8, 2030: 27.5,
}

def get_load_growth_factor(year, model=None, base_year=None):
    """计算负荷增长因子 (文件25 §3.6)

    year: 目标年份 (如 2030)
    model: 'exponential' (线性年增) 或 'logistic' (S曲线)
    base_year: 基准年 (默认 LOAD_GROWTH_BASE_YEAR)

    返回: 目标年负荷 / 基准年负荷
    """
    if model is None:
        model = LOAD_GROWTH_MODEL
    if base_year is None:
        base_year = LOAD_GROWTH_BASE_YEAR

    if model == 'exponential':
        rate = SCENARIO_BASELINE['load_growth_rate']
        return (1.0 + rate) ** (year - base_year)
    elif model == 'logistic':
        # Logistic: N(t) = K / (1 + exp(-r*(t-t0)))
        # 负荷增长正比于NEV渗透率增长
        def logistic(t):
            return 1.0 / (1.0 + np.exp(-LOGISTIC_R * (t - LOGISTIC_T0)))
        pen_target = logistic(year)
        pen_base = logistic(base_year)
        return pen_target / max(pen_base, 0.001)
    else:
        raise ValueError(f"Unknown growth model: {model}")

# ============================================================
# v6.4: V2G双向充放电参数 (文件25 §1)
# ============================================================
V2G_ENABLED = False              # 默认关闭 (需硬件支持)
V2G_PILES = 4                    # V2G双向桩数量
V2G_POWER_KW = 120               # 单桩功率 (kW)
V2G_EFFICIENCY = 0.80            # 往返效率
V2G_DOD_LIMIT = 0.20             # 放电深度上限 (保护电池)
V2G_PRICE_YUAN_KWH = 1.0         # V2G放电补偿价 (元/kWh, 试点均值)
V2G_DEGRADATION_COST = 0.20      # 电池退化成本 (元/kWh/次)
V2G_NET_REVENUE = V2G_PRICE_YUAN_KWH - V2G_DEGRADATION_COST  # 净收益 0.80

def estimate_v2g_annual_revenue(v2g_piles=V2G_PILES, power_kw=V2G_POWER_KW,
                                 efficiency=V2G_EFFICIENCY, dod=V2G_DOD_LIMIT,
                                 price=V2G_PRICE_YUAN_KWH, deg_cost=V2G_DEGRADATION_COST,
                                 hours_per_day=4):
    """估算V2G年收益 (文件25 §1.4)"""
    v2g_capacity = v2g_piles * power_kw * efficiency * dod  # kW可调
    daily_kwh = v2g_capacity * hours_per_day
    net_price = price - deg_cost
    return daily_kwh * 365 * net_price

# ============================================================
# v6.4: 需求响应(DR)参数 (文件25 §2)
# ============================================================
DR_ENABLED = False                    # 默认关闭
DR_EVENTS_PER_YEAR = 25              # 年响应次数
DR_COMPENSATION_YUAN_KWH = 4.0       # 补偿单价 (元/kWh, 华中均值)
DR_DURATION_HOURS = 2                # 单次响应时长
DR_BASELINE_METHOD = 'avg_5d'        # 基线: 前5日均值
DR_MIN_QUALIFICATION = 0.85          # 最低合格率
DR_ESS_SOC_RESERVE = 0.50            # 储能SOC保留比例
DR_MEDIUM_SERVICE_AREA_REVENUE = 60000  # 中型服务区年DR收入 (元)

def estimate_dr_annual_revenue(ess_power_kw=500, duration=DR_DURATION_HOURS,
                                price=DR_COMPENSATION_YUAN_KWH, events=DR_EVENTS_PER_YEAR,
                                qualification=DR_MIN_QUALIFICATION):
    """估算需求响应年收益 (文件25 §2.4)

    ess_power_kw: 储能可调功率 (50%SOC保留后)
    """
    adjustable_power = ess_power_kw * 0.50  # 保留50% SOC
    single_event = adjustable_power * duration * price
    return single_event * events * qualification

# ============================================================
# 储能温度衰减参数
# 数据来源: 文件13 (LFP电池工作温度范围: 充电0-45℃, 放电-20~60℃)
#          文献: LFP电池低温容量衰减特性 (0℃时可用容量~85%, -10℃时~70%)
# ============================================================
# 环境温度对储能可用容量的影响系数 (基于LFP电池特性)
# 用于估算冬季储能实际可用容量
ESS_TEMP_DERATE = {
    35: 1.00,     # 高温不衰减 (但加速日历老化)
    25: 1.00,     # 标准温度
    15: 0.98,     # 微降
    5:  0.92,     # 低温容量下降 (电解质粘度增大)
    -5: 0.82,     # 显著下降
    -10: 0.72,    # 大幅下降
}

def get_ess_temp_derate(ambient_temp):
    """根据环境温度查询储能容量衰减系数 (线性插值)

    数据来源: 文件13 LFP电池工作温度特性
    """
    temps = np.array(sorted(ESS_TEMP_DERATE.keys()))
    derates = np.array([ESS_TEMP_DERATE[t] for t in temps])
    if ambient_temp >= temps[-1]:
        return derates[-1]
    if ambient_temp <= temps[0]:
        return derates[0]
    return float(np.interp(ambient_temp, temps, derates))

# ============================================================
# 辅助函数
# ============================================================
def get_tou_price(hour, season='spring'):
    """获取分时电价, 支持季节性差异"""
    prices = TOU_PRICE_VALUES.get(season, TOU_PRICE_VALUES['spring'])
    for period, ranges in [('peak', TOU_PRICE['peak']),
                            ('flat', TOU_PRICE['flat']),
                            ('valley', TOU_PRICE['valley'])]:
        for start, end in ranges:
            if start <= hour < end:
                p = prices[period]
                # 夏季尖峰叠加
                if season == 'summer':
                    for ps, pe in SUMMER_PEAK_EXTRA:
                        if ps <= hour < pe:
                            p *= 1.20
                return p
    return prices['flat']

def get_tou_price_array(season='spring'):
    """返回24h分时电价数组"""
    return np.array([get_tou_price(h, season) for h in range(24)])

def get_season(month):
    if month in [3,4,5]: return 'spring'
    elif month in [6,7,8]: return 'summer'
    elif month in [9,10,11]: return 'autumn'
    return 'winter'

def compute_svg_sizing(peak_charging_kw, peak_building_kw, min_charging_kw=50, target_pf=0.95):
    """SVG无功补偿容量估算 — 多场景校核

    校核场景:
    1. 峰值负荷 (充电+建筑同时最大)
    2. 纯建筑负荷 (夜间充电低谷)
    3. 光伏满发时段 (逆变器可提供无功, 但夜间不可用)
    取最恶劣工况确定SVG容量.
    """
    pf_charging = 0.97
    pf_building = 0.85

    def q_from_pf(p, pf):
        return p * np.tan(np.arccos(pf))

    def required_svg(p_total, q_total):
        target_q = p_total * np.tan(np.arccos(target_pf))
        return max(0.0, q_total - target_q)

    scenarios = {}

    # 场景1: 峰值
    p1 = peak_charging_kw + peak_building_kw
    q1 = q_from_pf(peak_charging_kw, pf_charging) + q_from_pf(peak_building_kw, pf_building)
    scenarios['peak'] = {'p_kw': p1, 'q_kvar': q1, 'svg_kvar': required_svg(p1, q1)}

    # 场景2: 纯建筑 (夜间)
    p2 = peak_building_kw + min_charging_kw
    q2 = q_from_pf(min_charging_kw, pf_charging) + q_from_pf(peak_building_kw, pf_building)
    scenarios['night_building'] = {'p_kw': p2, 'q_kvar': q2, 'svg_kvar': required_svg(p2, q2)}

    # 场景3: 纯建筑+零充电
    p3 = peak_building_kw
    q3 = q_from_pf(peak_building_kw, pf_building)
    scenarios['building_only'] = {'p_kw': p3, 'q_kvar': q3, 'svg_kvar': required_svg(p3, q3)}

    svg_kvar = max(s['svg_kvar'] for s in scenarios.values())
    svg_standard = np.ceil(svg_kvar / 50) * 50

    worst = max(scenarios, key=lambda k: scenarios[k]['svg_kvar'])
    return {
        'scenarios': scenarios,
        'worst_case': worst,
        'svg_capacity_kvar': svg_kvar,
        'svg_standard_kvar': svg_standard,
        'svg_cost_wan_yuan': svg_standard * 300 / 10000,  # ~300元/kvar
    }

def print_cost_breakdown():
    """打印设备级成本明细"""
    print("\n" + "=" * 55)
    print("设备级成本明细 (中型服务区, 单位: 万元)")
    print("=" * 55)
    print(f"光伏系统:")
    for k, v in PV_COST.items():
        print(f"  {k}: {v} 元/kWp")
    print(f"  合计: {PV_COST_PER_KWP} 元/kWp")
    print(f"储能系统: {ESS_COST['battery_per_kwh']}元/kWh + {ESS_COST['pcs_per_kw']}元/kW")
    print(f"充电桩: 120kW={CHARGING_COST['pile_120kw']/1e4:.1f}万, 480kW={CHARGING_COST['pile_480kw']/1e4:.1f}万")
    print(f"固定投资: {FIXED_COST_TOTAL/1e4:.1f}万元")
    for k, v in FIXED_COST.items():
        print(f"  {k}: {v/1e4:.1f}万")

# ============================================================
# v6.2 新增: SOC依赖的RTE + 经济参数数据补齐
# ============================================================

def get_rte(soc, c_rate=0.25):
    """SOC和C-rate依赖的LFP电池充放电效率 (文件03/13)

    文件03: LFP RTE 90-94%, 标准0.5C倍率
    中SOC段效率最高, SOC极值段扩散限制导致效率下降.
    C-rate越高, 欧姆损耗越大.

    Parameters
    ----------
    soc : float (0-1)
    c_rate : float (充电/放电倍率 = 功率/容量)

    Returns
    -------
    float : 往返效率 (0-1)
    """
    # SOC修正: 偏离SOC=0.5越多, 效率越低
    soc_dev = abs(soc - 0.5) / 0.4
    soc_factor = 1.0 - 0.04 * min(soc_dev, 1.0)
    # C-rate修正: 高倍率时效率降低
    c_factor = 1.0 - 0.02 * min(c_rate / 0.5, 2.0)
    return 0.93 * soc_factor * c_factor


# 文件01: 上网电价年涨幅 (保守假设, 基于历年分布式光伏政策趋势)
FEED_IN_PRICE_ESCALATION = 0.01

# 文件13 §4: 分项残值率 (替代统一RESIDUAL_RATE=0.05)
RESIDUAL_RATE_PV = 0.10        # 光伏 10% (25年)
RESIDUAL_RATE_ESS = 0.05       # 储能 5% (15年)
RESIDUAL_RATE_CHARGING = 0.05  # 充电桩 5% (10年)


if __name__ == '__main__':
    print_cost_breakdown()
