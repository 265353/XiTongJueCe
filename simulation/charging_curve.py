"""
CC-CV充电曲线模型 (GB/T 27930-2023)
提供CC-CV充电功率曲线和充电时长计算
"""
import numpy as np


# 默认CC-CV参数 (与config.py CC_CV_PARAMS保持一致)
DEFAULT_CC_CV_PARAMS = {
    'cc_cv_soc': 0.80,           # CC-CV转换点 (SOC)
    'cv_end_power_ratio': 0.20,  # CV阶段结束时功率比例
    'efficiency': 0.93,          # 充电效率
    'min_power_ratio': 0.10,     # 最低功率比例
}


def cc_cv_power_curve(soc, rated_power, params=None):
    """
    计算CC-CV充电曲线在给定SOC下的充电功率

    GB/T 27930-2023 直流充电规范:
    - CC阶段 (SOC < cc_cv_soc): 恒功率 = rated_power
    - CV阶段 (SOC >= cc_cv_soc): 功率指数衰减
      P(soc) = rated_power * (1 - (1 - cv_end_power_ratio) * (soc - cc_cv_soc) / (1 - cc_cv_soc))

    Parameters
    ----------
    soc : float
        当前SOC (0-1)
    rated_power : float
        额定充电功率 (kW)
    params : dict, optional
        CC-CV参数, 默认使用 DEFAULT_CC_CV_PARAMS

    Returns
    -------
    float : 充电功率 (kW)
    """
    if params is None:
        params = DEFAULT_CC_CV_PARAMS

    cc_cv_soc = params.get('cc_cv_soc', 0.80)
    cv_end_power_ratio = params.get('cv_end_power_ratio', 0.20)
    min_power_ratio = params.get('min_power_ratio', 0.10)

    soc = max(0.0, min(1.0, soc))

    if soc < cc_cv_soc:
        # CC阶段: 恒功率
        return float(rated_power)
    else:
        # CV阶段: 功率从rated_power线性递减至cv_end_power_ratio * rated_power
        remaining_ratio = (1.0 - soc) / max(1.0 - cc_cv_soc, 0.001)
        power_ratio = cv_end_power_ratio + (1.0 - cv_end_power_ratio) * remaining_ratio
        power_ratio = max(min_power_ratio, power_ratio)
        return float(rated_power * power_ratio)


def calculate_charging_duration(soc_start, target_soc, battery_cap_kwh,
                                 rated_power_kw, efficiency=0.93, params=None):
    """
    计算CC-CV充电时长 (小时)

    通过积分充电曲线计算从soc_start到target_soc的充电时间。

    Parameters
    ----------
    soc_start : float
        起始SOC (0-1)
    target_soc : float
        目标SOC (0-1)
    battery_cap_kwh : float
        电池容量 (kWh)
    rated_power_kw : float
        额定充电功率 (kW)
    efficiency : float
        充电效率 (0-1), 默认 0.93
    params : dict, optional
        CC-CV参数

    Returns
    -------
    float : 充电时长 (小时)
    """
    if params is None:
        params = DEFAULT_CC_CV_PARAMS

    cc_cv_soc = params.get('cc_cv_soc', 0.80)
    cv_end_power_ratio = params.get('cv_end_power_ratio', 0.20)
    min_power_ratio = params.get('min_power_ratio', 0.10)

    if soc_start >= target_soc:
        return 0.0

    if rated_power_kw <= 0 or battery_cap_kwh <= 0:
        return float('inf')

    total_time = 0.0

    # 将充电过程分为CC和CV两个阶段分别计算
    # CC阶段: soc_start → min(target_soc, cc_cv_soc)
    cc_end_soc = min(target_soc, cc_cv_soc)
    if soc_start < cc_end_soc:
        # CC阶段: 恒功率, 能量 / (功率 × 效率)
        energy_cc = battery_cap_kwh * (cc_end_soc - soc_start)  # kWh
        time_cc = energy_cc / (rated_power_kw * efficiency)     # hours
        total_time += time_cc
        soc_start = cc_end_soc

    # CV阶段: min(cc_cv_soc, soc_start) → target_soc
    if soc_start < target_soc and soc_start >= cc_cv_soc:
        # CV阶段功率递减, 用数值积分
        n_steps = 200
        soc_range = np.linspace(soc_start, target_soc, n_steps)
        dt_total = 0.0
        for i in range(len(soc_range) - 1):
            soc_mid = (soc_range[i] + soc_range[i + 1]) / 2.0
            power = cc_cv_power_curve(soc_mid, rated_power_kw, params)
            power = max(power, rated_power_kw * min_power_ratio)
            delta_soc = soc_range[i + 1] - soc_range[i]
            energy_step = battery_cap_kwh * delta_soc  # kWh
            dt_total += energy_step / (power * efficiency)
        total_time += dt_total

    return float(total_time)


# ============================================================
# 预计算查找表 (LUT) 辅助函数
# ============================================================

def build_duration_lut(soc_bins=None, power_bins=None, batt_bins=None,
                        target_soc=0.85, efficiency=0.93, params=None):
    """
    构建充电时长三维查找表

    Parameters
    ----------
    soc_bins : array-like
        SOC起始值分档
    power_bins : array-like
        充电功率分档 (kW)
    batt_bins : array-like
        电池容量分档 (kWh)
    target_soc : float
        目标SOC
    efficiency : float
        充电效率
    params : dict
        CC-CV参数

    Returns
    -------
    dict : {(soc_idx, power_kw, batt_kwh): duration_hours}
    """
    if soc_bins is None:
        soc_bins = np.linspace(0.05, 0.95, 19)
    if power_bins is None:
        power_bins = [30, 60, 120, 240, 480]
    if batt_bins is None:
        batt_bins = [25, 35, 50, 65, 80, 85, 100]
    if params is None:
        params = DEFAULT_CC_CV_PARAMS

    lut = {}
    for i, soc_start in enumerate(soc_bins):
        for p in power_bins:
            for b in batt_bins:
                dur = calculate_charging_duration(soc_start, target_soc, b, p,
                                                    efficiency, params)
                lut[(i, p, b)] = dur
    return lut
