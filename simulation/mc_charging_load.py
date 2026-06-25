"""
蒙特卡洛充电负荷仿真模块
v6.6: 物理模型驱动 — 高速流量 + CC-CV曲线 + 里程-SOC物理关联
"""
import numpy as np
from scipy import stats
from config import (
    MILEAGE_MU, MILEAGE_SIGMA,
    SOC_START_SHAPE, SOC_START_SCALE,
    CHARGE_POWER_DIST, VEHICLE_TYPES,
    TARGET_SOC_MEAN, TARGET_SOC_STD, CHARGE_EFFICIENCY,
    HOURLY_ARRIVAL_RATE, WEATHER_CHARGING_COEFF,
    MONTHLY_ARRIVAL_MULTIPLIER, CHARGING_PENETRATION,
    PILE_AVAILABILITY,
    # v6.6: 物理模型参数
    HIGHWAY_TRAFFIC_PROFILE, NEV_PENETRATION_HIGHWAY,
    CHARGING_CONVERSION_RATE, HIGHWAY_MILEAGE_DIST,
    DEPARTURE_SOC_DIST, CC_CV_PARAMS, SERVICE_AREA_SPACING_KM,
    get_nev_penetration, compute_arrival_soc,
)
from charging_curve import cc_cv_power_curve, calculate_charging_duration

# v6.6: 预计算CC-CV充电时长查找表 [SOC_start_5pct][power_kw][batt_kwh]
_soc_bins = np.linspace(0.05, 0.95, 19)  # 5%步长
_batt_bins = np.array([25, 35, 50, 65, 85, 100, 80])  # 车型电池容量
_power_bins = np.array([60, 120, 240, 480])
_LUT_DURATION = {}  # key: (soc_idx, power, batt) -> 充电时长(h)


def _build_lut():
    global _LUT_DURATION
    from config import CC_CV_PARAMS, CHARGE_EFFICIENCY
    for i, soc_start in enumerate(_soc_bins):
        for p in _power_bins:
            for b in _batt_bins:
                dur = calculate_charging_duration(soc_start, 0.85, b, p,
                                                  CHARGE_EFFICIENCY, CC_CV_PARAMS)
                _LUT_DURATION[(i, p, b)] = dur


def _lookup_duration(soc_start, battery_cap, power):
    """从查找表近似获取充电时长 (线性插值)"""
    if not _LUT_DURATION:
        _build_lut()
    # 找到最近的SOC bin
    soc_idx = int(np.digitize(soc_start, _soc_bins))
    soc_idx = min(soc_idx, len(_soc_bins) - 1)
    # 找到最近的电池容量
    batt_idx = int(np.argmin(np.abs(_batt_bins - battery_cap)))
    batt = _batt_bins[batt_idx]
    # 找到功率
    power_idx = int(np.argmin(np.abs(_power_bins - power)))
    p = _power_bins[power_idx]
    return _LUT_DURATION.get((soc_idx, p, batt), 1.0)

# v6.6 (保留旧版GMM用于向后兼容)
_GMM_W1, _GMM_MU1, _GMM_SIG1 = 0.45, 11.0, 1.5
_GMM_W2, _GMM_MU2, _GMM_SIG2 = 0.55, 15.0, 1.8

# 文件05: 季节能耗因子
_SEASON_ENERGY_FACTOR = {
    12: 1.15, 1: 1.15, 2: 1.15,
    6: 1.08, 7: 1.08, 8: 1.08,
}

_EARLY_DEPARTURE_PROB = 0.25
_MAX_CHARGE_HOURS = {60: 4.0, 120: 2.5, 240: 1.5, 480: 1.0}

# 文件11 中型服务区参考值
_REF_DAILY_ENERGY = {
    'workday': 15000, 'weekend': 17000,
    'holiday': 19500, 'spring_festival': 23000,
}


class MonteCarloChargingSimulator:
    """蒙特卡洛充电负荷仿真器 (v6.6 物理模型)"""

    def __init__(self, service_area_size='medium', seed=None, year=2025,
                 use_traffic_model=True):
        self.size = service_area_size
        self.rng = np.random.RandomState(seed)
        self.power_levels = list(CHARGE_POWER_DIST.keys())
        self.power_probs = list(CHARGE_POWER_DIST.values())
        self._vt_names = list(VEHICLE_TYPES.keys())
        self._vt_weights = [VEHICLE_TYPES[k]['weight'] for k in self._vt_names]
        self.year = year
        self.use_traffic_model = use_traffic_model
        self._nev_pen = get_nev_penetration(year)

        # 仿真日志
        self._last_traffic_count = 0
        self._last_nev_count = 0
        self._last_charged_count = 0

    def _precompute_failures(self, recovery_array, pile_params):
        mtbf = pile_params['mtbf_hours']
        mttr = pile_params['mttr_hours']
        n_piles = len(recovery_array)
        if n_piles == 0:
            return
        for pile_idx in range(n_piles):
            for hour in range(24):
                p_fail = 1.0 / mtbf if mtbf > 0 else 0.0
                if self.rng.random() < p_fail:
                    repair_time = mttr * (0.7 + 0.6 * self.rng.random())
                    recovery_array[pile_idx] = max(
                        recovery_array[pile_idx], hour + repair_time)

    def _sample_vehicle(self, month=None):
        """v6.6: 物理SOC模型 — 里程反推到站SOC"""
        vt_name = self.rng.choice(self._vt_names, p=self._vt_weights)
        vt = VEHICLE_TYPES[vt_name]

        energy_cons = self.rng.normal(vt['cons_mean'], vt['cons_std'])
        energy_cons = max(8, min(35, energy_cons))
        if month is not None:
            energy_cons *= _SEASON_ENERGY_FACTOR.get(month, 1.0)

        battery_cap = self.rng.normal(vt['battery_mean'], vt['battery_std'])
        battery_cap = max(15, min(250, battery_cap))

        # v6.6: 出发SOC (用户出发前通常已充电)
        departure_soc = self.rng.normal(
            DEPARTURE_SOC_DIST.get('mean', 0.85),
            DEPARTURE_SOC_DIST.get('std', 0.10))
        departure_soc = np.clip(departure_soc, 0.50, 1.0)

        # 高速行驶里程
        mu_m = HIGHWAY_MILEAGE_DIST['mu_km']
        sigma_m = HIGHWAY_MILEAGE_DIST['sigma_km']
        mileage = self.rng.lognormal(np.log(mu_m), sigma_m / mu_m)
        mileage = np.clip(mileage, 50, 500)

        # 物理到站SOC
        soc_at_arrival = compute_arrival_soc(mileage, battery_cap, energy_cons, departure_soc)

        # 目标SOC
        target_soc = self.rng.normal(TARGET_SOC_MEAN, TARGET_SOC_STD)
        target_soc = np.clip(target_soc, 0.70, 0.98)

        # 充电功率
        max_p = vt['max_power']
        available_levels = [p for p in self.power_levels if p <= max_p]
        if not available_levels:
            power = min(self.power_levels)
        else:
            probs = np.array([CHARGE_POWER_DIST[p] for p in available_levels])
            probs = probs / probs.sum()
            power = self.rng.choice(available_levels, p=probs)
        power = min(power, max_p)

        charge_needed = max(0, (target_soc - soc_at_arrival) * battery_cap)

        return {
            'soc_arrival': soc_at_arrival,
            'target_soc': target_soc,
            'battery_cap': battery_cap,
            'power': power,
            'max_power': max_p,
            'charge_needed': charge_needed,
            'mileage_km': mileage,
            'energy_cons': energy_cons,
            'departure_soc': departure_soc,
            'vt_name': vt_name,
        }

    def _generate_traffic_arrivals(self, day_type, weather='clear'):
        """v6.6: 车流量驱动到达模型

        计算逻辑:
        1. 日过境车辆 × 逐时占比 → 逐时过境车流
        2. × NEV渗透率 → 逐时NEV数量
        3. × 充电转化率 → 逐时需要充电的NEV数量
        4. Poisson采样 → 实际到达时刻

        Returns
        -------
        list[float] : 到达时刻列表 (小时, 0-24)
        """
        if not self.use_traffic_model:
            # 回退到旧版GMM
            arrival_rates = HOURLY_ARRIVAL_RATE.get(day_type, HOURLY_ARRIVAL_RATE['workday'])
            total = int(sum(arrival_rates))
            weather_factor = WEATHER_CHARGING_COEFF.get(weather, 1.0)
            actual = self.rng.poisson(total * weather_factor)
            actual = max(0, min(actual, int(total * weather_factor * 1.5)))
            arrivals = []
            for _ in range(actual):
                if self.rng.random() < _GMM_W1:
                    t = self.rng.normal(_GMM_MU1, _GMM_SIG1)
                else:
                    t = self.rng.normal(_GMM_MU2, _GMM_SIG2)
                arrivals.append(np.clip(t, 0, 23.99))
            arrivals.sort()
            self._last_traffic_count = total
            self._last_nev_count = total
            self._last_charged_count = actual
            return arrivals

        # 物理模型
        profile = HIGHWAY_TRAFFIC_PROFILE.get(day_type, HIGHWAY_TRAFFIC_PROFILE['workday'])
        daily_base = profile['daily_base']
        hourly_pct = np.array(profile['hourly_pct'], dtype=float) / 100.0

        # 天气影响
        weather_factor = WEATHER_CHARGING_COEFF.get(weather, 1.0)
        daily_base = daily_base * weather_factor

        # 逐时NEV数量 = 过境 × 占比 × NEV渗透率
        hourly_traffic = daily_base * hourly_pct
        self._last_traffic_count = int(daily_base)

        nev_hourly = hourly_traffic * self._nev_pen
        self._last_nev_count = int(sum(nev_hourly))

        # 充电转化
        conv_rate = CHARGING_CONVERSION_RATE.get(day_type, 0.10)
        charging_hourly = nev_hourly * conv_rate

        # Poisson采样每小时的到达
        arrivals = []
        for h in range(24):
            expected = charging_hourly[h]
            n = self.rng.poisson(expected) if expected > 0 else 0
            for _ in range(n):
                t = h + self.rng.random()
                arrivals.append(np.clip(t, 0, 23.99))
        arrivals.sort()
        self._last_charged_count = len(arrivals)
        return arrivals

    def _calc_duration_cc_cv(self, vehicle, eff_power):
        """v6.6: CC-CV充电时长 (查找表加速)"""
        charge_needed = vehicle['charge_needed']
        if charge_needed <= 0 or eff_power <= 0:
            return 0.0

        # 查找表获取SOC 5%→85%基准时长, 再按实际SOC范围缩放
        dur_base = _lookup_duration(vehicle['soc_arrival'], vehicle['battery_cap'], eff_power)
        dur = dur_base * (vehicle['target_soc'] - vehicle['soc_arrival']) / (0.85 - vehicle['soc_arrival'])
        return min(dur, _MAX_CHARGE_HOURS.get(int(eff_power), 3.0))

    def simulate_day(self, day_type='workday', month=None, weather='clear'):
        """v6.6: 物理模型 + CC-CV + FCFS排队"""
        hourly_load = np.zeros(24)

        from config import SERVICE_AREA_CONFIG
        cfg = SERVICE_AREA_CONFIG.get(self.size, SERVICE_AREA_CONFIG['medium'])
        n_120kw = cfg['n_piles_120kw']
        n_480kw = cfg['n_piles_480kw'] * 5

        free_120 = np.zeros(n_120kw) if n_120kw > 0 else np.array([])
        free_480 = np.zeros(n_480kw) if n_480kw > 0 else np.array([])

        recover_120 = np.zeros(n_120kw) if n_120kw > 0 else np.array([])
        recover_480 = np.zeros(n_480kw) if n_480kw > 0 else np.array([])
        self._precompute_failures(recover_120, PILE_AVAILABILITY['120kw'])
        self._precompute_failures(recover_480, PILE_AVAILABILITY['480kw'])

        n_rejected = 0; n_queued = 0; n_early_departure = 0; n_failure_rejected = 0
        base_patience = {'workday': 0.75, 'weekend': 0.50,
                         'holiday': 0.33, 'spring_festival': 0.25}
        base_max_wait = base_patience.get(day_type, 0.5)

        arrival_hours = self._generate_traffic_arrivals(day_type, weather)

        # 预计算常用值
        eff = CHARGE_EFFICIENCY
        cc_params = CC_CV_PARAMS

        for arrival_time in arrival_hours:
            vehicle = self._sample_vehicle(month=month)
            chg_needed = vehicle['charge_needed']

            if self.rng.random() < _EARLY_DEPARTURE_PROB:
                practical_target = self.rng.uniform(0.65, 0.85)
                if practical_target <= vehicle['soc_arrival']:
                    continue
                vehicle['target_soc'] = practical_target
                chg_needed = max(0, (practical_target - vehicle['soc_arrival']) * vehicle['battery_cap'])
                vehicle['charge_needed'] = chg_needed
                if chg_needed <= 0:
                    continue
                n_early_departure += 1

            soc_arrival = vehicle['soc_arrival']
            soc_patience_factor = 0.5 + 0.7 * (soc_arrival - 0.05) / 0.90
            max_wait = base_max_wait * np.clip(soc_patience_factor, 0.4, 1.3)

            can_use_480 = (vehicle['max_power'] > 120 or vehicle['power'] > 120)
            ep480 = min(vehicle['power'], vehicle['max_power'])
            ep120 = min(ep480, 120)

            dur_480 = self._calc_duration_cc_cv(vehicle, ep480)
            dur_120 = self._calc_duration_cc_cv(vehicle, ep120)

            def _avail(free_arr, recover_arr, t):
                return (free_arr <= t) & (recover_arr <= t)

            a120 = _avail(free_120, recover_120, arrival_time)
            a480 = _avail(free_480, recover_480, arrival_time)
            early_480 = free_480[a480].min() if a480.any() else np.inf
            early_120 = free_120[a120].min() if a120.any() else np.inf

            assigned_type = None; assigned_dur = None; assigned_power = None

            if can_use_480 and arrival_time >= early_480:
                c = np.where(a480 & (free_480 <= arrival_time))[0]
                tid = c[free_480[c].argmin()]
                assigned_type = '480'; assigned_dur = dur_480; assigned_power = ep480
                assigned_tid = tid
            elif arrival_time >= early_120:
                c = np.where(a120 & (free_120 <= arrival_time))[0]
                tid = c[free_120[c].argmin()]
                assigned_type = '120'; assigned_dur = dur_120; assigned_power = ep120
                assigned_tid = tid
            else:
                best_earliest = np.inf; best_type = None; best_tid = None
                if can_use_480 and a480.any():
                    f480 = np.where(a480)[0]; tid = f480[free_480[f480].argmin()]
                    best_earliest = free_480[tid]; best_type = '480'; best_tid = tid
                if a120.any():
                    f120 = np.where(a120)[0]; tid = f120[free_120[f120].argmin()]
                    if free_120[tid] < best_earliest:
                        best_earliest = free_120[tid]; best_type = '120'; best_tid = tid
                if best_type is None:
                    n_rejected += 1; n_failure_rejected += 1; continue
                wait_time = best_earliest - arrival_time
                if wait_time > max_wait:
                    n_rejected += 1; continue
                n_queued += 1
                assigned_type = best_type; assigned_dur = (dur_480 if best_type == '480' else dur_120)
                assigned_power = (ep480 if best_type == '480' else ep120)
                assigned_tid = best_tid
                arrival_time = best_earliest

            start_time = arrival_time
            effective_end = min(start_time + assigned_dur, 24.0)

            if assigned_type == '480':
                free_480[assigned_tid] = effective_end
            else:
                free_120[assigned_tid] = effective_end

            # 累加小时负荷 (CC-CV加权功率)
            start_h = int(np.floor(start_time))
            end_h = int(np.floor(effective_end))
            for t in range(start_h, min(end_h + 1, 24)):
                overlap = min(t + 1, effective_end) - max(t, start_time)
                if overlap > 0 and assigned_dur > 0:
                    soc_mid = vehicle['soc_arrival'] + (vehicle['target_soc'] - vehicle['soc_arrival']) * 0.5 * (overlap / assigned_dur)
                    avg_p = cc_cv_power_curve(soc_mid, assigned_power, cc_params)
                    hourly_load[t] += avg_p * overlap

        self._last_rejected = n_rejected; self._last_queued = n_queued
        self._last_early_departure = n_early_departure; self._last_failure_rejected = n_failure_rejected
        return hourly_load

    def simulate_monte_carlo(self, day_type='workday', n_runs=10000, month=None, weather='clear'):
        """蒙特卡洛多次仿真"""
        all_runs = np.zeros((n_runs, 24))
        for i in range(n_runs):
            all_runs[i] = self.simulate_day(day_type, month=month, weather=weather)

        results = {
            'mean': all_runs.mean(axis=0),
            'std': all_runs.std(axis=0),
            'p5': np.percentile(all_runs, 5, axis=0),
            'p25': np.percentile(all_runs, 25, axis=0),
            'p50': np.percentile(all_runs, 50, axis=0),
            'p75': np.percentile(all_runs, 75, axis=0),
            'p95': np.percentile(all_runs, 95, axis=0),
            'max': all_runs.max(axis=0),
            'min': all_runs.min(axis=0),
            'peak_mean': all_runs.max(axis=1).mean(),
            'peak_std': all_runs.max(axis=1).std(),
            'daily_energy_mean': all_runs.sum(axis=1).mean(),
            'daily_energy_std': all_runs.sum(axis=1).std(),
            'all_runs': all_runs,
        }
        return results

    def simulate_all_scenarios(self, n_runs=5000):
        """模拟所有日类型场景"""
        scenarios = {}
        for day_type in ['workday', 'weekend', 'holiday', 'spring_festival']:
            print(f"  仿真 {day_type} ({n_runs}次)...")
            scenarios[day_type] = self.simulate_monte_carlo(day_type, n_runs)
        return scenarios

    def get_diagnostics(self):
        """v6.6: 返回仿真诊断信息"""
        return {
            'nev_penetration': self._nev_pen,
            'traffic_count': self._last_traffic_count,
            'nev_count': self._last_nev_count,
            'charged_count': self._last_charged_count,
            'rejected': getattr(self, '_last_rejected', 0),
            'queued': getattr(self, '_last_queued', 0),
            'early_departure': getattr(self, '_last_early_departure', 0),
            'failure_rejected': getattr(self, '_last_failure_rejected', 0),
        }


def print_summary(scenarios, sim=None):
    """打印仿真结果摘要"""
    print("\n" + "=" * 70)
    print("蒙特卡洛充电负荷仿真结果汇总 (v6.6 物理模型)")
    print("=" * 70)
    for day_type, result in scenarios.items():
        print(f"\n{day_type}:")
        print(f"  峰值负荷 (均值 +/- std):  {result['peak_mean']:.1f} +/- {result['peak_std']:.1f} kW")
        print(f"  峰值负荷 (P95):          {result['p95'].max():.1f} kW")
        print(f"  日均用电量 (均值 +/- std): {result['daily_energy_mean']:.1f} +/- {result['daily_energy_std']:.1f} kWh")
        print(f"  最大峰值时段:            {result['mean'].argmax()}:00")
    if sim is not None:
        diag = sim.get_diagnostics()
        print(f"\n  仿真诊断:")
        print(f"    NEV渗透率: {diag['nev_penetration']:.1%}")
        print(f"    过境车辆: {diag['traffic_count']} 辆/天")
        print(f"    其中NEV: {diag['nev_count']} 辆")
        print(f"    实际充电: {diag['charged_count']} 辆")
        print(f"    排队: {diag['queued']} 辆, 弃充: {diag['rejected']} 辆")
        print(f"    提前离充: {diag['early_departure']} 辆, 因故障: {diag['failure_rejected']} 辆")


if __name__ == '__main__':
    sim = MonteCarloChargingSimulator(service_area_size='medium', seed=42)
    scenarios = sim.simulate_all_scenarios(n_runs=3000)
    print_summary(scenarios, sim)
