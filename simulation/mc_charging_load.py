"""
蒙特卡洛充电负荷仿真模块
基于概率抽样模拟每辆EV的充电行为，叠加得到逐时充电负荷概率分布
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
    PILE_AVAILABILITY,  # v6.5: 充电桩可靠性参数
)

# 文件05: GMM起始充电时间参数 (双峰模型)
# 午间峰 N(11,1.5^2) 45% + 下午峰 N(15,1.8^2) 55%
_GMM_W1, _GMM_MU1, _GMM_SIG1 = 0.45, 11.0, 1.5
_GMM_W2, _GMM_MU2, _GMM_SIG2 = 0.55, 15.0, 1.8

# 文件05: 季节能耗因子 (基准春秋16 kWh/100km, 冬21/夏18)
_SEASON_ENERGY_FACTOR = {
    12: 1.15, 1: 1.15, 2: 1.15,   # 冬季
    6: 1.08, 7: 1.08, 8: 1.08,    # 夏季
}

# v6.3: 提前离充概率 — 部分车辆充到实用SOC即离开 (无需充满)
_EARLY_DEPARTURE_PROB = 0.25
# v6.3: 各功率等级最大充电时长 (小时)
_MAX_CHARGE_HOURS = {60: 3.0, 120: 2.0, 240: 1.5, 480: 1.0}

# 文件11 中型服务区参考值 (用于仿真偏差校验)
_REF_DAILY_ENERGY = {
    'workday': 15000, 'weekend': 17000,
    'holiday': 19500, 'spring_festival': 23000,
}


class MonteCarloChargingSimulator:
    """蒙特卡洛充电负荷仿真器"""

    def __init__(self, service_area_size='medium', seed=None):
        self.size = service_area_size
        self.rng = np.random.RandomState(seed)
        self.power_levels = list(CHARGE_POWER_DIST.keys())
        self.power_probs = list(CHARGE_POWER_DIST.values())
        # 预计算车型采样列表
        self._vt_names = list(VEHICLE_TYPES.keys())
        self._vt_weights = [VEHICLE_TYPES[k]['weight'] for k in self._vt_names]

    def _precompute_failures(self, recovery_array, pile_params):
        """v6.5: 预生成全天桩故障事件 (Poisson过程)

        每个桩每小时有独立概率 p = 1/MTBF 发生故障,
        故障持续时间为 MTTR (含随机波动 ±30%).

        recovery_array 被就地修改, 记录每个桩的恢复时刻.
        """
        mtbf = pile_params['mtbf_hours']
        mttr = pile_params['mttr_hours']
        n_piles = len(recovery_array)
        if n_piles == 0:
            return
        for pile_idx in range(n_piles):
            for hour in range(24):
                # 每小时故障概率: p = 1/MTBF (近似, MTBF >> 1h时准确)
                p_fail = 1.0 / mtbf if mtbf > 0 else 0.0
                if self.rng.random() < p_fail:
                    # 故障持续时间: MTTR ± 30%
                    repair_time = mttr * (0.7 + 0.6 * self.rng.random())
                    # 合并连续故障: 取最晚恢复时间
                    recovery_array[pile_idx] = max(
                        recovery_array[pile_idx],
                        hour + repair_time)

    def _sample_vehicle(self, month=None):
        """抽样单辆车的充电参数 (使用7种车型独立分布)

        Parameters
        ----------
        month : int or None
            月份(1-12), 用于季节能耗调整 (文件05)
        """
        # 按权重抽样车型
        vt_name = self.rng.choice(self._vt_names, p=self._vt_weights)
        vt = VEHICLE_TYPES[vt_name]

        # 日行驶里程 (km) — 与车型无关, 全局分布
        mileage = self.rng.lognormal(MILEAGE_MU, MILEAGE_SIGMA)

        # 百公里电耗 (kWh/100km) — 车型相关 (文件05: 季节调整)
        energy_cons = self.rng.normal(vt['cons_mean'], vt['cons_std'])
        energy_cons = max(8, min(35, energy_cons))
        if month is not None:
            energy_cons *= _SEASON_ENERGY_FACTOR.get(month, 1.0)

        # 电池容量 (kWh) — 车型相关
        battery_cap = self.rng.normal(vt['battery_mean'], vt['battery_std'])
        battery_cap = max(15, min(250, battery_cap))

        # 消耗电量
        consumed_energy = mileage * energy_cons / 100.0

        # 起始SOC — Gamma(2.8, scale=0.075), mean=21% (文件11)
        soc_start = self.rng.gamma(SOC_START_SHAPE, SOC_START_SCALE)
        soc_start = np.clip(soc_start, 0.05, 0.95)

        # 到达时SOC
        soc_at_arrival = soc_start - consumed_energy / battery_cap
        soc_at_arrival = max(0.05, min(soc_start, soc_at_arrival))

        # 目标SOC
        target_soc = self.rng.normal(TARGET_SOC_MEAN, TARGET_SOC_STD)
        target_soc = np.clip(target_soc, 0.70, 0.98)

        # 充电功率 (kW) — 受车型最大功率限制
        max_p = vt['max_power']
        available_levels = [p for p in self.power_levels if p <= max_p]
        if not available_levels:
            # 车型最大功率小于最小充电桩功率 → 使用最小桩, 实际功率受限
            power = min(self.power_levels)
        else:
            # 重新归一化概率
            probs = np.array([CHARGE_POWER_DIST[p] for p in available_levels])
            probs = probs / probs.sum()
            power = self.rng.choice(available_levels, p=probs)
        power = min(power, max_p)

        # 充电量 (kWh)
        charge_needed = (target_soc - soc_at_arrival) * battery_cap
        charge_needed = max(0, charge_needed)

        return {
            'soc_arrival': soc_at_arrival,
            'target_soc': target_soc,
            'battery_cap': battery_cap,
            'power': power,
            'max_power': max_p,
            'charge_needed': charge_needed,
        }

    def simulate_day(self, day_type='workday', month=None, weather='clear'):
        """模拟一天的充电负荷 (v6.5 FCFS排队 + 可靠性 + SOC依赖耐心 + 提前离充 + 天气)

        终端分为两组:
        - 120kW快充桩: 数量 n_120kw, 最大输出120kW
        - 480kW超充终端: 数量 n_480kw*5, 最大输出480kW

        v6.5 改进:
        - 可靠性: 充电桩随机故障 (MTBF/MTTR), 故障期间不可用
        - FCFS: 全忙时取最早释放的兼容终端, 不再因480kW偏好延长等待
        - SOC依赖耐心: 低SOC车辆耐心更短 (如SOC=5%时耐心×0.5)
        - 提前离充: 部分车辆充到实用SOC即离开 (不追求满充)
        - 天气修正: 恶劣天气减少出行 → 充电需求下降

        Parameters
        ----------
        day_type : str
            日类型
        month : int or None
            月份(1-12), 用于季节能耗和GMM到达微调
        weather : str
            天气类型: clear/partly_cloudy/cloudy/overcast/rain
        """
        hourly_load = np.zeros(24)
        arrival_rates = HOURLY_ARRIVAL_RATE.get(day_type, HOURLY_ARRIVAL_RATE['workday'])

        # v6.3: 月度车流量调节 (文件24 §2.2)
        monthly_mult = MONTHLY_ARRIVAL_MULTIPLIER.get(month, 1.0) if month is not None else 1.0

        total_daily_arrivals = int(sum(arrival_rates) * monthly_mult)

        # 天气对充电需求的影响 (恶劣天气减少出行)
        weather_chg_factor = WEATHER_CHARGING_COEFF.get(weather, 1.0)

        from config import SERVICE_AREA_CONFIG
        cfg = SERVICE_AREA_CONFIG.get(self.size, SERVICE_AREA_CONFIG['medium'])
        n_120kw = cfg['n_piles_120kw']           # 120kW终端数
        n_480kw = cfg['n_piles_480kw'] * 5         # 480kW终端数

        # 两组终端的释放时间 (使用完毕后空闲)
        free_120 = np.zeros(n_120kw) if n_120kw > 0 else np.array([])
        free_480 = np.zeros(n_480kw) if n_480kw > 0 else np.array([])

        # v6.5: 充电桩故障恢复时间 (故障期间不可用)
        recover_120 = np.zeros(n_120kw) if n_120kw > 0 else np.array([])
        recover_480 = np.zeros(n_480kw) if n_480kw > 0 else np.array([])

        # v6.5: 预生成全天故障事件 (每小时检查一次)
        self._precompute_failures(recover_120, PILE_AVAILABILITY['120kw'])
        self._precompute_failures(recover_480, PILE_AVAILABILITY['480kw'])

        n_rejected = 0
        n_queued = 0
        n_early_departure = 0
        n_failure_rejected = 0  # v6.5: 因故障不可用导致的额外弃充

        # 基础耐心值 (小时) — 按日类型分档
        base_patience = {'workday': 0.75, 'weekend': 0.50,
                         'holiday': 0.33, 'spring_festival': 0.25}
        base_max_wait = base_patience.get(day_type, 0.5)

        # GMM到达时间模型: 按天总量采样到达时刻, 考虑天气影响
        actual_arrivals = self.rng.poisson(total_daily_arrivals * weather_chg_factor)
        actual_arrivals = max(0, min(actual_arrivals, int(total_daily_arrivals * weather_chg_factor * 1.5)))

        arrival_hours = []
        for _ in range(actual_arrivals):
            if self.rng.random() < _GMM_W1:
                t = self.rng.normal(_GMM_MU1, _GMM_SIG1)
            else:
                t = self.rng.normal(_GMM_MU2, _GMM_SIG2)
            t = np.clip(t, 0, 23.99)
            arrival_hours.append(t)
        arrival_hours.sort()

        for arrival_time in arrival_hours:
            vehicle = self._sample_vehicle(month=month)

            # v6.3: 提前离充模型 — 部分车辆接受实用SOC, 不追求满充
            if self.rng.random() < _EARLY_DEPARTURE_PROB:
                practical_target = self.rng.uniform(0.65, 0.85)
                vehicle['target_soc'] = min(vehicle['target_soc'], practical_target)
                vehicle['charge_needed'] = max(0,
                    (vehicle['target_soc'] - vehicle['soc_arrival']) * vehicle['battery_cap'])
                if vehicle['charge_needed'] <= 0:
                    continue
                n_early_departure += 1

            # v6.3: SOC依赖耐心 — SOC越低耐心越短
            soc_arrival = vehicle['soc_arrival']
            soc_patience_factor = 0.5 + 0.7 * (soc_arrival - 0.05) / 0.90
            max_wait = base_max_wait * np.clip(soc_patience_factor, 0.4, 1.3)

            # 判断车辆是否能从480kW终端获益
            can_use_480 = (vehicle['max_power'] > 120 or vehicle['power'] > 120)

            # 确定在每种终端上的有效充电功率
            eff_power_480 = min(vehicle['power'], vehicle['max_power'])
            eff_power_120 = min(vehicle['power'], vehicle['max_power'], 120)

            # 充电时长 (取决于终端类型, 含上限)
            dur_480 = vehicle['charge_needed'] / (eff_power_480 * CHARGE_EFFICIENCY) if eff_power_480 > 0 else 99
            dur_120 = vehicle['charge_needed'] / (eff_power_120 * CHARGE_EFFICIENCY) if eff_power_120 > 0 else 99
            dur_480 = min(dur_480, _MAX_CHARGE_HOURS.get(480, 1.0))
            dur_120 = min(dur_120, _MAX_CHARGE_HOURS.get(120, 2.0))

            # v6.5: 过滤可用桩 (空闲 + 未故障)
            def _available_mask(free_arr, recover_arr, t):
                """桩可用: 空闲且未故障"""
                return (free_arr <= t) & (recover_arr <= t)

            avail_120 = _available_mask(free_120, recover_120, arrival_time)
            avail_480 = _available_mask(free_480, recover_480, arrival_time)

            earliest_480 = free_480[avail_480].min() if avail_480.any() else np.inf
            earliest_120 = free_120[avail_120].min() if avail_120.any() else np.inf

            assigned_terminal = None
            assigned_duration = None
            assigned_power = None

            # v6.5 FCFS: 空闲+健康时优先480kW, 全忙/全故障时取最早释放的兼容健康终端
            if can_use_480 and arrival_time >= earliest_480:
                # 480kW终端空闲且健康
                candidates = np.where(avail_480 & (free_480 <= arrival_time))[0]
                tid = candidates[free_480[candidates].argmin()]
                assigned_terminal = ('480', tid)
                assigned_duration = dur_480
                assigned_power = eff_power_480
            elif arrival_time >= earliest_120:
                # 120kW终端空闲且健康
                candidates = np.where(avail_120 & (free_120 <= arrival_time))[0]
                tid = candidates[free_120[candidates].argmin()]
                assigned_terminal = ('120', tid)
                assigned_duration = dur_120
                assigned_power = eff_power_120
            else:
                # 全忙或全故障 — 取最早释放的健康兼容终端
                best_earliest = np.inf
                best_ttype = None
                best_tid = None

                if can_use_480 and avail_480.any():
                    # 检查是否有健康且最早释放的480kW终端
                    future_480 = np.where(avail_480)[0]
                    if len(future_480) > 0:
                        tid = future_480[free_480[future_480].argmin()]
                        if free_480[tid] < best_earliest:
                            best_earliest = free_480[tid]
                            best_ttype = '480'
                            best_tid = tid
                if avail_120.any():
                    future_120 = np.where(avail_120)[0]
                    if len(future_120) > 0:
                        tid = future_120[free_120[future_120].argmin()]
                        if free_120[tid] < best_earliest:
                            best_earliest = free_120[tid]
                            best_ttype = '120'
                            best_tid = tid

                if best_ttype is None:
                    n_rejected += 1
                    n_failure_rejected += 1
                    continue

                wait_time = best_earliest - arrival_time
                if wait_time > max_wait:
                    n_rejected += 1
                    continue

                n_queued += 1
                assigned_terminal = (best_ttype, best_tid)
                if best_ttype == '480':
                    assigned_duration = dur_480
                    assigned_power = eff_power_480
                else:
                    assigned_duration = dur_120
                    assigned_power = eff_power_120
                arrival_time = best_earliest

            start_time = arrival_time
            end_time = start_time + assigned_duration

            # 午夜截断
            effective_end = min(end_time, 24.0)
            if end_time > 24.0:
                end_time = 24.0

            # 更新终端释放时间
            if assigned_terminal[0] == '480':
                free_480[assigned_terminal[1]] = effective_end
            else:
                free_120[assigned_terminal[1]] = effective_end

            # 计算与各小时的精确重叠
            start_h = int(np.floor(start_time))
            end_h = int(np.floor(effective_end))
            for t in range(start_h, min(end_h + 1, 24)):
                overlap = min(t + 1, effective_end) - max(t, start_time)
                if overlap > 0:
                    hourly_load[t] += assigned_power * overlap

        self._last_rejected = n_rejected
        self._last_queued = n_queued
        self._last_early_departure = n_early_departure
        self._last_failure_rejected = n_failure_rejected  # v6.5
        return hourly_load

    def simulate_monte_carlo(self, day_type='workday', n_runs=10000, month=None, weather='clear'):
        """蒙特卡洛多次仿真, 返回统计结果"""
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


def print_summary(scenarios, sim=None):
    """打印仿真结果摘要"""
    print("\n" + "=" * 70)
    print("蒙特卡洛充电负荷仿真结果汇总")
    print("=" * 70)
    for day_type, result in scenarios.items():
        print(f"\n{day_type}:")
        print(f"  峰值负荷 (均值 ± std):  {result['peak_mean']:.1f} ± {result['peak_std']:.1f} kW")
        print(f"  峰值负荷 (P95):          {result['p95'].max():.1f} kW")
        print(f"  日均用电量 (均值 ± std): {result['daily_energy_mean']:.1f} ± {result['daily_energy_std']:.1f} kWh")
        print(f"  最大峰值时段:            {result['mean'].argmax()}:00")
    if sim is not None:
        print(f"\n  排队/弃充/早离/故障统计 (最后一轮):")
        print(f"    排队: {sim._last_queued} 辆, 弃充: {sim._last_rejected} 辆, "
              f"提前离充: {getattr(sim, '_last_early_departure', 0)} 辆, "
              f"因故障: {getattr(sim, '_last_failure_rejected', 0)} 辆")


if __name__ == '__main__':
    sim = MonteCarloChargingSimulator(service_area_size='medium', seed=42)
    scenarios = sim.simulate_all_scenarios(n_runs=3000)
    print_summary(scenarios, sim)
