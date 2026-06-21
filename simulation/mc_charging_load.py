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
    HOURLY_ARRIVAL_RATE,
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

    def simulate_day(self, day_type='workday', month=None):
        """模拟一天的充电负荷 (双类终端排队模型 + GMM到达)

        终端分为两组:
        - 120kW快充桩: 数量 n_120kw, 最大输出120kW
        - 480kW超充终端: 数量 n_480kw*5, 最大输出480kW

        车辆优先选择480kW终端 (若车型支持>120kW),
        480kW忙时降级到120kW, 全忙时排队等待最先释放的终端.

        Parameters
        ----------
        day_type : str
            日类型
        month : int or None
            月份(1-12), 用于季节能耗和GMM到达微调
        """
        hourly_load = np.zeros(24)
        arrival_rates = HOURLY_ARRIVAL_RATE.get(day_type, HOURLY_ARRIVAL_RATE['workday'])
        total_daily_arrivals = int(sum(arrival_rates))

        from config import SERVICE_AREA_CONFIG
        cfg = SERVICE_AREA_CONFIG.get(self.size, SERVICE_AREA_CONFIG['medium'])
        n_120kw = cfg['n_piles_120kw']           # 120kW终端数
        n_480kw = cfg['n_piles_480kw'] * 5         # 480kW终端数

        # 两组终端的释放时间
        free_120 = np.zeros(n_120kw) if n_120kw > 0 else np.array([])
        free_480 = np.zeros(n_480kw) if n_480kw > 0 else np.array([])

        n_rejected = 0
        n_queued = 0

        patience = {'workday': 0.75, 'weekend': 0.50,
                     'holiday': 0.33, 'spring_festival': 0.25}
        max_wait = patience.get(day_type, 0.5)

        # 文件05 GMM到达时间模型: 按天总量采样到达时刻
        actual_arrivals = self.rng.poisson(total_daily_arrivals)
        actual_arrivals = max(0, min(actual_arrivals, int(total_daily_arrivals * 1.5)))

        # 从GMM采样到达时刻
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
            hour = int(arrival_time)
            vehicle = self._sample_vehicle(month=month)

            # 判断车辆是否能从480kW终端获益
            can_use_480 = (vehicle['max_power'] > 120 or vehicle['power'] > 120)

            # 确定在每种终端上的有效充电功率
            eff_power_480 = min(vehicle['power'], vehicle['max_power'])
            eff_power_120 = min(vehicle['power'], vehicle['max_power'], 120)

            # 充电时长 (取决于终端类型)
            dur_480 = vehicle['charge_needed'] / (eff_power_480 * CHARGE_EFFICIENCY) if eff_power_480 > 0 else 99
            dur_120 = vehicle['charge_needed'] / (eff_power_120 * CHARGE_EFFICIENCY) if eff_power_120 > 0 else 99
            dur_480 = min(dur_480, 1.0)
            dur_120 = min(dur_120, 2.0)

            earliest_480 = free_480.min() if len(free_480) > 0 else np.inf
            earliest_120 = free_120.min() if len(free_120) > 0 else np.inf

            # 选择终端: 优先480kW (若可用且车辆能获益), 否则120kW
            assigned_terminal = None
            assigned_duration = None
            assigned_power = None

            if can_use_480 and arrival_time >= earliest_480:
                # 480kW终端空闲, 优先使用
                tid = free_480.argmin()
                assigned_terminal = ('480', tid)
                assigned_duration = dur_480
                assigned_power = eff_power_480
            elif arrival_time >= earliest_120:
                # 120kW终端空闲
                tid = free_120.argmin()
                assigned_terminal = ('120', tid)
                assigned_duration = dur_120
                assigned_power = eff_power_120
            elif can_use_480 and earliest_480 <= earliest_120 and len(free_480) > 0:
                # 全忙, 但480kW终端先释放
                wait_time = earliest_480 - arrival_time
                if wait_time > max_wait:
                    n_rejected += 1
                    continue
                n_queued += 1
                tid = free_480.argmin()
                assigned_terminal = ('480', tid)
                assigned_duration = dur_480
                assigned_power = eff_power_480
                arrival_time = earliest_480
            elif len(free_120) > 0:
                # 全忙, 120kW终端先释放 (或车辆不能使用480kW)
                earliest = earliest_120
                wait_time = earliest - arrival_time
                if wait_time > max_wait:
                    n_rejected += 1
                    continue
                n_queued += 1
                tid = free_120.argmin()
                assigned_terminal = ('120', tid)
                assigned_duration = dur_120
                assigned_power = eff_power_120
                arrival_time = earliest
            elif len(free_480) > 0:
                # 只有480kW可用且车辆能使用
                earliest = earliest_480
                wait_time = earliest - arrival_time
                if wait_time > max_wait:
                    n_rejected += 1
                    continue
                n_queued += 1
                tid = free_480.argmin()
                assigned_terminal = ('480', tid)
                assigned_duration = dur_480
                assigned_power = eff_power_480
                arrival_time = earliest
            else:
                # 无可用终端 (n_120kw=0 且 n_480kw=0)
                n_rejected += 1
                continue

            start_time = arrival_time
            end_time = start_time + assigned_duration

            # 午夜截断
            effective_end = min(end_time, 24.0)
            if end_time > 24.0:
                end_time = 24.0

            # 更新终端释放时间
            if assigned_terminal[0] == '480':
                free_480[assigned_terminal[1]] = end_time
            else:
                free_120[assigned_terminal[1]] = end_time

            # 计算与各小时的精确重叠
            start_h = int(np.floor(start_time))
            end_h = int(np.floor(effective_end))
            for t in range(start_h, min(end_h + 1, 24)):
                overlap = min(t + 1, effective_end) - max(t, start_time)
                if overlap > 0:
                    hourly_load[t] += assigned_power * overlap

        self._last_rejected = n_rejected
        self._last_queued = n_queued
        return hourly_load

    def simulate_monte_carlo(self, day_type='workday', n_runs=10000):
        """蒙特卡洛多次仿真, 返回统计结果"""
        all_runs = np.zeros((n_runs, 24))

        for i in range(n_runs):
            all_runs[i] = self.simulate_day(day_type)

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


def print_summary(scenarios):
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


if __name__ == '__main__':
    sim = MonteCarloChargingSimulator(service_area_size='medium', seed=42)
    scenarios = sim.simulate_all_scenarios(n_runs=3000)
    print_summary(scenarios)
