"""
建筑负荷 Agent-Based 建模 — 基于人员行为模拟的动态负荷生成

理论来源: 文件06_服务区用能负荷特征.md §4.1 (主体建模参数)
         文件12_细粒度建模数据_建筑用能负荷.md

与config.py中的固定季节曲线互补: ABM提供日内随机波动和日间差异,
固定曲线提供基准剖面.

运行:
    python building_load_abm.py        # 模块自测
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    SERVICE_AREA_CONFIG, BUILDING_LOAD,
    WEATHER_BUILDING_COEFF, get_season,
)


class OccupantAgent:
    """单个建筑人员Agent

    状态:
    - present: 是否在建筑内
    - activity: 当前活动 (sleep/rest/cook/eat/work/leisure)
    - room: 所在空间类型
    """

    ACTIVITIES = ['sleep', 'rest', 'cook', 'eat', 'work', 'leisure']

    # 每种活动的用电设备及其功率 (W)
    ACTIVITY_LOADS = {
        'sleep':   {'lighting': 5, 'ac_fan': 20},
        'rest':    {'lighting': 15, 'ac': 80, 'tv': 100},
        'cook':    {'lighting': 20, 'ac': 100, 'cooking': 3000, 'exhaust': 200},
        'eat':     {'lighting': 20, 'ac': 100, 'hot_water': 500},
        'work':    {'lighting': 20, 'ac': 80, 'computer': 150},
        'leisure': {'lighting': 20, 'ac': 80, 'tv': 100, 'phone_charge': 10},
    }

    def __init__(self, agent_type='staff', rng=None):
        self.agent_type = agent_type  # 'staff', 'guest', 'driver'
        self.rng = rng if rng is not None else np.random.RandomState()
        self._init_behavior()

    def _init_behavior(self):
        """初始化行为参数"""
        if self.agent_type == 'staff':
            # 驻守员工: 工作日在岗, 有固定作息
            self.present_prob = {
                'night': 0.95,    # 23-6: 在宿舍
                'morning': 0.98,  # 6-9: 起床备餐
                'forenoon': 0.99,  # 9-12: 工作
                'noon': 0.85,     # 12-14: 午休/部分外出
                'afternoon': 0.99, # 14-18: 工作
                'evening': 0.90,  # 18-23: 在岗/休息
            }
            self.activity_probs = {
                'night':     {'sleep': 0.85, 'rest': 0.10, 'leisure': 0.05},
                'morning':   {'cook': 0.30, 'eat': 0.25, 'rest': 0.25, 'work': 0.20},
                'forenoon':  {'work': 0.60, 'rest': 0.25, 'cook': 0.10, 'leisure': 0.05},
                'noon':      {'eat': 0.35, 'cook': 0.25, 'rest': 0.25, 'work': 0.15},
                'afternoon': {'work': 0.55, 'rest': 0.30, 'leisure': 0.10, 'cook': 0.05},
                'evening':   {'leisure': 0.35, 'eat': 0.25, 'rest': 0.20, 'work': 0.15, 'cook': 0.05},
            }
        elif self.agent_type == 'guest':
            # 旅客/顾客: 高流动性, 随机停留
            self.present_prob = {
                'night': 0.30,
                'morning': 0.40,
                'forenoon': 0.65,
                'noon': 0.85,
                'afternoon': 0.70,
                'evening': 0.55,
            }
            self.activity_probs = {
                'night':     {'sleep': 0.70, 'rest': 0.20, 'leisure': 0.10},
                'morning':   {'eat': 0.45, 'rest': 0.25, 'leisure': 0.20, 'cook': 0.10},
                'forenoon':  {'leisure': 0.35, 'eat': 0.25, 'rest': 0.25, 'work': 0.15},
                'noon':      {'eat': 0.50, 'rest': 0.25, 'leisure': 0.15, 'cook': 0.10},
                'afternoon': {'leisure': 0.35, 'rest': 0.30, 'eat': 0.20, 'work': 0.15},
                'evening':   {'eat': 0.35, 'leisure': 0.35, 'rest': 0.20, 'work': 0.10},
            }
        else:  # driver
            self.present_prob = {
                'night': 0.10,
                'morning': 0.25,
                'forenoon': 0.50,
                'noon': 0.70,
                'afternoon': 0.55,
                'evening': 0.35,
            }
            self.activity_probs = {
                'night':     {'rest': 0.60, 'sleep': 0.30, 'leisure': 0.10},
                'morning':   {'eat': 0.40, 'rest': 0.35, 'leisure': 0.25},
                'forenoon':  {'rest': 0.40, 'leisure': 0.30, 'eat': 0.30},
                'noon':      {'eat': 0.45, 'rest': 0.35, 'leisure': 0.15, 'cook': 0.05},
                'afternoon': {'rest': 0.40, 'leisure': 0.30, 'eat': 0.25, 'cook': 0.05},
                'evening':   {'eat': 0.35, 'leisure': 0.30, 'rest': 0.25, 'cook': 0.10},
            }

    def get_hour_state(self, hour, day_type='workday', weather='clear'):
        """获取该人员某时刻的状态"""
        # 时段分类
        if 23 <= hour or hour < 6:
            period = 'night'
        elif 6 <= hour < 9:
            period = 'morning'
        elif 9 <= hour < 12:
            period = 'forenoon'
        elif 12 <= hour < 14:
            period = 'noon'
        elif 14 <= hour < 18:
            period = 'afternoon'
        else:
            period = 'evening'

        # 在室概率 (节假日调整)
        prob = self.present_prob[period]
        if day_type in ['holiday', 'spring_festival']:
            prob = min(1.0, prob * 1.2 if self.agent_type == 'guest' else prob * 0.8)
        elif day_type == 'weekend':
            prob = min(1.0, prob * 1.1 if self.agent_type == 'guest' else prob * 0.9)

        present = self.rng.random() < prob
        if not present:
            return None, {}

        # 活动采样
        probs = self.activity_probs[period]
        acts = list(probs.keys())
        weights = np.array([probs[a] for a in acts])
        weights /= weights.sum()
        activity = acts[self.rng.choice(len(acts), p=weights)]

        # 活动功耗 (含随机波动 ±20%)
        loads = {}
        for device, power in self.ACTIVITY_LOADS[activity].items():
            loads[device] = power * (1 + self.rng.uniform(-0.2, 0.2))

        return activity, loads


class BuildingLoadABM:
    """建筑负荷Agent-Based模型

    基于人员行为模拟的建筑逐时负荷, 与固定季节曲线互补.
    """

    def __init__(self, area_size='medium', seed=42):
        cfg = SERVICE_AREA_CONFIG[area_size]
        self.area_size = area_size
        self.peak_building_kw = cfg['peak_building_kw']
        self.building_area_m2 = cfg['building_area_m2']

        self.rng = np.random.RandomState(seed)

        # 人员配置 (文件06 §4.1: 37名员工 + 住宿 + 宾馆)
        n_staff = 37 if area_size == 'medium' else (25 if area_size == 'small' else 50)
        n_guest_rooms = 20 if area_size == 'medium' else (10 if area_size == 'small' else 30)
        n_guests = int(n_guest_rooms * 0.7)  # 70%入住率

        self.agents = []
        for _ in range(n_staff):
            self.agents.append(OccupantAgent('staff', self.rng))
        for _ in range(n_guests):
            self.agents.append(OccupantAgent('guest', self.rng))

        # 司机/旅客流动: 从MC充电负荷反推人数
        self.n_drivers_base = 80 if area_size == 'medium' else (40 if area_size == 'small' else 150)

        # 基础负荷 (不可调度, 独立于人员: 路灯/弱电/冷柜等)
        self.base_load_kw = self.peak_building_kw * 0.15  # 基础负荷占峰值15%
        self._cal_factor = None  # 延迟校准 (v6.2: 对标文件12设计峰值)

    def simulate_hour(self, hour, month, day_type='workday', weather='clear'):
        """模拟单小时建筑负荷

        Returns
        -------
        total_kw : float
            总建筑负荷 (kW)
        breakdown : dict
            分项负荷
        """
        season = get_season(month)

        # 天气对建筑负荷的影响 (阴雨增加照明和空调)
        wc = WEATHER_BUILDING_COEFF.get(weather, 1.0)

        # 司机人数根据日类型变化
        if day_type == 'spring_festival':
            n_drivers = int(self.n_drivers_base * 1.5)
        elif day_type == 'holiday':
            n_drivers = int(self.n_drivers_base * 1.3)
        elif day_type == 'weekend':
            n_drivers = int(self.n_drivers_base * 1.15)
        else:
            n_drivers = self.n_drivers_base

        # 收集所有agent的负荷
        total_device_loads = {}
        n_present = 0

        for agent in self.agents:
            _, loads = agent.get_hour_state(hour, day_type, weather)
            if loads:
                n_present += 1
                for dev, power in loads.items():
                    total_device_loads[dev] = total_device_loads.get(dev, 0) + power

        for _ in range(n_drivers):
            driver = OccupantAgent('driver', self.rng)
            _, loads = driver.get_hour_state(hour, day_type, weather)
            if loads:
                n_present += 1
                for dev, power in loads.items():
                    total_device_loads[dev] = total_device_loads.get(dev, 0) + power

        # W → kW, 叠加天气影响
        agent_load_kw = sum(total_device_loads.values()) / 1000 * wc

        # 对标到实测峰值: 基准负荷 + agent负荷
        # 固定曲线提供基准剖面, ABM提供随机波动
        base = self.base_load_kw * wc

        total_kw = base + agent_load_kw

        # v6.2: 校准到设计峰值 (文件12: 中型147kW)
        if self._cal_factor is None:
            self._calibrate()
        total_kw *= self._cal_factor

        breakdown = {
            'base': base * self._cal_factor,
            'agent': agent_load_kw * self._cal_factor,
            'total': total_kw,
            'n_present': n_present,
        }

        return total_kw, breakdown

    def _calibrate(self):
        """校准ABM输出到设计峰值 (v6.2: 文件12 中型147kW)

        模拟夏季工作日12:00 (典型峰值时段), 计算缩放因子。
        """
        raw_kw, _ = self._simulate_hour_raw(12, 7, 'workday', 'clear')
        self._cal_factor = self.peak_building_kw / max(raw_kw, 0.1)
        self._cal_factor = np.clip(self._cal_factor, 0.5, 2.0)

    def _simulate_hour_raw(self, hour, month, day_type='workday', weather='clear'):
        """无校准版本 — 供_calibrate内部使用"""
        from config import WEATHER_BUILDING_COEFF
        wc = WEATHER_BUILDING_COEFF.get(weather, 1.0)
        n_staff = 37 if self.area_size == 'medium' else (25 if self.area_size == 'small' else 50)
        n_guests = 14 if self.area_size == 'medium' else (8 if self.area_size == 'small' else 22)
        n_drivers = 80 if self.area_size == 'medium' else (40 if self.area_size == 'small' else 120)

        total_device_loads = {}
        n_present = 0
        for _ in range(n_staff):
            agent = OccupantAgent('staff', self.rng)
            _, loads = agent.get_hour_state(hour, day_type, weather)
            if loads:
                n_present += 1
                for dev, power in loads.items():
                    total_device_loads[dev] = total_device_loads.get(dev, 0) + power
        for _ in range(n_guests):
            agent = OccupantAgent('guest', self.rng)
            _, loads = agent.get_hour_state(hour, day_type, weather)
            if loads:
                n_present += 1
                for dev, power in loads.items():
                    total_device_loads[dev] = total_device_loads.get(dev, 0) + power
        for _ in range(n_drivers):
            agent = OccupantAgent('driver', self.rng)
            _, loads = agent.get_hour_state(hour, day_type, weather)
            if loads:
                n_present += 1
                for dev, power in loads.items():
                    total_device_loads[dev] = total_device_loads.get(dev, 0) + power

        agent_kw = sum(total_device_loads.values()) / 1000 * wc
        base = self.base_load_kw * wc
        return base + agent_kw, {'base': base, 'agent': agent_kw, 'n_present': n_present}

    def simulate_day(self, month, day_type='workday', weather='clear'):
        """模拟一天24h建筑负荷"""
        hourly_kw = np.zeros(24)
        hourly_breakdown = []
        for h in range(24):
            kw, bd = self.simulate_hour(h, month, day_type, weather)
            hourly_kw[h] = kw
            hourly_breakdown.append(bd)
        return hourly_kw, hourly_breakdown

    def simulate_annual(self, calendar_ctx):
        """模拟全年8760h建筑负荷 (需CalendarContext)"""
        hourly = np.zeros(8760)
        hour_idx = 0
        days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        for m in range(1, 13):
            for d in range(days[m - 1]):
                day_idx = sum(days[:m-1]) + d
                dt = calendar_ctx.day_types[day_idx] if hasattr(calendar_ctx, 'day_types') else 'workday'
                weather = calendar_ctx.weather_seq[day_idx] if hasattr(calendar_ctx, 'weather_seq') else 'clear'
                kw_24, _ = self.simulate_day(m, dt, weather)
                for h in range(24):
                    if hour_idx < 8760:
                        hourly[hour_idx] = kw_24[h]
                        hour_idx += 1
        return hourly

    def get_typical_day_curve(self, season='summer', day_type='workday'):
        """生成典型日曲线 (与config.py的BUILDING_LOAD格式兼容)"""
        month = {'spring': 4, 'summer': 7, 'autumn': 10, 'winter': 1}[season]
        kw_24, _ = self.simulate_day(month, day_type, 'clear')
        return kw_24

    def print_summary(self):
        """打印ABM建筑负荷摘要"""
        print("\n" + "=" * 55)
        print(f"建筑负荷 ABM 模型 [{self.area_size}服务区]")
        print(f"  驻守员工: {sum(1 for a in self.agents if a.agent_type=='staff')}")
        print(f"  住宿旅客: {sum(1 for a in self.agents if a.agent_type=='guest')}")
        print(f"  流动司机基数: {self.n_drivers_base}")
        print(f"  建筑峰值(标定): {self.peak_building_kw} kW")

        # 四季典型日对比
        print(f"\n--- 四季典型日负荷 (工作日/晴天) ---")
        print(f"{'时刻':<6}", end="")
        for season in ['spring', 'summer', 'autumn', 'winter']:
            print(f"{season:>10}", end="")
        print()
        for h in range(24):
            print(f"  {h:02d}:00", end="")
            for season in ['spring', 'summer', 'autumn', 'winter']:
                month = {'spring': 4, 'summer': 7, 'autumn': 10, 'winter': 1}[season]
                kw, _ = self.simulate_hour(h, month, 'workday', 'clear')
                print(f"  {kw:>7.1f}", end="")
            print()

        # 日类型对比
        print(f"\n--- 日类型对比 (夏季/晴天) ---")
        print(f"{'时刻':<6}", end="")
        for dt in ['workday', 'weekend', 'holiday', 'spring_festival']:
            print(f"{dt:>14}", end="")
        print()
        for h in range(24):
            print(f"  {h:02d}:00", end="")
            for dt in ['workday', 'weekend', 'holiday', 'spring_festival']:
                kw, _ = self.simulate_hour(h, 7, dt, 'clear')
                print(f"  {kw:>12.1f}", end="")
            print()


def self_test():
    """模块自测"""
    abm = BuildingLoadABM(area_size='medium', seed=42)
    abm.print_summary()


if __name__ == '__main__':
    self_test()
