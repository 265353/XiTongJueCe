"""
微网拓扑架构量化对比 — AC / DC / 交直流混合 / 多服务区链式 四种方案

理论来源: 文件07_微网系统架构.md §2 (四种拓扑架构方案)
         文件13_细粒度建模数据_设备参数与经济数据.md
         Wang et al. (2025) Energy: 多服务区链式微网

运行:
    python topology_comparison.py      # 模块自测
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    SERVICE_AREA_CONFIG, PV_COST, ESS_COST, CHARGING_COST,
    FIXED_COST, LIFESPAN, OM_RATE,
    MTBF, MTTR, get_availability,  # v6.3: MTBF/可用率数据 (文件24 §3.3)
)

# ============================================================
# 拓扑架构定义 (文件07 §2)
# ============================================================

# 设备效率参数
EFFICIENCY = {
    # 电力电子变换效率
    'pv_inverter': 0.985,         # 光伏逆变器
    'pcs': 0.975,                 # 储能PCS (AC/DC双向)
    'dc_dc_converter': 0.975,    # DC/DC变换器
    'ac_dc_rectifier': 0.970,    # AC/DC整流器 (电网侧)
    'dc_ac_inverter': 0.965,     # DC/AC逆变器 (直流母线→交流负荷)
    'charging_pile_ac': 0.945,   # 交流充电桩 (含车载OBC)
    'charging_pile_dc': 0.955,   # 直流充电桩
    'transformer': 0.990,        # 变压器
}

# 保护设备成本差异 (相对于交流方案)
PROTECTION_COST_FACTOR = {
    'ac': 1.00,         # 交流保护: 基准 (断路器+熔断器+继电器)
    'dc': 1.55,         # 直流保护: 贵55% (直流断路器+固态开关+电弧管理)
    'hybrid': 1.25,     # 混合: 贵25% (两侧均需保护)
    'ring': 1.30,       # 链式: 贵30% (额外通信+协调保护)
}

# 控制复杂度因子
CONTROL_COMPLEXITY = {
    'ac': 1.0,
    'dc': 1.4,
    'hybrid': 1.6,
    'ring': 2.0,
}


class TopologyAnalyzer:
    """微网拓扑架构量化对比分析器

    对比维度:
    1. 综合效率 (含各级变换损耗, v6.3: 负载率依赖)
    2. 设备投资差异
    3. 可靠性评估
    4. 适用场景匹配
    """

    # v6.3: 电力电子设备在不同负载率下的效率衰减
    # 轻载(<30%)时效率显著低于额定值
    _PE_LOAD_DERATE = {
        0.05: 0.82, 0.10: 0.87, 0.20: 0.92, 0.30: 0.95,
        0.50: 0.98, 0.75: 0.99, 1.00: 1.00,
    }

    @staticmethod
    def _pe_efficiency_at_load(base_eff, load_ratio):
        """v6.3: 电力电子设备在给定负载率下的实际效率

        base_eff: 额定效率 (满载)
        load_ratio: 当前负载率 (0-1)
        """
        load_ratio = np.clip(load_ratio, 0.01, 1.0)
        loads = np.array(sorted(TopologyAnalyzer._PE_LOAD_DERATE.keys()))
        derates = np.array([TopologyAnalyzer._PE_LOAD_DERATE[l] for l in loads])
        derate = float(np.interp(load_ratio, loads, derates))
        # 实际效率不能超过额定值
        return min(base_eff, base_eff / 0.985 * derate)

    def __init__(self, area_size='medium', pv_cap=1231, ess_cap=2000, ess_pow=1000):
        cfg = SERVICE_AREA_CONFIG[area_size]
        self.area_size = area_size
        self.pv_cap = pv_cap
        self.ess_cap = ess_cap
        self.ess_pow = ess_pow
        self.n_piles_120 = cfg['n_piles_120kw']
        self.n_piles_480 = cfg['n_piles_480kw']
        self.peak_building = cfg['peak_building_kw']
        self.building_area = cfg['building_area_m2']

    # ============================================================
    # 效率分析 — 基于实际能量流路径的综合效率
    #
    # 方法: 将系统效率分解为各路能量流路径的加权效率
    #   eta_system = Sum_over_paths (flow_ratio_i * eta_path_i)
    # 其中 flow_ratio_i 为各路径占比, 总和=1
    # 路径包括: PV→AC负载, PV→充电, PV→储能, 储能→负载, 电网→负载, 电网→充电
    # ============================================================

    def _compute_path_efficiency(self, pv_ratio, chg_ratio, bldg_ratio):
        """计算所有能量流路径的效率矩阵 (v6.3: 含负载率修正)

        电力电子设备在轻载时效率下降, 重载时效率接近额定值.
        使用PV占比作为系统负载率的代理变量.

        返回四种拓扑在各路径上的效率, 以及综合加权效率
        """
        # v6.3: 系统负载率 (PV越高→转换设备负载越高)
        total = pv_ratio + chg_ratio + bldg_ratio
        if total == 0:
            total = 1.0
        pv_r, chg_r, bldg_r = pv_ratio / total, chg_ratio / total, bldg_ratio / total
        # 系统负载率: PV比例越大, 电力电子越接近满载
        sys_load = np.clip(pv_r * 2.0, 0.1, 1.0)

        def pe_eff(key):
            """获取负载率修正后的电力电子效率"""
            return self._pe_efficiency_at_load(EFFICIENCY[key], sys_load)

        # 能量流路径占比 (典型场景):
        # 路径1: PV直接供AC负载 (pv_r × 0.3)
        # 路径2: PV直接供充电 (pv_r × 0.5, 优先供给高价值充电负荷)
        # 路径3: PV→储能→负载 (pv_r × 0.2)
        # 路径4: 电网→充电 (剩余充电需求, chg_r - pv_r×0.5)
        # 路径5: 电网→建筑负载 (bldg_r - pv_r×0.3)
        # 路径6: 储能放电→负载 (取决于调度)

        grid_to_chg = max(0, chg_r - pv_r * 0.5)
        grid_to_bldg = max(0, bldg_r - pv_r * 0.3)
        pv_to_bldg = min(pv_r * 0.3, bldg_r)
        pv_to_chg = min(pv_r * 0.5, chg_r)
        grid_total = grid_to_chg + grid_to_bldg + max(0, chg_r - pv_to_chg)

        paths = {}

        # AC拓扑路径效率 (v6.3: 负载率修正)
        paths['AC'] = {
            'pv_to_bldg': pe_eff('pv_inverter'),
            'pv_to_chg': pe_eff('pv_inverter') * pe_eff('charging_pile_dc'),
            'pv_to_ess': pe_eff('pv_inverter') * pe_eff('pcs'),
            'ess_to_load': pe_eff('pcs'),
            'grid_to_load': pe_eff('transformer'),
            'grid_to_chg': pe_eff('transformer') * pe_eff('charging_pile_dc'),
        }

        # DC拓扑路径效率 (减少变换级数)
        paths['DC'] = {
            'pv_to_bldg': (pe_eff('dc_dc_converter') *
                          pe_eff('ac_dc_rectifier') *
                          pe_eff('dc_ac_inverter')),
            'pv_to_chg': pe_eff('dc_dc_converter') ** 2,
            'pv_to_ess': pe_eff('dc_dc_converter') ** 2,
            'ess_to_load': pe_eff('dc_dc_converter') * pe_eff('dc_ac_inverter'),
            'grid_to_load': (pe_eff('transformer') *
                            pe_eff('ac_dc_rectifier') *
                            pe_eff('dc_ac_inverter')),
            'grid_to_chg': (pe_eff('transformer') *
                           pe_eff('ac_dc_rectifier') *
                           pe_eff('dc_dc_converter')),
        }

        # Hybrid拓扑路径效率
        paths['Hybrid'] = {
            'pv_to_bldg': pe_eff('pv_inverter'),
            'pv_to_chg': pe_eff('dc_dc_converter'),
            'pv_to_ess': pe_eff('dc_dc_converter') * pe_eff('pcs'),
            'ess_to_load': pe_eff('pcs'),
            'grid_to_load': pe_eff('transformer'),
            'grid_to_chg': (pe_eff('transformer') *
                           pe_eff('ac_dc_rectifier') *
                           pe_eff('dc_dc_converter')),
        }

        # Ring拓扑路径效率 (AC基础 + 区间传输损耗)
        ring_penalty = 0.98
        paths['Ring'] = {
            'pv_to_bldg': pe_eff('pv_inverter') * ring_penalty,
            'pv_to_chg': pe_eff('pv_inverter') * pe_eff('charging_pile_dc') * ring_penalty,
            'pv_to_ess': pe_eff('pv_inverter') * pe_eff('pcs') * ring_penalty,
            'ess_to_load': pe_eff('pcs') * ring_penalty,
            'grid_to_load': pe_eff('transformer') * ring_penalty,
            'grid_to_chg': pe_eff('transformer') * pe_eff('charging_pile_dc') * ring_penalty,
        }

        # 流路径权重 (文献校准)
        # 典型高速公路服务区: PV优先供充电(高价值), 其次建筑, 剩余储能
        flow_weights = {
            'pv_to_bldg': pv_to_bldg,
            'pv_to_chg': pv_to_chg,
            'pv_to_ess': pv_r * 0.2,
            'ess_to_load': pv_r * 0.15,   # 储能放电主要服务剩余需求
            'grid_to_load': grid_to_bldg,
            'grid_to_chg': grid_to_chg,
        }

        # 综合加权效率
        results = {}
        for topo in ['AC', 'DC', 'Hybrid', 'Ring']:
            eta_sum = 0.0
            weight_sum = 0.0
            for path, weight in flow_weights.items():
                if weight > 0:
                    eta_sum += weight * paths[topo][path]
                    weight_sum += weight
            results[topo] = eta_sum / max(weight_sum, 1e-6)

        return results

    def _ac_topology_efficiency(self, pv_ratio, chg_ratio, bldg_ratio):
        return self._compute_path_efficiency(pv_ratio, chg_ratio, bldg_ratio)['AC']

    def _dc_topology_efficiency(self, pv_ratio, chg_ratio, bldg_ratio):
        return self._compute_path_efficiency(pv_ratio, chg_ratio, bldg_ratio)['DC']

    def _hybrid_topology_efficiency(self, pv_ratio, chg_ratio, bldg_ratio):
        return self._compute_path_efficiency(pv_ratio, chg_ratio, bldg_ratio)['Hybrid']

    def _ring_topology_efficiency(self, pv_ratio, chg_ratio, bldg_ratio):
        return self._compute_path_efficiency(pv_ratio, chg_ratio, bldg_ratio)['Ring']

    # ============================================================
    # 投资差异
    # ============================================================

    def capex_differential(self):
        """四种拓扑的设备投资差异 (元)

        基准: AC拓扑 = 当前config.py的设备投资
        其他方案在基准上按保护/变换设备差异调整
        """
        # AC基准投资
        capex_base = {
            'pv': self.pv_cap * sum(PV_COST.values()),
            'ess': (self.ess_cap * (ESS_COST['battery_per_kwh'] + ESS_COST['fire_per_kwh']) +
                    self.ess_pow * ESS_COST['pcs_per_kw']),
            'charging': (self.n_piles_120 * CHARGING_COST['pile_120kw'] +
                         self.n_piles_480 * CHARGING_COST['pile_480kw']),
            'fixed': sum(FIXED_COST.values()),
        }
        total_ac = sum(capex_base.values())

        # DC拓扑: 增加DC/DC变换器(光伏侧+充电侧), 直流保护贵55%
        # 减少逆变器成本(光伏侧只用DC/DC), 增加中央AC/DC整流器
        dc_dc_cost = (self.pv_cap * 100 +  # PV侧DC/DC (100元/kW)
                      self.ess_cap * 80 +   # ESS侧DC/DC
                      (self.n_piles_120 * 15000 + self.n_piles_480 * 80000))  # 充电侧DC/DC
        central_acdc = 800 * 250  # 中央AC/DC 800kW×250元/kW
        dc_extra = dc_dc_cost + central_acdc
        dc_save = self.pv_cap * 150  # 节省逆变器
        protection_delta = capex_base['fixed'] * (PROTECTION_COST_FACTOR['dc'] - 1)
        total_dc = total_ac + dc_extra - dc_save + protection_delta

        # 混合拓扑: 两侧设备 + 互联变换器
        interlink_cost = 500 * 300  # 双向AC/DC 500kW×300元/kW
        hybrid_extra = dc_dc_cost * 0.5 + interlink_cost
        hybrid_save = dc_save * 0.5
        protection_delta_hybrid = capex_base['fixed'] * (PROTECTION_COST_FACTOR['hybrid'] - 1)
        total_hybrid = total_ac + hybrid_extra - hybrid_save + protection_delta_hybrid

        # 链式拓扑: AC基础 + 通信/协调
        comm_cost = 50e4  # 通信系统50万
        ring_extra = comm_cost
        protection_delta_ring = capex_base['fixed'] * (PROTECTION_COST_FACTOR['ring'] - 1)
        total_ring = total_ac + ring_extra + protection_delta_ring

        return {
            'AC': {'total_wan': total_ac / 1e4,
                   'protection_factor': PROTECTION_COST_FACTOR['ac'],
                   'control_complexity': CONTROL_COMPLEXITY['ac'],
                   'label': 'AC (交流)'},
            'DC': {'total_wan': total_dc / 1e4,
                   'protection_factor': PROTECTION_COST_FACTOR['dc'],
                   'control_complexity': CONTROL_COMPLEXITY['dc'],
                   'label': 'DC (直流)'},
            'Hybrid': {'total_wan': total_hybrid / 1e4,
                       'protection_factor': PROTECTION_COST_FACTOR['hybrid'],
                       'control_complexity': CONTROL_COMPLEXITY['hybrid'],
                       'label': 'Hybrid (混合)'},
            'Ring': {'total_wan': total_ring / 1e4,
                     'protection_factor': PROTECTION_COST_FACTOR['ring'],
                     'control_complexity': CONTROL_COMPLEXITY['ring'],
                     'label': 'Ring (链式)'},
        }

    # ============================================================
    # 综合对比
    # ============================================================

    def comprehensive_comparison(self):
        """四方案多维度综合对比"""
        # 典型能量流比例: PV占30%, 充电占55%, 建筑占15%
        pv_r, chg_r, bldg_r = 0.30, 0.55, 0.15

        topologies = {
            'AC': {
                'label': 'AC (交流)',
                'efficiency': self._ac_topology_efficiency(pv_r, chg_r, bldg_r),
                'pros': ['技术最成熟', '设备标准化高', '施工简单', '运维经验丰富'],
                'cons': ['多级变换损耗(3-5%)', 'PV->充电需DC->AC->DC'],
                'conversion_stages_pv2chg': 2,
                'applicable': '新建及改造项目主流方案',
                'maturity': 5,
                'renewable_utilization': '基准',
            },
            'DC': {
                'label': 'DC (直流)',
                'efficiency': self._dc_topology_efficiency(pv_r, chg_r, bldg_r),
                'pros': ['减少变换环节', '综合效率+3-5%', '光伏直供充电'],
                'cons': ['直流保护设备贵55%', '标准化程度低', '交流负荷需额外逆变'],
                'conversion_stages_pv2chg': 1,
                'applicable': '高比例充电+光伏场景, 新建项目',
                'maturity': 3,
                'renewable_utilization': '+5-8%',
            },
            'Hybrid': {
                'label': 'Hybrid (交直流混合)',
                'efficiency': self._hybrid_topology_efficiency(pv_r, chg_r, bldg_r),
                'pros': ['兼顾AC/DC优势', 'DC侧高效供充电', 'AC侧成熟供建筑'],
                'cons': ['系统最复杂', '需AC/DC互联变换器', '控制策略复杂'],
                'conversion_stages_pv2chg': 1,
                'applicable': '大型综合服务区, 多类型负荷',
                'maturity': 2,
                'renewable_utilization': '+3-6%',
            },
            'Ring': {
                'label': 'Ring (多服务区链式)',
                'efficiency': self._ring_topology_efficiency(pv_r, chg_r, bldg_r),
                'pros': ['区间功率互济', '可再生能源利用率+20.2%', '自维持率+6.4%'],
                'cons': ['需通信协调', '区间输电损耗', '故障隔离复杂'],
                'conversion_stages_pv2chg': 2,
                'applicable': '高速公路带状分布, 多服务区联动',
                'maturity': 2,
                'renewable_utilization': '+15-22%',
            },
        }

        # 添加工况效率矩阵
        efficiency_matrix = {}
        for name in topologies:
            row = {}
            for weather in ['clear', 'cloudy', 'night']:
                if weather == 'clear':
                    p, c, b = 0.45, 0.40, 0.15
                elif weather == 'cloudy':
                    p, c, b = 0.15, 0.55, 0.30
                else:
                    p, c, b = 0.0, 0.20, 0.80
                if name == 'AC':
                    row[weather] = self._ac_topology_efficiency(p, c, b)
                elif name == 'DC':
                    row[weather] = self._dc_topology_efficiency(p, c, b)
                elif name == 'Hybrid':
                    row[weather] = self._hybrid_topology_efficiency(p, c, b)
                else:
                    row[weather] = self._ring_topology_efficiency(p, c, b)
            efficiency_matrix[name] = row

        return topologies, efficiency_matrix

    def print_comparison(self):
        """打印完整对比报告"""
        topologies, eff_matrix = self.comprehensive_comparison()
        capex = self.capex_differential()

        print("\n" + "=" * 68)
        print(f"微网拓扑架构量化对比 — {self.area_size}服务区")
        print(f"  (PV={self.pv_cap}kWp, ESS={self.ess_cap}kWh/{self.ess_pow}kW)")
        print("=" * 68)

        # 效率对比表
        print(f"\n--- 综合效率对比 (不同工况) ---")
        print(f"{'拓扑':<20} {'晴天':>8} {'阴天':>8} {'夜间':>8} {'综合':>8}")
        print("-" * 56)
        for name, info in topologies.items():
            effs = eff_matrix[name]
            avg = np.mean(list(effs.values()))
            label = info.get('label', name)
            print(f"  {label:<18} {effs['clear']:>7.1%} {effs['cloudy']:>7.1%} "
                  f"{effs['night']:>7.1%} {avg:>7.1%}")

        # 投资对比表
        print(f"\n--- 设备投资对比 ---")
        print(f"{'拓扑':<20} {'总投资(万)':>10} {'保护系数':>8} {'控制复杂度':>10}")
        print("-" * 52)
        for name, data in capex.items():
            label = data.get('label', name)
            print(f"  {label:<18} {data['total_wan']:>8.1f}  "
                  f"{data['protection_factor']:>6.2f}  {data['control_complexity']:>8.1f}")

        # 优劣势定性对比
        print(f"\n--- 定性对比 ---")
        for name, info in topologies.items():
            label = info.get('label', name)
            print(f"\n  [{label}]")
            print(f"    效率: {info['efficiency']:.1%}")
            print(f"    PV->充电变换级数: {info['conversion_stages_pv2chg']}")
            mat_str = '*' * info['maturity'] + '-' * (5 - info['maturity'])
            print(f"    技术成熟度: {mat_str}")
            print(f"    可再生能源利用: {info['renewable_utilization']}")
            print(f"    优势: {', '.join(info['pros'][:3])}")
            print(f"    劣势: {', '.join(info['cons'][:3])}")
            print(f"    适用: {info['applicable']}")

        # 推荐
        print(f"\n--- 推荐 ---")
        print(f"  一般场景: AC微网 (成熟可靠, 成本最优)")
        print(f"  高充电比例: DC微网 (效率提升3-5%)")
        print(f"  大型综合: 交直流混合 (兼顾效率与可靠性)")
        print(f"  带状分布: 链式微网 (可再生能源利用率+20%)")

    def get_soi_scores(self):
        """返回用于TOPSIS决策的拓扑方案评分矩阵

        六个维度: 效率/成本/可靠性/技术成熟度/可扩展性/运维复杂度
        """
        capex = self.capex_differential()
        topologies, eff_matrix = self.comprehensive_comparison()

        names = list(capex.keys())
        n = len(names)
        # 六个维度: efficiency, cost, reliability, maturity, scalability, complexity
        scores = np.zeros((n, 6))

        # v6.3: 组件可靠度基于MTBF/MTTR (文件24 §3.3)
        # R = MTBF / (MTBF + MTTR), 串联可靠度: R_total = prod(R_i ^ count_i)
        _avail_inv = get_availability(MTBF['pv_inverter_string'], MTTR['pv_inverter'])
        _avail_pcs = get_availability(MTBF['ess_pcs'], MTTR['ess_pcs'])
        _avail_dcdc = get_availability(MTBF['dc_dc_converter'], MTTR['dc_dc_converter'])
        _avail_xfmr = get_availability(MTBF['transformer'], 0.01)  # 变压器MTTR≈0
        comp_reliability = {
            'pv_inverter': _avail_inv,       # 0.99997 (华为实测)
            'transformer': _avail_xfmr,       # ~0.99999
            'pcs': _avail_pcs,                # 0.99992
            'dc_dc': _avail_dcdc,             # 0.99996
            'ac_dc': _avail_pcs,              # v6.3: 使用PCS可用率近似
            'dc_ac': _avail_pcs,              # v6.3: 使用PCS可用率近似
        }
        # 主要串联组件数量 (各拓扑特有)
        comp_counts = {
            'AC': {'pv_inverter': 1, 'transformer': 1, 'pcs': 0, 'dc_dc': 0, 'ac_dc': 0, 'dc_ac': 0},
            'DC': {'pv_inverter': 0, 'transformer': 1, 'pcs': 0, 'dc_dc': 2, 'ac_dc': 1, 'dc_ac': 1},
            'Hybrid': {'pv_inverter': 1, 'transformer': 1, 'pcs': 1, 'dc_dc': 1, 'ac_dc': 1, 'dc_ac': 0},
            'Ring': {'pv_inverter': 1, 'transformer': 1, 'pcs': 0, 'dc_dc': 0, 'ac_dc': 0, 'dc_ac': 0},
        }

        for i, name in enumerate(names):
            eff_avg = np.mean(list(eff_matrix[name].values()))
            scores[i, 0] = eff_avg * 100  # 效率百分比
            scores[i, 1] = capex[name]['total_wan']  # 万元
            # 串联组件可靠性 (v6.2: 动态计算替代硬编码)
            cnt = comp_counts.get(name, comp_counts['AC'])
            reliability = 1.0
            for comp, count in cnt.items():
                if count > 0:
                    reliability *= comp_reliability[comp] ** count
            scores[i, 2] = reliability * 100  # 可靠性 %
            scores[i, 3] = {'AC': 5, 'DC': 3, 'Hybrid': 2, 'Ring': 2}.get(name, 3)  # 成熟度
            scores[i, 4] = {'AC': 3, 'DC': 4, 'Hybrid': 5, 'Ring': 5}.get(name, 3)  # 可扩展性
            scores[i, 5] = capex[name]['control_complexity']  # 复杂度

        return names, scores


def self_test():
    """模块自测"""
    analyzer = TopologyAnalyzer(area_size='medium', pv_cap=1231,
                                ess_cap=2000, ess_pow=1000)
    analyzer.print_comparison()

    # 输出TOPSIS评分矩阵
    names, scores = analyzer.get_soi_scores()
    print(f"\n--- TOPSIS评分矩阵 (供决策模块使用) ---")
    dims = ['效率(%)↑', '成本(万)↓', '可靠性(%)↑', '成熟度↑', '可扩展性↑', '复杂度↓']
    print(f"{'拓扑':<20} " + " ".join(f"{d:>10}" for d in dims))
    for i, name in enumerate(names):
        print(f"  {name:<18} " + " ".join(f"{scores[i,j]:>10.1f}" for j in range(6)))


if __name__ == '__main__':
    self_test()
