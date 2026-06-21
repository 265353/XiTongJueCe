"""
独立经济计算模块 — NPC / LCOE / IRR / PBP / DPP / ROI / BCR 全生命周期经济性评估

理论来源: 文件08_系统容量优化配置.md §3 (目标函数体系)
         文件13_细粒度建模数据_设备参数与经济数据.md
         文件01_充电价格与电价数据.md

运行:
    python economic_model.py          # 模块自测 (中型服务区基准方案)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    PV_COST, PV_COST_PER_KWP, ESS_COST, CHARGING_COST, FIXED_COST_TOTAL,
    OM_RATE, REPLACEMENT, LIFESPAN, DISCOUNT_RATE, PROJECT_LIFE,
    RESIDUAL_RATE, PV_FIRST_YEAR_DEGRADATION, PV_ANNUAL_DEGRADATION,
    ESS_CYCLE_LIFE, ESS_CALENDAR_LIFE,
    ESS_CAPACITY_FADE_PER_CYCLE, ESS_CAPACITY_FADE_CALENDAR,
    CCER_PRICE, CCER_PRICE_ESCALATION, CCER_PRICE_FORECAST,  # v6.3: 碳价预测
    PV_SUBSIDY_PER_KWP, ESS_SUBSIDY_PER_KWH, ESS_SUBSIDY_PER_KW,
    CHARGING_SUBSIDY_PER_PILE, MAX_SUBSIDY_RATIO,
    SCENARIOS, CARBON_FACTOR_GRID, FEED_IN_PRICE,
    SERVICE_AREA_CONFIG,
    # v6.2: 分项残值率 + 上网电价涨幅
    RESIDUAL_RATE_PV, RESIDUAL_RATE_ESS, RESIDUAL_RATE_CHARGING,
    FEED_IN_PRICE_ESCALATION,
    # v6.4: S曲线增长 + DR + V2G (文件25)
    LOAD_GROWTH_MODEL, LOAD_GROWTH_BASE_YEAR, get_load_growth_factor,
    DR_ENABLED, DR_EVENTS_PER_YEAR, DR_COMPENSATION_YUAN_KWH,
    DR_DURATION_HOURS, DR_MIN_QUALIFICATION, DR_ESS_SOC_RESERVE,
    estimate_dr_annual_revenue,
    # v6.5: 电池退化与温度/DOD关系
    get_calendar_fade_rate, get_cycle_life_at_dod, get_capacity_fade_per_cycle,
    get_battery_yearly_degradation,
)


class EconomicModel:
    """全生命周期经济性评估引擎

    Parameters
    ----------
    pv_capacity : float
        光伏装机容量 (kWp)
    ess_capacity : float
        储能容量 (kWh)
    ess_power : float
        储能功率 (kW)
    n_piles_120kw : int
        120kW快充桩数量
    n_piles_480kw : int
        480kW超充堆数量
    scenario : str
        情景名称 ('baseline', 'conservative', 'aggressive')
    """

    def __init__(self, pv_capacity=0, ess_capacity=0, ess_power=0,
                 n_piles_120kw=0, n_piles_480kw=0, scenario='baseline'):
        self.pv_cap = pv_capacity
        self.ess_e = ess_capacity
        self.ess_p = ess_power
        self.n_piles_120 = n_piles_120kw
        self.n_piles_480 = n_piles_480kw
        self.sc_params = SCENARIOS.get(scenario, SCENARIOS['baseline'])
        self.scenario = scenario

    # ============================================================
    # 投资成本 (CAPEX)
    # ============================================================

    def capex_pv(self):
        """光伏系统投资 (元)"""
        return self.pv_cap * PV_COST_PER_KWP

    def capex_ess(self):
        """储能系统投资 (元) — 容量+功率"""
        return (self.ess_e * (ESS_COST['battery_per_kwh'] + ESS_COST['fire_per_kwh']) +
                self.ess_p * ESS_COST['pcs_per_kw'])

    def capex_charging(self):
        """充电设施投资 (元)"""
        return (self.n_piles_120 * CHARGING_COST['pile_120kw'] +
                self.n_piles_480 * CHARGING_COST['pile_480kw'])

    def capex_total(self):
        """总投资 (元)"""
        return (self.capex_pv() + self.capex_ess() +
                self.capex_charging() + FIXED_COST_TOTAL)

    # ============================================================
    # 补贴收入
    # ============================================================

    def subsidy_total(self):
        """总补贴 (元), 上限为总投资30%"""
        raw = (self.pv_cap * PV_SUBSIDY_PER_KWP +
               self.ess_e * ESS_SUBSIDY_PER_KWH +
               self.ess_p * ESS_SUBSIDY_PER_KW +
               (self.n_piles_120 + self.n_piles_480) * CHARGING_SUBSIDY_PER_PILE)
        return min(raw, self.capex_total() * MAX_SUBSIDY_RATIO)

    def capex_after_subsidy(self):
        """补贴后投资 (元)"""
        return self.capex_total() - self.subsidy_total()

    # ============================================================
    # 年运维成本 (OPEX)
    # ============================================================

    def opex_annual(self, year=1, escalation_rate=0.02):
        """年运维成本 (元/年), v6.3: 含逐年递增 (设备老化)

        escalation_rate: 年运维递增率, 默认2%/年
        """
        pv_cap_effective = self.pv_cap * self._pv_degradation_factor(year)
        base_opex = (self.capex_pv() * OM_RATE['pv'] * (pv_cap_effective / max(self.pv_cap, 1)) +
                     self.capex_ess() * OM_RATE['ess'] +
                     self.capex_charging() * OM_RATE['charging'])
        # v6.3: 运维成本随设备老化递增
        return base_opex * (1 + escalation_rate) ** (year - 1)

    # ============================================================
    # 购电成本 / 上网收入
    # ============================================================

    def grid_cost_annual(self, annual_grid_import_kwh, annual_tou_prices):
        """年网购电成本 (元/年) — 逐时电价计算"""
        return float(np.sum(annual_grid_import_kwh * annual_tou_prices))

    def grid_revenue_annual(self, annual_grid_export_kwh):
        """年上网收入 (元/年)"""
        return annual_grid_export_kwh * FEED_IN_PRICE

    # ============================================================
    # 碳交易收入
    # ============================================================

    def carbon_revenue_annual(self, self_consumed_kwh, year=1):
        """年碳减排收入 (元/年)

        v6.3: 使用CCER价格预测表 (文件24 §4.1), 含2025-2040年碳价路径.
        year=1 对应项目第一年 (默认2026), 超出预测范围的年份用线性外推.
        """
        carbon_reduction_t = self_consumed_kwh / 1000 * CARBON_FACTOR_GRID
        # v6.3: 从预测表查碳价, 线性插值
        calendar_year = 2025 + year  # year=1 → 2026
        forecast_years = np.array(sorted(CCER_PRICE_FORECAST.keys()))
        forecast_prices = np.array([CCER_PRICE_FORECAST[y] for y in forecast_years])
        if calendar_year <= forecast_years[-1]:
            carbon_price = np.interp(calendar_year, forecast_years, forecast_prices)
        else:
            # 2040后线性外推 (保守估计年均+2%)
            carbon_price = CCER_PRICE_FORECAST[2040] * (1.02) ** (calendar_year - 2040)
        return carbon_reduction_t * carbon_price, carbon_reduction_t

    # ============================================================
    # 更换成本
    # ============================================================

    def replacement_schedule(self, annual_ess_cycles=0, avg_temp_c=25.0, avg_dod=0.50):
        """返回更换事件列表 [(year, cost, description), ...]

        v6.5: 储能电池同时检查温度加速日历寿命和DOD依赖循环寿命, 取先到者.
        calendar_life(T) = 1 / get_calendar_fade_rate(T) * 0.20  (20%总衰减阈值)
        cycle_life(DOD) = get_cycle_life_at_dod(DOD) / annual_cycles
        """
        events = []
        # 逆变器更换
        inv_life = LIFESPAN['pv_inverter']
        inv_cost = self.pv_cap * PV_COST['inverter'] * REPLACEMENT['inverter']
        for y in range(inv_life, PROJECT_LIFE, inv_life):
            events.append((y, inv_cost, f'光伏逆变器更换 (第{y}年)'))

        # v6.5: 储能电池 — 温度加速日历寿命 + DOD依赖循环寿命
        if self.ess_e > 0 and annual_ess_cycles > 0:
            # 日历寿命 (80% SOH, 即20%总衰减)
            cal_fade_rate = get_calendar_fade_rate(avg_temp_c + 7.0)  # 柜内温度 ≈ 环境+7°C
            cal_life_years = 0.20 / max(cal_fade_rate, 0.005)
            # 循环寿命
            dod_eff = max(0.10, min(0.90, avg_dod))
            cycle_life_total = get_cycle_life_at_dod(dod_eff)
            cycle_life_years = cycle_life_total / max(annual_ess_cycles, 1.0)
            batt_life = min(LIFESPAN['ess_battery'], cal_life_years, cycle_life_years)
        else:
            batt_life = LIFESPAN['ess_battery']
        batt_cost = self.ess_e * ESS_COST['battery_per_kwh'] * REPLACEMENT['ess_battery']
        y = batt_life
        while y <= PROJECT_LIFE:
            events.append((int(np.ceil(y)), batt_cost, f'储能电池更换 (第{int(np.ceil(y))}年)'))
            y += batt_life

        # PCS更换
        pcs_life = LIFESPAN['ess_pcs']
        pcs_cost = self.ess_p * ESS_COST['pcs_per_kw'] * REPLACEMENT['pcs']
        for y in range(pcs_life, PROJECT_LIFE, pcs_life):
            events.append((y, pcs_cost, f'PCS更换 (第{y}年)'))

        # 充电桩更换
        chg_life = LIFESPAN['charging_pile']
        chg_cost = (self.n_piles_120 * CHARGING_COST['pile_120kw'] +
                    self.n_piles_480 * CHARGING_COST['pile_480kw']) * REPLACEMENT['charging']
        for y in range(chg_life, PROJECT_LIFE, chg_life):
            events.append((y, chg_cost, f'充电桩更换 (第{y}年)'))

        events.sort(key=lambda x: x[0])
        return events

    # ============================================================
    # 光伏衰减
    # ============================================================

    @staticmethod
    def discount_rate_for_year(year, base_rate=None):
        """v6.3: 分段折现率 — 远期使用更低折现率

        1-5年: base, 6-10年: base-0.5%, 11-20年: base-1%
        反映长期无风险利率通常低于短期.
        """
        if base_rate is None:
            base_rate = DISCOUNT_RATE
        if year <= 5:
            return base_rate
        elif year <= 10:
            return max(base_rate - 0.005, 0.03)
        else:
            return max(base_rate - 0.01, 0.03)

    def _discount_factor(self, year, base_rate=None):
        """v6.3: 基于分段折现率的折现因子"""
        r = self.discount_rate_for_year(year, base_rate)
        return 1.0 / (1 + r) ** year

    def _pv_degradation_factor(self, year):
        """光伏容量保持率 (第year年, 首年衰减+逐年衰减)"""
        if year <= 0:
            return 1.0
        return (1 - PV_FIRST_YEAR_DEGRADATION) * \
               (1 - PV_ANNUAL_DEGRADATION) ** (year - 1)

    # ============================================================
    # 核心经济指标
    # ============================================================

    def npc(self, annual_grid_import_kwh, annual_tou_prices,
            annual_self_consumed_kwh, annual_grid_export_kwh=0):
        """净现值成本 NPC (元)

        NPC = 初始投资 - 补贴 + Σ(运维+网购电-上网收入-碳收入+更换)/(1+r)^n - 残值
        """
        r = self.sc_params['discount_rate']
        capex_net = self.capex_after_subsidy()
        npc_val = capex_net
        replacements = self.replacement_schedule()
        rep_idx = 0

        for y in range(1, PROJECT_LIFE + 1):
            discount = self._discount_factor(y, r)
            # 运维 (v6.3: 含年递增2%)
            opex = self.opex_annual(y, escalation_rate=0.02)
            # 购电成本 — 考虑电价上涨
            price_esc = self.sc_params['grid_price_escalation']
            grid_cost = float(np.sum(annual_grid_import_kwh * annual_tou_prices)) * \
                        (1 + price_esc) ** (y - 1)
            # 上网收入
            grid_rev = annual_grid_export_kwh * FEED_IN_PRICE * \
                       (1 + price_esc) ** (y - 1) * \
                       (1 + FEED_IN_PRICE_ESCALATION) ** (y - 1)
            # 碳收入
            pv_eff = self._pv_degradation_factor(y)
            self_use = annual_self_consumed_kwh * pv_eff
            carbon_rev, _ = self.carbon_revenue_annual(self_use, y)

            annual_cost = (opex + grid_cost - grid_rev) * discount
            annual_benefit = carbon_rev * discount

            npc_val += annual_cost - annual_benefit

            # 更换成本
            while rep_idx < len(replacements) and replacements[rep_idx][0] <= y:
                rep_cost = replacements[rep_idx][1]
                npc_val += rep_cost * self._discount_factor(replacements[rep_idx][0], r)
                rep_idx += 1

        # v6.3: 分项残值 (含充电桩+变压器)
        capex_pv = self.capex_pv()
        capex_ess = self.capex_ess()
        capex_charging = self.capex_charging()
        residual = ((capex_pv * RESIDUAL_RATE_PV + capex_ess * RESIDUAL_RATE_ESS +
                     capex_charging * RESIDUAL_RATE_CHARGING +
                     FIXED_COST_TOTAL * 0.10)  # 固定投资残值10%
                    * self._discount_factor(PROJECT_LIFE, r))
        npc_val -= residual

        return npc_val

    def npc_simple(self, annual_grid_cost, annual_self_consumed_kwh,
                   annual_grid_export_kwh=0, constant_opex=None):
        """简化NPC计算 — 使用年均网购电成本和等额运维"""
        r = self.sc_params['discount_rate']
        if constant_opex is None:
            constant_opex = self.opex_annual(1)

        capex_net = self.capex_after_subsidy()
        npc_val = capex_net

        for y in range(1, PROJECT_LIFE + 1):
            discount = self._discount_factor(y, r)
            price_esc = self.sc_params['grid_price_escalation']
            grid_cost = annual_grid_cost * (1 + price_esc) ** (y - 1)
            grid_rev = annual_grid_export_kwh * FEED_IN_PRICE * (1 + price_esc) ** (y - 1)
            carbon_rev, _ = self.carbon_revenue_annual(annual_self_consumed_kwh, y)
            opex_y = constant_opex * (1 + 0.02) ** (y - 1)  # v6.3: 运维递增
            npc_val += (opex_y + grid_cost - grid_rev - carbon_rev) * discount

        replacements = self.replacement_schedule()
        for year, cost, _ in replacements:
            npc_val += cost * self._discount_factor(year, r)

        # v6.3: 全分项残值
        residual = ((self.capex_pv() * RESIDUAL_RATE_PV +
                     self.capex_ess() * RESIDUAL_RATE_ESS +
                     self.capex_charging() * RESIDUAL_RATE_CHARGING +
                     FIXED_COST_TOTAL * 0.10)
                    * self._discount_factor(PROJECT_LIFE, r))
        npc_val -= residual

        return npc_val

    def lcoe(self, npc_val, total_energy_served_kwh):
        """平准化度电成本 LCOE (元/kWh)"""
        if total_energy_served_kwh <= 0:
            return float('inf')
        r = self.sc_params['discount_rate']
        discounted_energy = sum(total_energy_served_kwh / (1 + r) ** y
                                for y in range(1, PROJECT_LIFE + 1))
        return npc_val / discounted_energy if discounted_energy > 0 else float('inf')

    def payback_period(self, annual_net_saving, method='dynamic'):
        """投资回收期 (年)

        Parameters
        ----------
        annual_net_saving : float
            年均净节省 (元/年): 网购电节省 + 碳收入 + 上网收入 - 运维 - 网购电
        method : str
            'static' — 静态回收期
            'dynamic' — 动态回收期 (折现后现金流)
        """
        if annual_net_saving <= 0:
            return float('inf')
        if method == 'static':
            return self.capex_after_subsidy() / annual_net_saving

        r = self.sc_params['discount_rate']
        cumulative = -self.capex_after_subsidy()
        replacements = self.replacement_schedule()
        rep_idx = 0
        for y in range(1, PROJECT_LIFE + 1):
            discount = self._discount_factor(y, r)
            cash_flow = annual_net_saving * discount
            while rep_idx < len(replacements) and replacements[rep_idx][0] <= y:
                cash_flow -= replacements[rep_idx][1] * \
                             self._discount_factor(replacements[rep_idx][0], r)
                rep_idx += 1
            cumulative += cash_flow
            if cumulative >= 0:
                # 线性插值
                prev = cumulative - cash_flow
                return y - 1 + abs(prev) / (cash_flow / discount) if cash_flow != 0 else y
        return PROJECT_LIFE  # 未回收

    def irr(self, annual_cash_flows):
        """内部收益率 IRR (使用牛顿法近似)

        annual_cash_flows: list of (year, net_cash_flow)
        """
        flows = []
        replacements = {y: c for y, c, _ in self.replacement_schedule()}
        flows.append((-self.capex_after_subsidy(), 0))
        for y in range(1, PROJECT_LIFE + 1):
            cf = annual_cash_flows.get(y, 0)
            if y in replacements:
                cf -= replacements[y]
            flows.append((cf, y))
        flows.append((self.capex_pv() * RESIDUAL_RATE * 0.5, PROJECT_LIFE))

        def npv(r):
            return sum(cf / (1 + r) ** t for cf, t in flows)

        # 尝试多个折现率找符号变化
        for r_try in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            if npv(r_try) < 0:
                break
        else:
            return 0.50  # 如果50%仍为正, 返回上限

        lo, hi = 0.0, r_try
        for _ in range(50):
            mid = (lo + hi) / 2
            v = npv(mid)
            if abs(v) < 1e-4:
                return mid
            if v > 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def roi(self, annual_net_saving):
        """投资回报率 ROI"""
        if self.capex_after_subsidy() <= 0:
            return float('inf')
        return annual_net_saving / self.capex_after_subsidy()

    def bcr(self, total_benefit_npv, total_cost_npv):
        """效益成本比 BCR"""
        if total_cost_npv <= 0:
            return float('inf')
        return total_benefit_npv / total_cost_npv

    # ============================================================
    # 情景演化经济指标
    # ============================================================

    def update_config(self, pv_cap=None, ess_e=None, ess_p=None,
                       n_piles_120=None, n_piles_480=None, scenario=None):
        """快速更新配置 (供优化循环复用同一实例)"""
        if pv_cap is not None:
            self.pv_cap = pv_cap
        if ess_e is not None:
            self.ess_e = ess_e
        if ess_p is not None:
            self.ess_p = ess_p
        if n_piles_120 is not None:
            self.n_piles_120 = n_piles_120
        if n_piles_480 is not None:
            self.n_piles_480 = n_piles_480
        if scenario is not None:
            self.scenario = scenario
            self.sc_params = SCENARIOS.get(scenario, SCENARIOS['baseline'])
        return self

    def npc_from_aggregates(self, annual_grid_cost, annual_ess_cycles,
                             annual_carbon_revenue=0.0, subsidy=0.0,
                             capex_detail=None, scenario_params=None,
                             annual_dr_revenue=0.0,
                             avg_temp_c=25.0, avg_dod=0.50):
        """热路径NPC — 聚合年值输入, 供PSO/GA优化循环 (~2000次调用)

        与 _calculate_npc_detailed 对标, 但使用 EconomicModel 自身 CAPEX 方法。
        若传入 capex_detail 则优先使用 (兼容 capacity_optimizer 的 capex dict)。

        v6.4: S曲线负荷增长 + DR收入 (文件25 §2-3)
        v6.5: 温度/DOD依赖的电池退化模型 (文件26 §1)

        Parameters
        ----------
        annual_grid_cost : float
            年网购电成本 (元)
        annual_ess_cycles : float
            年储能等效循环次数
        annual_carbon_revenue : float
            年碳交易收入 (元)
        subsidy : float
            补贴金额 (元)
        capex_detail : dict or None
            若为None, 使用 self.capex_pv/ess/charging 计算
        scenario_params : dict or None
            演化情景参数, None则使用当前情景
        annual_dr_revenue : float
            v6.4: 年需求响应收入 (元)
        avg_temp_c : float
            v6.5: 年均环境温度 (℃), 用于Arrhenius日历老化
        avg_dod : float
            v6.5: 年均等效DOD, 用于Wöhler循环寿命
        """
        sp = scenario_params if scenario_params is not None else {
            'load_growth_rate': self.sc_params.get('load_growth_rate', 0.0),
            'grid_price_escalation': self.sc_params.get('grid_price_escalation', 0.0),
            'carbon_price_growth': self.sc_params.get('carbon_price_growth', 0.0),
            'discount_rate': self.sc_params.get('discount_rate', DISCOUNT_RATE),
        }

        if capex_detail is not None:
            capital = capex_detail['total'] - subsidy
            om_pv = capex_detail['pv_subtotal'] * OM_RATE['pv']
            om_ess = capex_detail['ess_subtotal'] * OM_RATE['ess']
            om_charging = capex_detail['charging_subtotal'] * OM_RATE['charging']
        else:
            capital = self.capex_after_subsidy()
            om_pv = self.capex_pv() * OM_RATE['pv']
            om_ess = self.capex_ess() * OM_RATE['ess']
            om_charging = self.capex_charging() * OM_RATE['charging']

        om_annual = om_pv + om_ess + om_charging

        # v6.5: 复合退化率 — 温度加速日历 + DOD依赖循环
        annual_degradation = get_battery_yearly_degradation(
            avg_temp_c + 7.0, annual_ess_cycles, avg_dod)  # 柜内温度≈环境+7°C
        ess_degrade_factor = 1.0

        dr = sp.get('discount_rate', DISCOUNT_RATE)
        lgr = sp.get('load_growth_rate', 0.0)
        gpe = sp.get('grid_price_escalation', 0.0)
        cpg = sp.get('carbon_price_growth', 0.0)
        # v6.4: S曲线负荷增长
        use_logistic = sp.get('load_growth_model', LOAD_GROWTH_MODEL) == 'logistic'

        npv_grid_cost = 0.0
        npv_carbon_rev = 0.0
        npv_dr_rev = 0.0

        for yr in range(1, PROJECT_LIFE + 1):
            discount = self._discount_factor(yr, dr)
            # v6.4: S曲线 或 指数增长
            if use_logistic:
                calendar_year = LOAD_GROWTH_BASE_YEAR + (yr - 1)
                load_factor = get_load_growth_factor(calendar_year, model='logistic')
            else:
                load_factor = (1.0 + lgr) ** (yr - 1)
            price_factor = (1.0 + gpe) ** (yr - 1)
            degrade_penalty = 1.0 + (1.0 - ess_degrade_factor) * 0.3

            yr_grid_cost = (annual_grid_cost * degrade_penalty
                           * load_factor * price_factor)
            npv_grid_cost += yr_grid_cost * discount

            carbon_factor = (1.0 + cpg) ** (yr - 1)
            yr_carbon_rev = annual_carbon_revenue * carbon_factor
            npv_carbon_rev += yr_carbon_rev * discount

            # v6.4: 需求响应收入 (文件25 §2)
            if annual_dr_revenue > 0:
                dr_escalation = 1.02  # DR补偿年涨2%
                yr_dr_rev = annual_dr_revenue * (dr_escalation ** (yr - 1))
                npv_dr_rev += yr_dr_rev * discount

            ess_degrade_factor = max(0.60, ess_degrade_factor - annual_degradation)

        # v6.3: 运维逐年递增
        npv_om = 0.0
        for y in range(1, PROJECT_LIFE + 1):
            opex_y = om_annual * (1 + 0.02) ** (y - 1)
            npv_om += opex_y * self._discount_factor(y, dr)

        # 更换成本 (v6.3: 电池双重寿命检查 + 分段折现)
        replacement_cost = 0
        if capex_detail is not None:
            if LIFESPAN['pv_inverter'] < PROJECT_LIFE:
                rep_year = LIFESPAN['pv_inverter']
                replacement_cost += (capex_detail['pv_inverter'] * REPLACEMENT['inverter'] *
                                     self._discount_factor(rep_year, dr))
            if self.ess_e > 0:
                # v6.5: 三重寿命 (日历温度加速 / DOD循环 / 制造商规格)
                if annual_ess_cycles > 0:
                    cal_fade_r = get_calendar_fade_rate(avg_temp_c + 7.0)
                    cal_life_yr = 0.20 / max(cal_fade_r, 0.005)
                    dod_eff = max(0.10, min(0.90, avg_dod))
                    cycle_life_total = get_cycle_life_at_dod(dod_eff)
                    cycle_life_yr = cycle_life_total / max(annual_ess_cycles, 1.0)
                    batt_life = min(LIFESPAN['ess_battery'], cal_life_yr, cycle_life_yr)
                else:
                    batt_life = LIFESPAN['ess_battery']
                y = batt_life
                while y <= PROJECT_LIFE:
                    replacement_cost += (capex_detail['ess_battery'] * REPLACEMENT['ess_battery'] *
                                         self._discount_factor(int(np.ceil(y)), dr))
                    y += batt_life
            if self.ess_p > 0 and LIFESPAN['ess_pcs'] < PROJECT_LIFE:
                rep_year = LIFESPAN['ess_pcs']
                replacement_cost += (capex_detail['ess_pcs'] * REPLACEMENT['pcs'] *
                                     self._discount_factor(rep_year, dr))
            if LIFESPAN['charging_pile'] < PROJECT_LIFE:
                rep_year = LIFESPAN['charging_pile']
                replacement_cost += (capex_detail['charging_subtotal'] * REPLACEMENT['charging'] *
                                     self._discount_factor(rep_year, dr))
            if LIFESPAN['ems_hw'] < PROJECT_LIFE:
                ems_hw_cost = capex_detail['fixed']['ems'] * 0.4
                for yr in [5, 10, 15]:
                    if yr <= PROJECT_LIFE:
                        replacement_cost += (ems_hw_cost * REPLACEMENT['ems_hw'] *
                                             self._discount_factor(yr, dr))
        else:
            for year, cost, _ in self.replacement_schedule(annual_ess_cycles, avg_temp_c, avg_dod):
                replacement_cost += cost * self._discount_factor(year, dr)

        # v6.3: 分项残值率 + 固定投资残值
        salvage_pv = (capex_detail['pv_subtotal'] if capex_detail else self.capex_pv()) * RESIDUAL_RATE_PV
        salvage_ess = (capex_detail['ess_subtotal'] if capex_detail else self.capex_ess()) * RESIDUAL_RATE_ESS
        salvage_charging = (capex_detail.get('charging_subtotal', 0) if capex_detail else self.capex_charging()) * RESIDUAL_RATE_CHARGING
        salvage_fixed = FIXED_COST_TOTAL * 0.10  # v6.3: 固定投资(含变压器)残值10%
        salvage = (salvage_pv + salvage_ess + salvage_charging + salvage_fixed) * self._discount_factor(PROJECT_LIFE, dr)

        return (capital + npv_om + npv_grid_cost + replacement_cost
                - salvage - npv_carbon_rev - npv_dr_rev)

    def npc_with_scenario(self, annual_grid_import_kwh, annual_tou_prices,
                          annual_self_consumed_kwh, annual_grid_export_kwh=0,
                          scenario_name='baseline'):
        """使用指定情景参数计算NPC"""
        old_params = self.sc_params
        self.sc_params = SCENARIOS.get(scenario_name, self.sc_params)
        result = self.npc(annual_grid_import_kwh, annual_tou_prices,
                          annual_self_consumed_kwh, annual_grid_export_kwh)
        self.sc_params = old_params
        return result

    # ============================================================
    # 打印与报告
    # ============================================================

    def print_cost_breakdown(self):
        """打印详细成本分解"""
        print("\n" + "=" * 60)
        print(f"全生命周期经济性评估 [{self.scenario}情景]")
        print("=" * 60)
        print(f"\n--- 初始投资 (CAPEX) ---")
        print(f"  光伏({self.pv_cap:.0f} kWp):  {self.capex_pv()/1e4:.1f} 万元")
        print(f"  储能({self.ess_e:.0f} kWh/{self.ess_p:.0f} kW): {self.capex_ess()/1e4:.1f} 万元")
        print(f"  充电桩({self.n_piles_120}×120kW + {self.n_piles_480}×480kW): {self.capex_charging()/1e4:.1f} 万元")
        print(f"  固定投资: {FIXED_COST_TOTAL/1e4:.1f} 万元")
        print(f"  总投资: {self.capex_total()/1e4:.1f} 万元")
        print(f"  补贴: {self.subsidy_total()/1e4:.1f} 万元")
        print(f"  补贴后投资: {self.capex_after_subsidy()/1e4:.1f} 万元")

        print(f"\n--- 年运维成本 (OPEX) ---")
        print(f"  光伏运维: {self.capex_pv()*OM_RATE['pv']/1e4:.1f} 万元/年")
        print(f"  储能运维: {self.capex_ess()*OM_RATE['ess']/1e4:.1f} 万元/年")
        print(f"  充电桩运维: {self.capex_charging()*OM_RATE['charging']/1e4:.1f} 万元/年")
        print(f"  年运维合计: {self.opex_annual()/1e4:.1f} 万元/年")

        print(f"\n--- 更换计划 ---")
        for year, cost, desc in self.replacement_schedule():
            print(f"  第{year}年: {cost/1e4:.1f} 万元 ({desc})")

        print(f"\n--- 情景参数 ---")
        print(f"  折现率: {self.sc_params['discount_rate']:.0%}")
        print(f"  负荷增长率: {self.sc_params['load_growth_rate']:.0%}")
        print(f"  EV渗透率增长: {self.sc_params['ev_penetration_growth']:.0%}")
        print(f"  电价上涨率: {self.sc_params['grid_price_escalation']:.0%}")
        print(f"  光伏成本降幅: {self.sc_params['pv_cost_reduction']:.0%}")
        print(f"  储能成本降幅: {self.sc_params['ess_cost_reduction']:.0%}")
        print(f"  碳价上涨率: {self.sc_params['carbon_price_growth']:.0%}")


# ============================================================
# 模块自测
# ============================================================

def self_test():
    """中型服务区基准方案经济性评估"""
    cfg = SERVICE_AREA_CONFIG['medium']

    em = EconomicModel(
        pv_capacity=1231,
        ess_capacity=2000,
        ess_power=1000,
        n_piles_120kw=cfg['n_piles_120kw'],
        n_piles_480kw=cfg['n_piles_480kw'],
        scenario='baseline',
    )

    em.print_cost_breakdown()

    # 示例: 使用模拟的年均数据计算关键指标
    annual_grid_cost = 30 * 1e4          # 30万元/年网购电
    annual_self_use = 110 * 1e4          # 110万kWh自用
    npc_val = em.npc_simple(annual_grid_cost, annual_self_use)
    annual_saving = annual_grid_cost * 0.6  # 节省60%购电费
    pbp = em.payback_period(annual_saving)
    lcoe = em.lcoe(npc_val, annual_self_use + annual_grid_cost / 0.7)

    print(f"\n--- 关键经济指标 ---")
    print(f"  NPC (20年): {npc_val/1e4:.1f} 万元")
    print(f"  LCOE: {lcoe:.4f} 元/kWh")
    print(f"  动态回收期: {pbp:.1f} 年")
    print(f"  ROI: {em.roi(annual_saving):.1%}")

    print(f"\n--- 三情景NPC对比 ---")
    for sc in ['conservative', 'baseline', 'aggressive']:
        npc_sc = em.npc_with_scenario(0, np.zeros(24), annual_self_use, 0, sc)
        print(f"  {sc}: NPC={npc_sc/1e4:.1f}万元")


if __name__ == '__main__':
    self_test()
