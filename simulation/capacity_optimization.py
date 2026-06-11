"""
系统容量优化 — 设备级粒度 (v2)
PSO求解最优光伏+储能配置, 成本计算细化到每个设备
"""
import numpy as np
from config import (
    SERVICE_AREA_CONFIG, PV_COEFF, WEATHER_COEFF, MONTHLY_WEATHER_DAYS,
    BUILDING_LOAD,
    WEATHER_CHARGING_COEFF, WEATHER_BUILDING_COEFF, HOLIDAY_TOU_FACTOR,
    PV_COST, PV_COST_PER_KWP, ESS_COST, CHARGING_COST, FIXED_COST, FIXED_COST_TOTAL,
    LIFESPAN, OM_RATE, REPLACEMENT,
    STATION_AUX_DAILY_KWH,
    TOU_PRICE_VALUES, TOU_PRICE, FEED_IN_PRICE,
    DISCOUNT_RATE, PROJECT_LIFE,
    RESIDUAL_RATE, SOC_MIN, SOC_MAX, SELF_SUFFICIENCY_MIN,
    PV_AREA_RATIO, CARBON_FACTOR_GRID, get_season,
    PV_FIRST_YEAR_DEGRADATION, PV_ANNUAL_DEGRADATION,
    ESS_CYCLE_LIFE, ESS_CAPACITY_FADE_PER_CYCLE, ESS_CAPACITY_FADE_CALENDAR,
    TRANSFORMER_CAPACITY_KVA, TRANSFORMER_PF_MIN,
    get_tou_price, get_tou_price_array,
    RESPONSE_TIME,
)

# 日类型年权重 (来源于文件12)
DAY_TYPE_ANNUAL = {
    'workday': 250, 'weekend': 104, 'holiday': 8, 'spring_festival': 3,
}
ANNUAL_DAYS_TOTAL = sum(DAY_TYPE_ANNUAL.values())  # 365

# 建筑负荷日类型修正系数 (来源于文件12 Section 5.2)
BUILDING_DAY_TYPE_COEFF = {
    'workday': 1.00, 'weekend': 1.15, 'holiday': 1.30, 'spring_festival': 1.50,
}


class MicrogridOptimizer:
    """设备级光储充微网容量优化器

    Parameters
    ----------
    size : str
        服务区规模: 'small' / 'medium' / 'large'
    mc_scenarios : dict
        蒙特卡洛充电负荷仿真结果
    seed : int or None
        随机种子
    calendar_ctx : CalendarContext or None
        统一日历上下文, 提供日类型和天气序列.
        若为None则使用旧版随机分配模式.
    """

    def __init__(self, size='medium', mc_scenarios=None, seed=None, calendar_ctx=None):
        self.size = size
        self.area_config = SERVICE_AREA_CONFIG[size]
        self.rng = np.random.RandomState(seed)
        self.calendar_ctx = calendar_ctx

        if mc_scenarios is None:
            raise ValueError("需要传入MC仿真结果 mc_scenarios")

        self.charging_load = {}
        for dt in ['workday', 'weekend', 'holiday', 'spring_festival']:
            self.charging_load[dt] = mc_scenarios[dt]['p50']

        self.building_load = {}
        self._scale_building_load()
        self._build_typical_days()

        # 固定充电桩数量
        self.n_piles_120 = self.area_config['n_piles_120kw']
        self.n_piles_480 = self.area_config['n_piles_480kw']

    def _build_typical_days(self):
        self.typical_days = []
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        annual_days = sum(days_in_month)
        for month in range(1, 13):
            season = get_season(month)
            wdays = MONTHLY_WEATHER_DAYS[month]
            for wtype, count in wdays.items():
                if count > 0:
                    self.typical_days.append({
                        'month': month, 'season': season,
                        'weather': wtype, 'days': count,
                        'weight': count / annual_days,
                    })

    def _build_8760h_sequence(self):
        """构建全年8760h逐时PV系数和负荷序列 (用于终验)

        使用CalendarContext提供的真实日历日类型和Markov天气序列.
        若calendar_ctx为None, 回退到旧版随机分配模式.
        """
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # --- 天气序列 ---
        if self.calendar_ctx is not None:
            day_weather = self.calendar_ctx.weather_seq
        else:
            # 旧版: 独立随机shuffle
            day_weather = []
            for month in range(1, 13):
                wdays = MONTHLY_WEATHER_DAYS[month]
                types = []
                for w, c in wdays.items():
                    types.extend([w] * c)
                remaining = days_in_month[month-1] - len(types)
                if remaining > 0:
                    total = sum(wdays.values())
                    fill = self.rng.choice(list(wdays.keys()), remaining,
                                           p=[wdays[t]/total for t in wdays])
                    types.extend(fill)
                self.rng.shuffle(types)
                day_weather.extend(types)

        # --- 季节序列 ---
        day_season = []
        for month in range(1, 13):
            season = get_season(month)
            day_season.extend([season] * days_in_month[month-1])

        # --- 日类型序列 ---
        if self.calendar_ctx is not None:
            day_types_final = self.calendar_ctx.day_types
        else:
            # 旧版: 按DAY_TYPE_ANNUAL权重随机分配 (无日历结构)
            day_types_final = []
            counts = dict(DAY_TYPE_ANNUAL)
            for _ in range(365):
                available = [k for k, v in counts.items() if v > 0]
                dt = self.rng.choice(available)
                counts[dt] -= 1
                day_types_final.append(dt)

        pv_coeff_seq = np.zeros(8760)
        load_seq = np.zeros(8760)
        tou_seq = np.zeros(8760)
        seasons_seq = [''] * 8760

        for d in range(365):
            season = day_season[d]
            weather = day_weather[d]
            dt = day_types_final[d]

            pv_coeff = np.array(PV_COEFF[season]) * WEATHER_COEFF.get(weather, 1.0)
            pv_coeff_seq[d*24:(d+1)*24] = pv_coeff

            # 负荷含天气修正
            load = self.get_total_load(season, dt, weather)
            load_seq[d*24:(d+1)*24] = load

            # TOU含节假日调整
            tou_arr = get_tou_price_array(season)
            tou_factor = HOLIDAY_TOU_FACTOR.get(dt, 1.0)
            tou_seq[d*24:(d+1)*24] = tou_arr * tou_factor

            for h in range(24):
                seasons_seq[d*24+h] = season

        return pv_coeff_seq, load_seq, tou_seq, seasons_seq

    def verify_8760h(self, pv_cap, ess_cap, ess_pow):
        """8760h全时序验证 — 对最优配置运行全年仿真"""
        self.pv_capacity = pv_cap
        self.ess_capacity = ess_cap
        self.ess_power = ess_pow

        pv_coeff_seq, load_seq, tou_seq, seasons_seq = self._build_8760h_sequence()

        total_grid_import = 0.0
        total_grid_export = 0.0
        total_grid_cost = 0.0
        total_pv_gen = 0.0
        total_load = 0.0
        total_self_used = 0.0
        total_loss_of_load = 0.0
        total_ess_cycles = 0.0

        for d in range(365):
            season = seasons_seq[d*24]
            pv_profile = pv_coeff_seq[d*24:(d+1)*24] * pv_cap
            load_profile = load_seq[d*24:(d+1)*24]

            tou_hourly = tou_seq[d*24:(d+1)*24]
            result = self.simulate_daily_operation(pv_profile, load_profile, season,
                                                   tou_prices=tou_hourly)
            total_grid_import += result['grid_import'].sum()
            total_grid_export += result['grid_export'].sum()
            total_pv_gen += pv_profile.sum()
            total_load += load_profile.sum()
            total_self_used += pv_profile.sum() - result['grid_export'].sum()
            total_loss_of_load += result['loss_of_load'].sum()

            total_grid_cost += np.sum(result['grid_import'] * tou_hourly)
            total_grid_cost -= result['grid_export'].sum() * FEED_IN_PRICE

            if ess_cap > 0:
                total_ess_cycles += result['ess_discharge'].sum() / ess_cap

        # 站用电
        station_aux = STATION_AUX_DAILY_KWH * 365
        if ess_cap > 0:
            n_cab = max(1, int(np.ceil(ess_cap / 215)))
            station_aux += n_cab * 3.0 * 24 * 365 * 0.3
        flat_price = np.mean([v['flat'] for v in TOU_PRICE_VALUES.values()])
        total_grid_import += station_aux
        total_grid_cost += station_aux * flat_price

        ssr = total_self_used / max(total_load, 1.0)
        lolp = total_loss_of_load / max(total_load, 1e-6)

        return {
            'grid_import_kwh': total_grid_import,
            'grid_export_kwh': total_grid_export,
            'grid_cost': total_grid_cost,
            'pv_gen_kwh': total_pv_gen,
            'load_kwh': total_load,
            'self_sufficiency': ssr,
            'loss_of_load_pct': lolp,
            'ess_cycles': total_ess_cycles,
        }

    def _scale_building_load(self):
        base_peak = 147.0
        target_peak = self.area_config['peak_building_kw']
        scale = target_peak / base_peak
        for season in BUILDING_LOAD:
            self.building_load[season] = np.array(BUILDING_LOAD[season]) * scale

    def get_total_load(self, season, day_type='workday', weather='clear'):
        """计算总负荷, 考虑日类型和天气影响

        Parameters
        ----------
        season : str
            季节
        day_type : str
            日类型: workday/weekend/holiday/spring_festival
        weather : str
            天气类型: clear/partly_cloudy/cloudy/overcast/rain
        """
        charging = self.charging_load.get(day_type, self.charging_load['workday'])
        building = self.building_load.get(season, self.building_load['spring'])
        # 建筑负荷日类型修正
        bldg_coeff = BUILDING_DAY_TYPE_COEFF.get(day_type, 1.0)
        # 天气修正: 恶劣天气 → 充电需求下降, 建筑负荷上升
        weather_chg = WEATHER_CHARGING_COEFF.get(weather, 1.0)
        weather_bldg = WEATHER_BUILDING_COEFF.get(weather, 1.0)
        return charging * weather_chg + building * bldg_coeff * weather_bldg

    def simulate_daily_operation(self, pv_profile, load_profile, season='spring',
                                  tou_prices=None):
        """TOU感知最优调度 — 24h迭代套利算法

        核心逻辑: 在已知24h PV+负荷曲线和分时电价的前提下,
        通过ESS在低价时段充电、高价时段放电实现最优套利.

        Parameters
        ----------
        tou_prices : np.ndarray or None
            24h电价数组, 若为None则按季节自动获取.
            传入时可包含节假日调整因子.
        """
        T = 24
        prices = tou_prices if tou_prices is not None else get_tou_price_array(season)
        net_load = load_profile - pv_profile  # + = deficit

        # 初始化: ESS不参与, 纯直连
        grid_import = np.maximum(0, net_load)
        grid_export = np.maximum(0, -net_load)
        ess_ch = np.zeros(T)
        ess_disch = np.zeros(T)

        has_ess = self.ess_capacity > 1e-3
        if not has_ess:
            return {
                'grid_import': grid_import, 'grid_export': grid_export,
                'soc_curve': np.full(T, 0.5),
                'ess_charge': ess_ch, 'ess_discharge': ess_disch,
                'loss_of_load': np.zeros(T),
            }

        eta_rt = 0.92  # RTE (AC侧)
        eta_one_way = np.sqrt(eta_rt)

        # 迭代寻找最优充放电对
        for _ in range(80):
            best_benefit = 0
            best_trade = None

            for t_ch in range(T):
                ch_headroom = self.ess_power - ess_ch[t_ch]
                if ch_headroom < 0.5:
                    continue

                # 充电成本: 优先用弃光(机会成本=上网电价), 不足时网购(成本=TOU)
                from_export = min(ch_headroom, grid_export[t_ch])
                from_grid = ch_headroom - from_export
                ch_cost = FEED_IN_PRICE if from_export > 0 else prices[t_ch]

                max_energy = from_export + from_grid
                if max_energy < 0.5:
                    continue

                for t_dis in range(T):
                    if grid_import[t_dis] < 0.5:
                        continue
                    dis_headroom = min(self.ess_power - ess_disch[t_dis],
                                       grid_import[t_dis])
                    if dis_headroom < 0.5:
                        continue

                    energy = min(max_energy, dis_headroom / eta_rt)
                    if energy < 0.5:
                        continue

                    benefit = prices[t_dis] * energy * eta_rt
                    cost = ch_cost * energy
                    if benefit - cost > best_benefit:
                        best_benefit = benefit - cost
                        best_trade = (t_ch, t_dis, energy)

            if best_trade is None or best_benefit <= 0:
                break

            t_ch, t_dis, energy = best_trade
            ess_ch[t_ch] += energy
            dis_energy = energy * eta_rt
            ess_disch[t_dis] += dis_energy

            # 更新电网交互: 优先减少弃光, 再增加网购
            if grid_export[t_ch] >= energy:
                grid_export[t_ch] -= energy
            else:
                taken_from_export = grid_export[t_ch]
                grid_export[t_ch] = 0
                grid_import[t_ch] += energy - taken_from_export

            grid_import[t_dis] = max(0, grid_import[t_dis] - dis_energy)

        # 重建SOC曲线
        soc = 0.5
        soc_curve = np.zeros(T)
        for t in range(T):
            soc += (ess_ch[t] * eta_one_way - ess_disch[t] / eta_one_way) / self.ess_capacity
            soc = np.clip(soc, SOC_MIN, SOC_MAX)
            soc_curve[t] = soc

        # 变压器容量约束: 网购功率不超过变压器极限
        trans_limit_kw = TRANSFORMER_CAPACITY_KVA * TRANSFORMER_PF_MIN
        loss_of_load = np.zeros(T)
        for t in range(T):
            if grid_import[t] > trans_limit_kw:
                excess = grid_import[t] - trans_limit_kw
                grid_import[t] = trans_limit_kw
                loss_of_load[t] = excess

        return {
            'grid_import': grid_import, 'grid_export': grid_export,
            'soc_curve': soc_curve,
            'ess_charge': ess_ch, 'ess_discharge': ess_disch,
            'loss_of_load': loss_of_load,
        }

    def evaluate_config(self, pv_cap, ess_cap, ess_pow):
        """评估配置 — 季节性TOU + 最优调度 + 年化指标"""
        self.pv_capacity = pv_cap
        self.ess_capacity = ess_cap
        self.ess_power = ess_pow

        annual_grid_import = 0.0
        annual_grid_export = 0.0
        annual_grid_cost = 0.0       # 网购电成本 (分时计价)
        annual_grid_export_rev = 0.0 # 上网收入
        annual_pv_gen = 0.0
        annual_load = 0.0
        annual_self_used = 0.0
        annual_loss_of_load = 0.0
        annual_ess_cycles = 0.0      # 储能等效循环次数

        day_type_list = ['workday', 'weekend', 'holiday', 'spring_festival']
        day_type_fractions = {dt: DAY_TYPE_ANNUAL[dt] / ANNUAL_DAYS_TOTAL for dt in day_type_list}

        for td in self.typical_days:
            season = td['season']
            weather = td['weather']
            weight_days = td['days']

            pv_coeff = np.array(PV_COEFF[season])
            weather_factor = WEATHER_COEFF.get(weather, 1.0)
            tou_hourly = get_tou_price_array(season)

            for dt in day_type_list:
                dt_fraction = day_type_fractions[dt]
                dt_weight = weight_days * dt_fraction

                pv_profile = pv_coeff * pv_cap * weather_factor
                # 负荷含天气修正 (恶劣天气 → 充电↓, 建筑↑)
                load_profile = self.get_total_load(season, dt, weather)

                # 节假日TOU调整 (春节/国庆等节假日电价低于工作日)
                tou_factor = HOLIDAY_TOU_FACTOR.get(dt, 1.0)
                tou_hourly_adj = tou_hourly * tou_factor

                result = self.simulate_daily_operation(pv_profile, load_profile, season,
                                                       tou_prices=tou_hourly_adj)
                daily_pv_gen = pv_profile.sum()
                daily_export = result['grid_export'].sum()
                daily_load = load_profile.sum()
                daily_self_used = daily_pv_gen - daily_export

                annual_grid_import += result['grid_import'].sum() * dt_weight
                annual_grid_export += daily_export * dt_weight
                annual_pv_gen += daily_pv_gen * dt_weight
                annual_load += daily_load * dt_weight
                annual_self_used += daily_self_used * dt_weight
                annual_loss_of_load += result['loss_of_load'].sum() * dt_weight

                # 电网购电成本 = Σ(逐时网购电量 × 调整后分时电价)
                annual_grid_cost += np.sum(result['grid_import'] * tou_hourly_adj) * dt_weight
                annual_grid_export_rev += daily_export * FEED_IN_PRICE * dt_weight

                # 储能日循环次数 = 日放电量 / 容量
                if ess_cap > 0:
                    daily_cycles = result['ess_discharge'].sum() / ess_cap
                    annual_ess_cycles += daily_cycles * dt_weight

        # 光伏衰减 (年化平均)
        avg_degradation = PV_FIRST_YEAR_DEGRADATION + PV_ANNUAL_DEGRADATION * (PROJECT_LIFE - 1) / 2
        annual_pv_gen *= (1 - avg_degradation)
        annual_self_used *= (1 - avg_degradation)
        annual_grid_export *= (1 - avg_degradation)
        annual_grid_export_rev *= (1 - avg_degradation)

        self_sufficiency = annual_self_used / max(annual_load, 1.0)
        loss_of_load_pct = annual_loss_of_load / max(annual_load, 1e-6)

        # 站用电 (来源于架构文档)
        station_aux = STATION_AUX_DAILY_KWH * 365
        if ess_cap > 0:
            n_cabinets = max(1, int(np.ceil(ess_cap / 215)))
            station_aux += n_cabinets * 3.0 * 24 * 365 * 0.3
        annual_grid_import += station_aux
        # 站用电成本按平电价计算
        flat_price = np.mean([v['flat'] for v in TOU_PRICE_VALUES.values()])
        annual_grid_cost += station_aux * flat_price

        annual_grid_cost -= annual_grid_export_rev

        # 经济计算
        capex_detail = self._capex_detail(pv_cap, ess_cap, ess_pow)
        total_capex = capex_detail['total']
        npc = self._calculate_npc_detailed(capex_detail, pv_cap, ess_cap, ess_pow,
                                            annual_grid_cost, annual_ess_cycles)

        # 年收益 = 无光储时的网购电成本 - 有光储时的网购电成本
        # 无光储时: 全年负荷按平电价全购电
        annual_cost_without = annual_load * flat_price
        annual_saving = annual_cost_without - annual_grid_cost
        payback = self._dynamic_payback(total_capex, annual_saving)
        carbon_reduction = annual_self_used / 1000 * CARBON_FACTOR_GRID

        return {
            'pv_capacity': pv_cap,
            'ess_capacity': ess_cap,
            'ess_power': ess_pow,
            'npc': npc,
            'self_sufficiency': self_sufficiency,
            'loss_of_load_pct': loss_of_load_pct,
            'annual_grid_import_kwh': annual_grid_import,
            'annual_grid_export_kwh': annual_grid_export,
            'annual_pv_gen_kwh': annual_pv_gen,
            'annual_load_kwh': annual_load,
            'annual_self_used_kwh': annual_self_used,
            'annual_ess_cycles': annual_ess_cycles,
            'annual_grid_cost': annual_grid_cost,
            'carbon_reduction_t': carbon_reduction,
            'capital_cost': total_capex,
            'payback_years': payback,
            'capex_detail': capex_detail,
        }

    def _dynamic_payback(self, capex, annual_saving):
        """动态投资回收期 (含折现率)"""
        if annual_saving <= 0 or capex <= 0:
            return float('inf')
        cum = 0.0
        for yr in range(1, PROJECT_LIFE + 1):
            cum += annual_saving / (1 + DISCOUNT_RATE) ** yr
            if cum >= capex:
                # 线性插值
                if yr > 1:
                    prev = cum - annual_saving / (1 + DISCOUNT_RATE) ** yr
                    frac = (capex - prev) / (cum - prev)
                    return yr - 1 + frac
                return float(yr)
        return float('inf')

    def _capex_detail(self, pv_cap, ess_cap, ess_pow):
        """设备级CAPEX明细"""
        detail = {}

        # 光伏系统
        detail['pv_module'] = pv_cap * PV_COST['module']
        detail['pv_inverter'] = pv_cap * PV_COST['inverter']
        detail['pv_combiner'] = pv_cap * PV_COST['combiner_box']
        detail['pv_structure'] = pv_cap * PV_COST['structure_carport']
        detail['pv_dc_cable'] = pv_cap * PV_COST['dc_cable']
        detail['pv_subtotal'] = pv_cap * PV_COST_PER_KWP

        # 储能系统
        detail['ess_battery'] = ess_cap * ESS_COST['battery_per_kwh']
        detail['ess_pcs'] = ess_pow * ESS_COST['pcs_per_kw']
        detail['ess_fire'] = ess_cap * ESS_COST['fire_per_kwh']
        detail['ess_subtotal'] = (detail['ess_battery'] + detail['ess_pcs'] + detail['ess_fire'])

        # 充电桩
        detail['charging_120kw'] = self.n_piles_120 * CHARGING_COST['pile_120kw']
        detail['charging_480kw'] = self.n_piles_480 * CHARGING_COST['pile_480kw']
        detail['charging_subtotal'] = detail['charging_120kw'] + detail['charging_480kw']

        # 固定投资
        detail['fixed'] = FIXED_COST.copy()
        detail['fixed_subtotal'] = FIXED_COST_TOTAL

        # 汇总
        detail['total'] = (detail['pv_subtotal'] + detail['ess_subtotal'] +
                           detail['charging_subtotal'] + detail['fixed_subtotal'])
        return detail

    def _calculate_npc_detailed(self, capex, pv_cap, ess_cap, ess_pow,
                                 annual_grid_cost, annual_ess_cycles):
        """设备级NPC — 分时计价网购成本 + 电池逐年衰减 + 设备更换"""
        capital = capex['total']

        om_pv = capex['pv_subtotal'] * OM_RATE['pv']
        om_ess = capex['ess_subtotal'] * OM_RATE['ess']
        om_charging = capex['charging_subtotal'] * OM_RATE['charging']
        om_annual = om_pv + om_ess + om_charging

        # 电池容量逐年衰减 → 储能调节能力下降, 网购电成本逐年增加
        annual_degradation = (ESS_CAPACITY_FADE_CALENDAR +
                              ESS_CAPACITY_FADE_PER_CYCLE * annual_ess_cycles)
        ess_degrade_factor = 1.0

        npv_grid_cost = 0.0
        for yr in range(1, PROJECT_LIFE + 1):
            degrade_penalty = 1.0 + (1.0 - ess_degrade_factor) * 0.3
            yr_grid_cost = annual_grid_cost * degrade_penalty
            npv_grid_cost += yr_grid_cost / (1 + DISCOUNT_RATE) ** yr
            ess_degrade_factor = max(0.60, ess_degrade_factor - annual_degradation)

        npv_om = om_annual * sum(1.0 / (1 + DISCOUNT_RATE) ** y
                                 for y in range(1, PROJECT_LIFE + 1))

        # --- 设备更换 (按各自寿命) ---
        replacement_cost = 0

        if LIFESPAN['pv_inverter'] < PROJECT_LIFE:
            rep_year = LIFESPAN['pv_inverter']
            replacement_cost += (capex['pv_inverter'] * REPLACEMENT['inverter'] /
                                 (1 + DISCOUNT_RATE) ** rep_year)

        if ess_cap > 0 and LIFESPAN['ess_battery'] < PROJECT_LIFE:
            rep_year = LIFESPAN['ess_battery']
            replacement_cost += (capex['ess_battery'] * REPLACEMENT['ess_battery'] /
                                 (1 + DISCOUNT_RATE) ** rep_year)

        if ess_pow > 0 and LIFESPAN['ess_pcs'] < PROJECT_LIFE:
            rep_year = LIFESPAN['ess_pcs']
            replacement_cost += (capex['ess_pcs'] * REPLACEMENT['pcs'] /
                                 (1 + DISCOUNT_RATE) ** rep_year)

        if LIFESPAN['charging_pile'] < PROJECT_LIFE:
            rep_year = LIFESPAN['charging_pile']
            replacement_cost += (capex['charging_subtotal'] * REPLACEMENT['charging'] /
                                 (1 + DISCOUNT_RATE) ** rep_year)

        if LIFESPAN['ems_hw'] < PROJECT_LIFE:
            ems_hw_cost = capex['fixed']['ems'] * 0.4
            for yr in [5, 10, 15]:
                if yr <= PROJECT_LIFE:
                    replacement_cost += (ems_hw_cost * REPLACEMENT['ems_hw'] /
                                         (1 + DISCOUNT_RATE) ** yr)

        salvage = capital * RESIDUAL_RATE / (1 + DISCOUNT_RATE) ** PROJECT_LIFE

        return capital + npv_om + npv_grid_cost + replacement_cost - salvage

    def optimize_pso(self, pop_size=50, max_iter=30, verbose=True):
        """PSO优化"""
        bounds = np.array([
            [50, self.area_config['pv_area_m2'] / PV_AREA_RATIO],
            [0, 3000],
            [0, 1000],
        ])

        dim = 3
        pos = np.zeros((pop_size, dim))
        vel = np.zeros((pop_size, dim))
        pbest_pos = np.zeros((pop_size, dim))
        pbest_fit = np.full(pop_size, np.inf)
        gbest_pos = np.zeros(dim)
        gbest_fit = np.inf
        best_result = None

        for i in range(pop_size):
            pos[i, 0] = self.rng.uniform(bounds[0, 0], bounds[0, 1])
            pos[i, 1] = self.rng.uniform(bounds[1, 0], bounds[1, 1])
            pos[i, 2] = self.rng.uniform(bounds[2, 0], min(pos[i, 1] * 0.5, bounds[2, 1]))

        def fitness(x):
            pv, ess_e, ess_p = x[0], x[1], x[2]
            if ess_e < 10:
                ess_e = 0; ess_p = 0
            ess_p = min(ess_p, ess_e * 0.5)
            result = self.evaluate_config(pv, ess_e, ess_p)
            npc_val = result['npc']
            ssr = result['self_sufficiency']
            penalty = 0
            # 自洽率软约束: 达不到目标时按NPC比例惩罚
            if ssr < SELF_SUFFICIENCY_MIN:
                shortfall = SELF_SUFFICIENCY_MIN - ssr
                penalty += shortfall * npc_val * 2.0
            # 面积约束: 超出可用面积时惩罚
            area_needed = pv * PV_AREA_RATIO
            area_avail = self.area_config['pv_area_m2']
            if area_needed > area_avail:
                penalty += (area_needed - area_avail) * PV_COST_PER_KWP * 0.5
            return npc_val + penalty, result

        w = 0.7
        c1, c2 = 1.5, 2.0

        for it in range(max_iter):
            for i in range(pop_size):
                pos[i] = np.clip(pos[i], bounds[:, 0], bounds[:, 1])
                pos[i, 2] = min(pos[i, 2], pos[i, 1] * 0.5)
                fit, result = fitness(pos[i])
                if fit < pbest_fit[i]:
                    pbest_fit[i] = fit; pbest_pos[i] = pos[i].copy()
                if fit < gbest_fit:
                    gbest_fit = fit; gbest_pos = pos[i].copy()
                    best_result = result
            for i in range(pop_size):
                r1, r2 = self.rng.random(dim), self.rng.random(dim)
                vel[i] = (w * vel[i] + c1 * r1 * (pbest_pos[i] - pos[i]) +
                          c2 * r2 * (gbest_pos - pos[i]))
                pos[i] += vel[i]
            w = 0.7 - 0.4 * it / max_iter

            if verbose and (it + 1) % 5 == 0 and best_result:
                print(f"  PSO iter {it+1}/{max_iter}: NPC={gbest_fit/1e4:.1f}万元, "
                      f"PV={gbest_pos[0]:.0f}kWp, ESS={gbest_pos[1]:.0f}kWh, "
                      f"SSR={best_result['self_sufficiency']:.1%}")

        return best_result, gbest_pos

    def print_capex_table(self, result):
        """打印设备级CAPEX明细表"""
        capex = result['capex_detail']
        print("\n" + "=" * 55)
        print("设备级初始投资明细 (CAPEX)")
        print("=" * 55)
        items = [
            ("光伏组件", capex['pv_module']),
            ("光伏逆变器", capex['pv_inverter']),
            ("汇流箱", capex['pv_combiner']),
            ("车棚钢结构", capex['pv_structure']),
            ("PV直流电缆", capex['pv_dc_cable']),
            ("-- 光伏系统小计", capex['pv_subtotal']),
            ("储能电池(含BMS+柜体+温控)", capex['ess_battery']),
            ("PCS变流器", capex['ess_pcs']),
            ("消防系统", capex['ess_fire']),
            ("-- 储能系统小计", capex['ess_subtotal']),
            ("120kW快充桩", capex['charging_120kw']),
            ("480kW超充堆", capex['charging_480kw']),
            ("-- 充电系统小计", capex['charging_subtotal']),
            ("箱式变电站", capex['fixed']['transformer']),
            ("交直流配电柜", capex['fixed']['switchgear_ac'] + capex['fixed']['switchgear_dc']),
            ("交流电缆桥架", capex['fixed']['cables_ac']),
            ("EMS能量管理", capex['fixed']['ems']),
            ("监控计量", capex['fixed']['monitoring']),
            ("站用电+UPS", capex['fixed']['ups_station']),
            ("土建工程", capex['fixed']['civil']),
            ("设计并网管理", capex['fixed']['soft']),
            ("-- 固定投资小计", capex['fixed_subtotal']),
        ]
        for name, cost in items:
            bar = "--" if name.startswith("--") else "  "
            print(f"{bar}{name:<30s}: {cost/1e4:>8.1f} 万元")
        print(f"{'='*45}")
        print(f"  {'总计':<30s}: {capex['total']/1e4:>8.1f} 万元")


if __name__ == '__main__':
    from mc_charging_load import MonteCarloChargingSimulator
    sim = MonteCarloChargingSimulator(service_area_size='medium', seed=42)
    mc = sim.simulate_all_scenarios(n_runs=2000)
    opt = MicrogridOptimizer(size='medium', mc_scenarios=mc, seed=42)
    result, best_x = opt.optimize_pso(pop_size=30, max_iter=20)
    opt.print_capex_table(result)

    print(f"\n关键指标:")
    print(f"  NPC (20年):    {result['npc']/1e4:.1f} 万元")
    print(f"  自洽率:        {result['self_sufficiency']:.1%}")
    print(f"  年碳减排:      {result['carbon_reduction_t']:.1f} tCO2")
    print(f"  投资回收期:    {result['payback_years']:.1f} 年")
    print(f"\n设备响应时间链:")
    for device, rt in RESPONSE_TIME.items():
        print(f"  {device}: {rt} ms")
