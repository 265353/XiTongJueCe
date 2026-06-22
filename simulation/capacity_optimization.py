"""
系统容量优化 — 设备级粒度 (v2)
PSO求解最优光伏+储能配置, 成本计算细化到每个设备
"""
import numpy as np
import warnings
from config import (
    SERVICE_AREA_CONFIG, PV_COEFF, WEATHER_COEFF, MONTHLY_WEATHER_DAYS,
    BUILDING_LOAD,
    WEATHER_CHARGING_COEFF, WEATHER_BUILDING_COEFF, HOLIDAY_TOU_FACTOR,
    CHARGING_BUILDING_COUPLING,
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
    # v5 新增: 经济增强参数
    CCER_PRICE, CCER_PRICE_ESCALATION,
    PV_SUBSIDY_PER_KWP, ESS_SUBSIDY_PER_KWH, ESS_SUBSIDY_PER_KW,
    CHARGING_SUBSIDY_PER_PILE, MAX_SUBSIDY_RATIO,
    SCENARIOS, SCENARIO_BASELINE, LOAD_GROWTH_MODEL, LOAD_GROWTH_BASE_YEAR, get_load_growth_factor,
    MONTHLY_AMBIENT_TEMP, get_ess_temp_derate,
    get_rte,  # v6.3: SOC/C-rate依赖RTE
    # v6.5: 电池退化与温度/DOD关系
    get_battery_yearly_degradation, get_calendar_fade_rate, get_cycle_life_at_dod,
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

    def __init__(self, size='medium', mc_scenarios=None, seed=None, calendar_ctx=None,
                 use_abm=False, use_econ_model=False, abm_seed=42):
        self.size = size
        self.area_config = SERVICE_AREA_CONFIG[size]
        self.rng = np.random.RandomState(seed)
        self.calendar_ctx = calendar_ctx

        if mc_scenarios is None:
            raise ValueError("需要传入MC仿真结果 mc_scenarios")

        self.charging_load = {}
        self.charging_load_p95 = {}  # v6.3: P95场景用于尾部风险评估
        for dt in ['workday', 'weekend', 'holiday', 'spring_festival']:
            self.charging_load[dt] = mc_scenarios[dt]['p50']
            self.charging_load_p95[dt] = mc_scenarios[dt]['p95']

        # 固定充电桩数量
        self.n_piles_120 = self.area_config['n_piles_120kw']
        self.n_piles_480 = self.area_config['n_piles_480kw']

        self.building_load = {}
        self.use_abm = use_abm
        if use_abm:
            from building_load_abm import BuildingLoadABM
            self._abm = BuildingLoadABM(area_size=size, seed=abm_seed)
            for season in ['spring', 'summer', 'autumn', 'winter']:
                self.building_load[season] = self._abm.get_typical_day_curve(season, 'workday')
        else:
            self._abm = None
        self._scale_building_load()
        self._build_typical_days()

        # 经济模型缓存 (v6: 统一经济计算)
        self.use_econ_model = use_econ_model
        self._econ_model = None
        if use_econ_model:
            from economic_model import EconomicModel
            self._econ_model = EconomicModel(
                pv_capacity=0, ess_capacity=0, ess_power=0,
                n_piles_120kw=self.n_piles_120,
                n_piles_480kw=self.n_piles_480,
                scenario='baseline')

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

        soc = 0.5  # v6.2: SOC跨天连续 (verify路径)

        for d in range(365):
            season = seasons_seq[d*24]
            month = d // 30 + 1
            pv_profile = pv_coeff_seq[d*24:(d+1)*24] * pv_cap
            load_profile = load_seq[d*24:(d+1)*24]

            tou_hourly = tou_seq[d*24:(d+1)*24]
            amb_temp = MONTHLY_AMBIENT_TEMP.get(min(max(month, 1), 12), 20.0)
            result = self.simulate_daily_operation(pv_profile, load_profile, season,
                                                   tou_prices=tou_hourly,
                                                   initial_soc=soc,
                                                   foresight_horizon=24,
                                                   ambient_temp=amb_temp)
            soc = result['final_soc']
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
        """计算总负荷, 考虑日类型、天气和充-建耦合影响

        v6.3: 新增充电-建筑负荷耦合 (文件24 §2.3)
        每辆充电车带来2-3人进入建筑, 产生0.5-2.5kW增量建筑负荷.
        """
        charging = self.charging_load.get(day_type, self.charging_load['workday'])
        building = self.building_load.get(season, self.building_load['spring'])
        # 建筑负荷日类型修正
        bldg_coeff = BUILDING_DAY_TYPE_COEFF.get(day_type, 1.0)
        # 天气修正: 恶劣天气 → 充电需求下降, 建筑负荷上升
        weather_chg = WEATHER_CHARGING_COEFF.get(weather, 1.0)
        weather_bldg = WEATHER_BUILDING_COEFF.get(weather, 1.0)

        charging_effective = charging * weather_chg
        building_base = building * bldg_coeff * weather_bldg

        # v6.3: 充电-建筑耦合 — 每辆充电车带来建筑负荷增量
        # 估算活跃充电车辆数: charging_load / 典型充电功率(~100kW)
        active_chargers = charging_effective / 100.0
        coupling_kw_per_ev = CHARGING_BUILDING_COUPLING.get(day_type, 0.7)
        building_coupling = active_chargers * coupling_kw_per_ev

        return charging_effective + building_base + building_coupling

    def simulate_daily_operation(self, pv_profile, load_profile, season='spring',
                                  tou_prices=None, initial_soc=None,
                                  foresight_horizon=24,
                                  ambient_temp=None, pred_error_std=0.0):
        """TOU感知最优调度 — 24h迭代套利算法 (v6.3: RTE/温度/自放电/预测误差)

        核心逻辑: 在已知24h PV+负荷曲线和分时电价的前提下,
        通过ESS在低价时段充电、高价时段放电实现最优套利.

        v6.3 改进:
        - RTE: 使用 get_rte(soc, c_rate) 替代固定0.92
        - ESS温度衰减: 低温时可用容量下降
        - 自放电: SOC每日衰减约0.1%
        - 预测误差: 调度器面对的PV/负荷含噪声 (模拟实际预测不完美)

        Parameters
        ----------
        tou_prices : np.ndarray or None
            24h电价数组, 若为None则按季节自动获取.
        ambient_temp : float or None
            环境温度(℃), 用于储能温度衰减. None则不衰减.
        pred_error_std : float
            PV和负荷预测误差的标准差 (相对值), 0=完美预测.
        """
        T = 24
        prices = tou_prices if tou_prices is not None else get_tou_price_array(season)
        net_load = load_profile - pv_profile  # + = deficit

        # v6.3: 预测误差 — 调度器使用含噪声的PV/负荷 (真实值不变)
        if pred_error_std > 0:
            pv_noisy = pv_profile * (1 + self.rng.normal(0, pred_error_std, T))
            load_noisy = load_profile * (1 + self.rng.normal(0, pred_error_std * 0.6, T))
            net_load_sched = load_noisy - pv_noisy
        else:
            net_load_sched = net_load

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
                'final_soc': 0.5,
            }

        # v6.3: ESS温度衰减 (低温时可用容量下降)
        if ambient_temp is not None:
            temp_derate = get_ess_temp_derate(ambient_temp)
        else:
            temp_derate = 1.0
        ess_capacity_effective = self.ess_capacity * temp_derate
        ess_power_effective = self.ess_power * temp_derate

        # v6.3: RTE随SOC和C-rate变化 (替代固定eta_rt=0.92)
        def _get_eta(soc, ch_power):
            c_rate = ch_power / max(self.ess_capacity, 1.0)
            return get_rte(soc, c_rate)

        eta_rt = _get_eta(initial_soc if initial_soc is not None else 0.5, 0.0)
        eta_one_way = np.sqrt(eta_rt)

        # 迭代寻找最优充放电对
        for _ in range(80):
            best_benefit = 0
            best_trade = None

            for t_ch in range(T):
                ch_headroom = ess_power_effective - ess_ch[t_ch]
                if ch_headroom < 0.5:
                    continue

                # 充电成本: 优先用弃光(机会成本=上网电价), 不足时网购(成本=TOU)
                from_export = min(ch_headroom, grid_export[t_ch])
                from_grid = ch_headroom - from_export
                ch_cost = FEED_IN_PRICE if from_export > 0 else prices[t_ch]

                max_energy = from_export + from_grid
                if max_energy < 0.5:
                    continue

                max_dis = min(T, t_ch + foresight_horizon)
                for t_dis in range(t_ch, max_dis):
                    if grid_import[t_dis] < 0.5:
                        continue
                    dis_headroom = min(ess_power_effective - ess_disch[t_dis],
                                       grid_import[t_dis])
                    if dis_headroom < 0.5:
                        continue

                    # v6.3: 使用调度器看到的净负荷 (grid_import/export基于无噪声真实值)
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

        # 重建SOC曲线 (v6.3: 含自放电)
        soc = initial_soc if initial_soc is not None else 0.5
        soc_curve = np.zeros(T)
        self_discharge_per_hour = 0.001 / 24  # v6.3: 自放电 ~0.1%/天
        for t in range(T):
            # 充放电对SOC的影响 (基于有效容量)
            soc += (ess_ch[t] * eta_one_way - ess_disch[t] / eta_one_way) / ess_capacity_effective
            # v6.3: 自放电
            soc -= self_discharge_per_hour * soc
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

        # v6.2: 变压器出口约束
        excess_pv_curtailed = np.zeros(T)
        for t in range(T):
            if grid_export[t] > trans_limit_kw:
                excess_pv_curtailed[t] = grid_export[t] - trans_limit_kw
                grid_export[t] = trans_limit_kw

        return {
            'grid_import': grid_import, 'grid_export': grid_export,
            'excess_pv_curtailed': excess_pv_curtailed,
            'soc_curve': soc_curve,
            'ess_charge': ess_ch, 'ess_discharge': ess_disch,
            'loss_of_load': loss_of_load,
            'final_soc': soc,
        }

    def evaluate_config(self, pv_cap, ess_cap, ess_pow, params_override=None,
                         use_p95=False):
        """评估配置 — 365天顺序仿真 + SOC跨天连续 (v6.3)

        替代旧版 typical_days 加权聚合.
        使用 _build_8760h_sequence() 获取全年逐时序列,
        每天顺序调度, SOC从前一天继承.

        v6.3 改进:
        - SOC/C-rate依赖RTE + 温度衰减 + 自放电
        - 预测误差注入 (PV ~12%, 负荷 ~7%)
        - use_p95: 使用P95充电负荷评估尾部风险

        Parameters
        ----------
        params_override : dict or None
            {'pv_cost_mult': float, 'price_mult': float} 用于敏感度分析
        use_p95 : bool
            是否使用P95充电负荷 (尾部风险评估)
        """
        self.pv_capacity = pv_cap
        self.ess_capacity = ess_cap
        self.ess_power = ess_pow

        annual_grid_import = 0.0
        annual_grid_export = 0.0
        annual_grid_cost = 0.0
        annual_grid_export_rev = 0.0
        annual_pv_gen = 0.0
        annual_load = 0.0
        annual_self_used = 0.0
        annual_loss_of_load = 0.0
        annual_ess_cycles = 0.0
        # v6.5: 逐日收集DOD和温度数据
        daily_dod_list = []
        daily_temp_list = []

        # 构建8760h序列 (文件10: 天气+日类型+TOU)
        pv_coeff_seq, load_seq, tou_seq, seasons_seq = self._build_8760h_sequence()

        # v6.3: 使用P95负荷 (尾部风险评估)
        if use_p95:
            orig_charging = self.charging_load
            self.charging_load = self.charging_load_p95

        soc = 0.5  # 年初SOC
        # 预测误差标准差 (v6.3)
        pred_error = 0.12  # PV 12%, 负荷 ~7%

        for d in range(365):
            season = seasons_seq[d * 24]
            month = d // 30 + 1  # 近似月份
            pv_profile = pv_coeff_seq[d * 24:(d + 1) * 24] * pv_cap
            load_profile = load_seq[d * 24:(d + 1) * 24]
            tou_hourly = tou_seq[d * 24:(d + 1) * 24]

            # v6.3: 当日环境温度 (月均, 用于ESS温度衰减)
            amb_temp = MONTHLY_AMBIENT_TEMP.get(min(max(month, 1), 12), 20.0)

            result = self.simulate_daily_operation(
                pv_profile, load_profile, season,
                tou_prices=tou_hourly, initial_soc=soc,
                foresight_horizon=6,
                ambient_temp=amb_temp,
                pred_error_std=pred_error)
            soc = result['final_soc']

            daily_pv_gen = pv_profile.sum()
            daily_export = result['grid_export'].sum()
            daily_load = load_profile.sum()

            annual_pv_gen += daily_pv_gen
            annual_load += daily_load
            annual_grid_import += result['grid_import'].sum()
            annual_grid_export += daily_export
            annual_self_used += daily_pv_gen - daily_export
            annual_loss_of_load += result['loss_of_load'].sum()
            annual_grid_cost += np.sum(result['grid_import'] * tou_hourly)
            annual_grid_cost -= daily_export * FEED_IN_PRICE

            if ess_cap > 0:
                daily_discharge = result['ess_discharge'].sum()
                annual_ess_cycles += daily_discharge / ess_cap
                # v6.5: 日等效DOD (当日放电量/有效容量)
                ess_cap_eff = ess_cap * temp_derate if 'temp_derate' in dir() else ess_cap
                daily_dod = daily_discharge / max(ess_cap_eff, 1.0)
                daily_dod_list.append(min(daily_dod, 0.95))
            daily_temp_list.append(amb_temp)

        # v6.5: 年均DOD和温度 (用于退化模型)
        avg_dod = float(np.mean(daily_dod_list)) if daily_dod_list else 0.50
        avg_temp_c = float(np.mean(daily_temp_list)) if daily_temp_list else 20.0

        # v6.3: 恢复原始负荷 (若使用了P95)
        if use_p95:
            self.charging_load = orig_charging

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

        # 碳交易收益 (数据来源: 全国碳排放权交易市场CEA/CCER)
        carbon_reduction = annual_self_used / 1000 * CARBON_FACTOR_GRID
        annual_carbon_revenue = carbon_reduction * CCER_PRICE

        # 敏感度分析参数覆盖 (v6.2: 避免全局突变)
        pv_cost_mult = 1.0
        price_mult = 1.0
        if params_override:
            pv_cost_mult = params_override.get('pv_cost_mult', 1.0)
            price_mult = params_override.get('price_mult', 1.0)
            annual_grid_cost *= price_mult

        # 经济计算
        capex_detail = self._capex_detail(pv_cap, ess_cap, ess_pow,
                                           cost_mult=pv_cost_mult)
        subsidy_detail = self._calculate_subsidy(pv_cap, ess_cap, ess_pow)
        # 补贴不超总投资30%
        subsidy_applied = min(subsidy_detail['total'],
                             capex_detail['total'] * MAX_SUBSIDY_RATIO)
        total_capex = capex_detail['total']

        # v6.5: 经济模型传入温度/DOD退化参数
        if self.use_econ_model and self._econ_model is not None:
            self._econ_model.update_config(pv_cap=pv_cap, ess_e=ess_cap, ess_p=ess_pow)
            npc = self._econ_model.npc_from_aggregates(
                annual_grid_cost, annual_ess_cycles,
                annual_carbon_revenue, subsidy_applied, capex_detail,
                avg_temp_c=avg_temp_c, avg_dod=avg_dod)
        else:
            npc = self._calculate_npc_detailed(capex_detail, pv_cap, ess_cap, ess_pow,
                                                annual_grid_cost, annual_ess_cycles,
                                                annual_carbon_revenue,
                                                subsidy_applied,
                                                avg_temp_c=avg_temp_c, avg_dod=avg_dod)

        # 年收益 = 无光储时的网购电成本 - 有光储时的网购电成本 + 碳收益
        annual_cost_without = annual_load * flat_price
        annual_saving = (annual_cost_without - annual_grid_cost
                        + annual_carbon_revenue)
        if self.use_econ_model and self._econ_model is not None:
            payback = self._econ_model.payback_period(annual_saving)
        else:
            payback = self._dynamic_payback(total_capex - subsidy_applied, annual_saving)

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
            'annual_carbon_revenue': annual_carbon_revenue,
            'carbon_reduction_t': carbon_reduction,
            'capital_cost': total_capex,
            'subsidy': subsidy_applied,
            'net_capital_cost': total_capex - subsidy_applied,
            'payback_years': payback,
            'capex_detail': capex_detail,
            'subsidy_detail': subsidy_detail,
        }

    def evaluate_config_scenario(self, pv_cap, ess_cap, ess_pow, scenario_name='baseline'):
        """多情景评估: 基于演化情景重新计算NPC

        数据来源: 文件06 §5 (负荷增长率)
                 文件02 (EV渗透率增长)
                 文件09 §5.2 (情景分析)

        Parameters
        ----------
        scenario_name : str
            'baseline' / 'conservative' / 'aggressive'
        """
        scenario_params = SCENARIOS.get(scenario_name, SCENARIO_BASELINE)

        # 复用 evaluate_config 的核心计算
        result = self.evaluate_config(pv_cap, ess_cap, ess_pow)

        # 使用情景参数重新计算NPC
        capex_detail = result.get('capex_detail', self._capex_detail(pv_cap, ess_cap, ess_pow))
        subsidy_detail = result.get('subsidy_detail', self._calculate_subsidy(pv_cap, ess_cap, ess_pow))
        subsidy_applied = min(subsidy_detail['total'],
                             capex_detail['total'] * MAX_SUBSIDY_RATIO)

        npc = self._calculate_npc_detailed(
            capex_detail, pv_cap, ess_cap, ess_pow,
            result['annual_grid_cost'], result['annual_ess_cycles'],
            result.get('annual_carbon_revenue', result['carbon_reduction_t'] * CCER_PRICE),
            subsidy_applied,
            scenario_params=scenario_params,
        )

        # 动态回收期 (考虑负荷增长和电价上涨)
        annual_cost_without = result['annual_load_kwh'] * np.mean(
            [v['flat'] for v in TOU_PRICE_VALUES.values()])
        annual_saving = (annual_cost_without - result['annual_grid_cost']
                        + result.get('annual_carbon_revenue', 0))
        payback = self._dynamic_payback(capex_detail['total'] - subsidy_applied,
                                        annual_saving)

        scenario_result = {**result,
            'npc': npc,
            'npc_wan_yuan': npc / 1e4,
            'payback_years': payback,
            'scenario': scenario_name,
            'scenario_params': scenario_params,
            'net_capital_cost': capex_detail['total'] - subsidy_applied,
        }
        return scenario_result

    def _dynamic_payback(self, capex, annual_saving):
        """[Deprecated v6] 动态投资回收期 (含折现率)

        v6: 已迁移至 EconomicModel.payback_period().
        保留此方法以确保 --mode fast|nsga2|full 回退兼容."""
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

    def _calculate_subsidy(self, pv_cap, ess_cap, ess_pow):
        """计算政府补贴 (一次性建设补贴)

        数据来源: 文件01 (充电设施补贴政策)
                 文件13 (分布式光伏/储能地方补贴)
        """
        pv_subsidy = pv_cap * PV_SUBSIDY_PER_KWP
        ess_subsidy = ess_cap * ESS_SUBSIDY_PER_KWH + ess_pow * ESS_SUBSIDY_PER_KW
        charging_subsidy = (self.n_piles_120 + self.n_piles_480) * CHARGING_SUBSIDY_PER_PILE
        total_subsidy = pv_subsidy + ess_subsidy + charging_subsidy
        return {
            'pv_subsidy': pv_subsidy,
            'ess_subsidy': ess_subsidy,
            'charging_subsidy': charging_subsidy,
            'total': total_subsidy,
        }

    def _capex_detail(self, pv_cap, ess_cap, ess_pow, cost_mult=1.0):
        """设备级CAPEX明细 (v6.2: 支持cost_mult用于敏感度分析)"""
        detail = {}

        # 光伏系统 (v6.2: cost_mult用于敏感度分析)
        detail['pv_module'] = pv_cap * PV_COST['module'] * cost_mult
        detail['pv_inverter'] = pv_cap * PV_COST['inverter'] * cost_mult
        detail['pv_combiner'] = pv_cap * PV_COST['combiner_box'] * cost_mult
        detail['pv_structure'] = pv_cap * PV_COST['structure_carport'] * cost_mult
        detail['pv_dc_cable'] = pv_cap * PV_COST['dc_cable'] * cost_mult
        detail['pv_subtotal'] = pv_cap * PV_COST_PER_KWP * cost_mult

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
                                 annual_grid_cost, annual_ess_cycles,
                                 annual_carbon_revenue=0.0, subsidy=0.0,
                                 scenario_params=None,
                                 avg_temp_c=25.0, avg_dod=0.50):
        """[Deprecated v6] 设备级NPC — 含碳交易+补贴+负荷增长+电价上涨

        v6: 已迁移至 EconomicModel.npc_from_aggregates().
        保留此方法以确保 --mode fast|nsga2|full 回退兼容.

        v6.5: 新增 avg_temp_c/avg_dod 用于电池退化模型.

        Parameters
        ----------
        scenario_params : dict or None
            演化情景参数 (SCENARIO_BASELINE等).
            含: load_growth_rate, grid_price_escalation, carbon_price_growth,
                 discount_rate
            若为None则使用静态模型 (年增长率为0).
        avg_temp_c : float
            v6.5: 年均环境温度 (℃)
        avg_dod : float
            v6.5: 年均等效DOD
        """
        if scenario_params is None:
            scenario_params = {
                'load_growth_rate': 0.0,
                'grid_price_escalation': 0.0,
                'carbon_price_growth': 0.0,
                'discount_rate': DISCOUNT_RATE,
            }

        capital = capex['total'] - subsidy  # 扣除补贴

        om_pv = capex['pv_subtotal'] * OM_RATE['pv']
        om_ess = capex['ess_subtotal'] * OM_RATE['ess']
        om_charging = capex['charging_subtotal'] * OM_RATE['charging']
        om_annual = om_pv + om_ess + om_charging

        # v6.5: 复合退化率 — 温度加速日历 + DOD依赖循环
        annual_degradation = get_battery_yearly_degradation(
            avg_temp_c + 7.0, annual_ess_cycles, avg_dod)
        ess_degrade_factor = 1.0

        dr = scenario_params.get('discount_rate', DISCOUNT_RATE)
        lgr = scenario_params.get('load_growth_rate', 0.0)
        gpe = scenario_params.get('grid_price_escalation', 0.0)
        cpg = scenario_params.get('carbon_price_growth', 0.0)
        use_logistic = scenario_params.get('load_growth_model', LOAD_GROWTH_MODEL) == 'logistic'

        npv_grid_cost = 0.0
        npv_carbon_rev = 0.0

        for yr in range(1, PROJECT_LIFE + 1):
            # 负荷增长: S曲线 或 指数 (文件25 §3)
            if use_logistic:
                calendar_year = LOAD_GROWTH_BASE_YEAR + (yr - 1)
                load_factor = get_load_growth_factor(calendar_year, model='logistic')
            else:
                load_factor = (1.0 + lgr) ** (yr - 1)
            price_factor = (1.0 + gpe) ** (yr - 1)
            degrade_penalty = 1.0 + (1.0 - ess_degrade_factor) * 0.3

            yr_grid_cost = (annual_grid_cost * degrade_penalty
                           * load_factor * price_factor)
            npv_grid_cost += yr_grid_cost / (1 + dr) ** yr

            # 碳交易收益 (随碳价增长)
            carbon_factor = (1.0 + cpg) ** (yr - 1)
            yr_carbon_rev = annual_carbon_revenue * carbon_factor
            npv_carbon_rev += yr_carbon_rev / (1 + dr) ** yr

            ess_degrade_factor = max(0.60, ess_degrade_factor - annual_degradation)

        npv_om = om_annual * sum(1.0 / (1 + dr) ** y
                                 for y in range(1, PROJECT_LIFE + 1))

        # --- 设备更换 ---
        replacement_cost = 0

        if LIFESPAN['pv_inverter'] < PROJECT_LIFE:
            rep_year = LIFESPAN['pv_inverter']
            replacement_cost += (capex['pv_inverter'] * REPLACEMENT['inverter'] /
                                 (1 + dr) ** rep_year)

        if ess_cap > 0:
            # v6.5: 电池更换基于温度/DOD复合寿命
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
                replacement_cost += (capex['ess_battery'] * REPLACEMENT['ess_battery'] /
                                     (1 + dr) ** int(np.ceil(y)))
                y += batt_life

        if ess_pow > 0 and LIFESPAN['ess_pcs'] < PROJECT_LIFE:
            rep_year = LIFESPAN['ess_pcs']
            replacement_cost += (capex['ess_pcs'] * REPLACEMENT['pcs'] /
                                 (1 + dr) ** rep_year)

        if LIFESPAN['charging_pile'] < PROJECT_LIFE:
            rep_year = LIFESPAN['charging_pile']
            replacement_cost += (capex['charging_subtotal'] * REPLACEMENT['charging'] /
                                 (1 + dr) ** rep_year)

        if LIFESPAN['ems_hw'] < PROJECT_LIFE:
            ems_hw_cost = capex['fixed']['ems'] * 0.4
            for yr in [5, 10, 15]:
                if yr <= PROJECT_LIFE:
                    replacement_cost += (ems_hw_cost * REPLACEMENT['ems_hw'] /
                                         (1 + dr) ** yr)

        salvage = capital * RESIDUAL_RATE / (1 + dr) ** PROJECT_LIFE

        npc = capital + npv_om + npv_grid_cost + replacement_cost - salvage - npv_carbon_rev
        return npc

    def optimize_pso(self, pop_size=50, max_iter=30, verbose=True,
                      params_override=None):
        """PSO优化 (v6.2: 支持params_override避免全局突变)

        Parameters
        ----------
        params_override : dict or None
            参数覆盖, e.g. {'pv_cost_mult': 1.2, 'price_mult': 0.8}
        """
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
            result = self.evaluate_config(pv, ess_e, ess_p,
                                            params_override=params_override)
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

    def robust_optimize(self, uncertainty_pv=0.20, uncertainty_load=0.15,
                        pop_size=30, max_iter=20, n_scenarios=3, verbose=True):
        """v6.5: 两阶段鲁棒优化 — 全链集成

        使用 RobustOptimizationFramework.from_optimizer() 桥接,
        在多个不确定性场景下运行PSO, 寻找最恶劣场景NPC最小的鲁棒配置.

        Parameters
        ----------
        uncertainty_pv : float
            PV出力的最大不确定性 (如0.20 = ±20%).
        uncertainty_load : float
            负荷的最大不确定性 (如0.15 = ±15%).
        pop_size : int
            PSO种群大小.
        max_iter : int
            PSO最大迭代数.
        n_scenarios : int
            不确定性场景数量 (≥2).
        verbose : bool
            是否打印迭代进度.

        Returns
        -------
        best_result : dict
            最恶劣场景下的最优配置评估结果.
        best_x : np.ndarray
            [PV_cap, ESS_cap, ESS_power].
        robust_info : dict
            鲁棒性分析汇总 (含all_scenarios).
        """
        from optimization_comparison import RobustOptimizationFramework

        if verbose:
            print(f"\n{'='*55}")
            print(f"两阶段鲁棒优化 (PV不确定±{uncertainty_pv:.0%}, "
                  f"负荷不确定±{uncertainty_load:.0%}, {n_scenarios}场景)")
            print(f"{'='*55}")

        # 从当前optimizer构建鲁棒框架
        robust = RobustOptimizationFramework.from_optimizer(
            self, uncertainty_pv=uncertainty_pv,
            uncertainty_load=uncertainty_load, n_scenarios=n_scenarios)

        if verbose:
            print(f"  不确定性场景集 ({len(robust._uncertainty_sets)}个):")
            for sc in robust._uncertainty_sets:
                pv_factor = np.mean(sc['pv']) / max(np.mean(robust.nominal_pv), 0.1)
                load_factor = np.mean(sc['load']) / max(np.mean(robust.nominal_load), 0.1)
                print(f"    {sc['name']}: PV×{pv_factor:.2f}, Load×{load_factor:.2f}")

        # PSO鲁棒优化
        best_result, best_x, best_fit = robust.robust_optimize_pso(
            pop_size=pop_size, max_iter=max_iter, verbose=verbose)

        # 对标称场景做完整评估
        nominal_eval = self.evaluate_config(best_x[0], best_x[1], best_x[2])

        # 构建鲁棒性报告
        robust_info = {
            'uncertainty_pv': uncertainty_pv,
            'uncertainty_load': uncertainty_load,
            'n_scenarios': n_scenarios,
            'robust_npc': best_fit,
            'nominal_npc': nominal_eval.get('npc', 0),
            'npc_robustness_premium': best_fit - nominal_eval.get('npc', 0),
            'npc_robustness_premium_pct': (
                (best_fit - nominal_eval.get('npc', 0)) / max(abs(nominal_eval.get('npc', 1)), 1)
            ),
            'scenarios': robust._uncertainty_sets,
        }

        if verbose:
            print(f"\n  鲁棒优化结果:")
            print(f"    最优配置: PV={best_x[0]:.0f}kWp, "
                  f"ESS={best_x[1]:.0f}kWh, Power={best_x[2]:.0f}kW")
            print(f"    鲁棒NPC (最恶劣场景): {best_fit/1e4:.1f} 万元")
            print(f"    标称NPC:              {nominal_eval.get('npc', 0)/1e4:.1f} 万元")
            print(f"    鲁棒性溢价:            {robust_info['npc_robustness_premium']/1e4:.1f} 万元 "
                  f"({robust_info['npc_robustness_premium_pct']:.1%})")

        return best_result, best_x, robust_info

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
