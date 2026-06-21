"""
主程序 — 高速服务区光储充一体化仿真全流程 (v6)
运行: python main.py [--mode fast|nsga2|full|v6]
"""
import numpy as np
import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from mc_charging_load import MonteCarloChargingSimulator, print_summary
from pv_generation import PVGenerator
from capacity_optimization import MicrogridOptimizer
from calendar_utils import CalendarContext
from visualization import (
    fig1_charging_load_probability, fig2_scenario_comparison,
    fig3_power_balance, fig4_pareto_frontier,
    fig5_sensitivity_heatmap, fig6_radar_chart,
    fig7_monthly_energy_balance,
    fig8_decision_topsis, fig9_scenario_comparison,
    fig10_pareto_3d,
    fig11_topology_radar, fig12_economic_breakdown,
    fig13_decision_cross_validation,
)
from config import (
    SERVICE_AREA_CONFIG, PV_COEFF, WEATHER_COEFF,
    PV_AREA_RATIO, SELF_SUFFICIENCY_MIN, PROJECT_LIFE,
)
from decision_framework import DecisionFramework
from nsga2 import NSGA2, print_pareto_summary, save_pareto_results

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
SEED = 42


def run_monte_carlo(n_runs=3000):
    """Step 1: 蒙特卡洛充电负荷仿真"""
    print("\n" + "=" * 60)
    print("Step 1: Monte Carlo EV Charging Load Simulation")
    print("=" * 60)

    sim = MonteCarloChargingSimulator(service_area_size='medium', seed=SEED)
    scenarios = sim.simulate_all_scenarios(n_runs=n_runs)
    print_summary(scenarios)
    return scenarios


def run_pv_synthesis(pv_capacity=500, calendar_ctx=None):
    """Step 2: 光伏出力合成"""
    print("\n" + "=" * 60)
    print("Step 2: PV Generation Synthesis")
    print("=" * 60)

    pv = PVGenerator(pv_capacity_kwp=pv_capacity, seed=SEED, calendar_ctx=calendar_ctx)
    annual = pv.generate_annual()
    metrics = pv.compute_annual_metrics(annual)
    monthly = pv.get_monthly_generation(annual)

    print(f"PV Capacity: {pv_capacity} kWp")
    print(f"Annual Generation: {metrics['annual_generation_kwh']:.0f} kWh")
    print(f"Equivalent Hours: {metrics['equivalent_hours']:.1f} h")
    return pv, annual, monthly, metrics


def run_optimization(mc_scenarios, size='medium', calendar_ctx=None):
    """Step 3: 容量优化"""
    print("\n" + "=" * 60)
    print("Step 3: System Capacity Optimization (PSO)")
    print("=" * 60)

    opt = MicrogridOptimizer(size=size, mc_scenarios=mc_scenarios, seed=SEED,
                              calendar_ctx=calendar_ctx)
    result, best_x = opt.optimize_pso(pop_size=50, max_iter=40, verbose=True)

    print(f"\n--- PSO Optimal (typical-day weighted) ---")
    print(f"  PV: {result['pv_capacity']:.0f} kWp")
    print(f"  ESS Capacity: {result['ess_capacity']:.0f} kWh")
    print(f"  ESS Power: {result['ess_power']:.0f} kW")
    print(f"  NPC: {result['npc']/1e4:.1f} wan yuan")
    print(f"  Self-sufficiency: {result['self_sufficiency']:.1%}")
    print(f"  Annual CO2 reduction: {result['carbon_reduction_t']:.1f} t")
    print(f"  Capital cost: {result['capital_cost']/1e4:.1f} wan yuan")
    print(f"  Payback: {result['payback_years']:.1f} years")

    # 8760h 全时序验证
    print(f"\n--- 8760h Full-year Verification ---")
    verify = opt.verify_8760h(result['pv_capacity'],
                              result['ess_capacity'],
                              result['ess_power'])
    print(f"  Self-sufficiency (8760h): {verify['self_sufficiency']:.1%}")
    print(f"  Loss of Load: {verify['loss_of_load_pct']:.3%}")
    print(f"  Annual ESS cycles: {verify['ess_cycles']:.0f}")
    print(f"  Grid cost: {verify['grid_cost']/1e4:.1f} wan yuan")
    result['verify_8760h'] = verify

    return opt, result, best_x


def run_pareto_analysis(opt, n_samples=200):
    """Step 4: Pareto前沿分析 (使用非支配排序)"""
    print("\n" + "=" * 60)
    print("Step 4: Pareto Frontier Analysis")
    print("=" * 60)

    results = []
    rng = np.random.RandomState(SEED)
    max_pv = opt.area_config['pv_area_m2'] / PV_AREA_RATIO

    for i in range(n_samples):
        pv = rng.uniform(50, max_pv * 0.95)
        ess_e = rng.uniform(0, 3000)
        ess_p = rng.uniform(0, min(ess_e * 0.5, 800))

        result = opt.evaluate_config(pv, ess_e if ess_e > 20 else 0,
                                      ess_p if ess_e > 20 else 0)
        results.append(result)

    # 非支配排序: 目标 min NPC, max SSR
    pareto_front = []
    for i, ri in enumerate(results):
        dominated = False
        for j, rj in enumerate(results):
            if i == j: continue
            # j dominates i if: NPC_j <= NPC_i AND SSR_j >= SSR_i AND at least one strictly better
            if rj['npc'] <= ri['npc'] and rj['self_sufficiency'] >= ri['self_sufficiency']:
                if rj['npc'] < ri['npc'] or rj['self_sufficiency'] > ri['self_sufficiency']:
                    dominated = True
                    break
        if not dominated:
            pareto_front.append(ri)

    # 按SSR排序
    pareto_front.sort(key=lambda x: x['self_sufficiency'])
    print(f"Total samples: {len(results)}, Pareto-optimal: {len(pareto_front)}")
    if pareto_front:
        print(f"SSR range: {pareto_front[0]['self_sufficiency']:.1%} - {pareto_front[-1]['self_sufficiency']:.1%}")
    return pareto_front


def run_sensitivity(opt):
    """Step 5: 敏感性分析"""
    print("\n" + "=" * 60)
    print("Step 5: Sensitivity Analysis")
    print("=" * 60)

    pv_mult = [0.6, 0.8, 1.0, 1.2, 1.4]
    price_mult = [0.6, 0.8, 1.0, 1.2, 1.4]

    pv_cap_grid = np.zeros((5, 5))
    npc_grid = np.zeros((5, 5))

    for i, pvm in enumerate(pv_mult):
        for j, prm in enumerate(price_mult):
            # v6.2: params_override替代全局突变
            result, _ = opt.optimize_pso(
                pop_size=30, max_iter=25, verbose=False,
                params_override={'pv_cost_mult': pvm, 'price_mult': prm})

            pv_cap_grid[i, j] = result['pv_capacity']
            npc_grid[i, j] = result['npc']

    print("Sensitivity analysis complete")
    return {
        'pv_cost_mult': pv_mult,
        'grid_price_mult': price_mult,
        'pv_capacity': pv_cap_grid,
        'npc': npc_grid,
    }


def run_scheme_comparison(opt):
    """Step 6: 不同方案对比 (方案4使用放宽场地约束后的PSO优化)"""
    print("\n" + "=" * 60)
    print("Step 6: Multi-scheme Comparison")
    print("=" * 60)

    # 方案1: 纯电网
    s1 = opt.evaluate_config(0, 0, 0)
    # 方案2: 仅光伏 (场地受限最优)
    s2, _ = opt.optimize_pso(pop_size=30, max_iter=20, verbose=False)
    # 暂存ess配置, 取纯光伏部分
    pv_only_cap = s2['pv_capacity']
    s2_pure = opt.evaluate_config(pv_only_cap, 0, 0)

    # 方案3: 光伏+储能(完整优化)
    s3, _ = opt.optimize_pso(pop_size=50, max_iter=40, verbose=False)

    # 方案4: 大光伏+大储能 (放宽场地约束, 只优化自洽率)
    # 临时扩大可用面积
    orig_area = opt.area_config['pv_area_m2']
    opt.area_config['pv_area_m2'] = orig_area * 1.5
    s4, _ = opt.optimize_pso(pop_size=50, max_iter=40, verbose=False)
    opt.area_config['pv_area_m2'] = orig_area

    results_list = [
        ('Grid Only', s1),
        ('PV Only', s2_pure),
        ('PV+ESS (Optimal)', s3),
        ('Large PV+ESS', s4),
    ]

    npc_values = [r[1]['npc'] for r in results_list if r[1]['npc'] > 0]
    max_npc = max(npc_values) if npc_values else 1e6

    schemes = {}
    for name, r in results_list:
        # 归一化到0-100
        econ_score = max(0, (1 - r['npc'] / max_npc)) * 100 if r['npc'] > 0 else 100
        carbon_score = min(100, r['carbon_reduction_t'] / 800 * 100)
        # 光伏消纳率
        pv_util = (1 - r['annual_grid_export_kwh'] / max(r['annual_pv_gen_kwh'], 1)) * 100
        # 供电可靠率
        reliability = 99.5

        schemes[name] = [
            r['self_sufficiency'] * 100,
            econ_score,
            carbon_score,
            min(100, max(10, 100 - r['payback_years'] * 5)),
            pv_util,
            reliability,
        ]

    for name, scores in schemes.items():
        print(f"  {name}: SSR={scores[0]:.1f}%, PV_util={scores[4]:.1f}%")

    print("Scheme comparison complete")
    return schemes


def run_nsga2_optimization(opt, pop_size=80, n_gen=40, verbose=True):
    """Step 3a: NSGA-II多目标优化 (理论来源: 文件08)"""
    print("\n" + "=" * 60)
    print("Step 3a: NSGA-II Multi-Objective Optimization")
    print("=" * 60)

    max_pv = opt.area_config['pv_area_m2'] / PV_AREA_RATIO
    bounds = np.array([
        [50, max_pv],
        [0, 3000],
        [0, 1000],
    ])

    nsga2 = NSGA2(opt.evaluate_config, bounds, pop_size=pop_size,
                  n_gen=n_gen, seed=SEED)
    pareto_solutions, pareto_obj, history = nsga2.optimize(verbose=verbose)

    print_pareto_summary(pareto_solutions)
    save_pareto_results(pareto_solutions)

    # 找折中最优解 (SSR > 25% 中NPC最低)
    feasible = [s for s in pareto_solutions if s['self_sufficiency'] > 0.25]
    if feasible:
        best = min(feasible, key=lambda s: s['npc'])
        print(f"\n推荐折中解: PV={best['pv_capacity']:.0f}kWp, "
              f"ESS={best['ess_capacity']:.0f}kWh, "
              f"NPC={best['npc_wan_yuan']:.1f}万元, "
              f"SSR={best['ssr_pct']:.1f}%")
        nsga2_best = best
    else:
        nsga2_best = None

    return pareto_solutions, nsga2_best


def run_decision_framework(opt):
    """Step 6a: AHP-TOPSIS综合决策遴选 (理论来源: 文件09)"""
    print("\n" + "=" * 60)
    print("Step 6a: AHP-TOPSIS Decision Framework")
    print("=" * 60)

    # 四方案评估 (简化: 使用典型配置)
    # 方案1: 纯电网
    s1 = opt.evaluate_config(0, 0, 0)
    # 方案2: 仅光伏 (场地受限最优)
    max_pv = opt.area_config['pv_area_m2'] / PV_AREA_RATIO
    pv_only_cap = max_pv * 0.8
    s2 = opt.evaluate_config(pv_only_cap, 0, 0)
    # 方案3: 光伏+储能 (当前最优)
    s3, _ = opt.optimize_pso(pop_size=30, max_iter=25, verbose=False)
    # 方案4: 大光伏+大储能
    s4 = opt.evaluate_config(max_pv * 0.95, 2500, 1200)

    schemes = ['方案A:纯电网', '方案B:仅光伏', '方案C:光储均衡', '方案D:大光储']
    indicators = ['能源自洽率(%)', 'NPC(万元)', '碳减排(tCO2/年)',
                  '投资回收期(年)', '光伏消纳率(%)', '供电可靠率(%)']
    directions = ['benefit', 'cost', 'benefit', 'cost', 'benefit', 'benefit']

    values = np.array([
        [s1['self_sufficiency'] * 100, s1['npc'] / 1e4,
         s1['carbon_reduction_t'], s1['payback_years'],
         min(100, (1 - s1.get('annual_grid_export_kwh', 0) /
          max(s1.get('annual_pv_gen_kwh', 1), 1)) * 100), 99.5],
        [s2['self_sufficiency'] * 100, s2['npc'] / 1e4,
         s2['carbon_reduction_t'], s2['payback_years'],
         min(100, (1 - s2.get('annual_grid_export_kwh', 0) /
          max(s2.get('annual_pv_gen_kwh', 1), 1)) * 100), 99.5],
        [s3['self_sufficiency'] * 100, s3['npc'] / 1e4,
         s3['carbon_reduction_t'], s3['payback_years'],
         min(100, (1 - s3.get('annual_grid_export_kwh', 0) /
          max(s3.get('annual_pv_gen_kwh', 1), 1)) * 100), 99.8],
        [s4['self_sufficiency'] * 100, s4['npc'] / 1e4,
         s4['carbon_reduction_t'], s4['payback_years'],
         min(100, (1 - s4.get('annual_grid_export_kwh', 0) /
          max(s4.get('annual_pv_gen_kwh', 1), 1)) * 100), 99.0],
    ])

    criteria_labels = ['能源性', '经济性', '可靠性', '环境性']
    indicator_criteria_map = [0, 1, 1, 1, 0, 2]

    df = DecisionFramework()
    result = df.evaluate(
        schemes=schemes, indicators=indicators,
        values=values, directions=directions,
        criteria_labels=criteria_labels,
        indicator_criteria_map=indicator_criteria_map,
    )
    result.print_summary()
    result.save()

    print(f"\n  推荐方案: {result.get_best_scheme()}")
    return result


def run_multi_scenario(opt, pv_cap, ess_cap, ess_pow):
    """Step 6b: 多情景分析 (保守/基准/激进)"""
    print("\n" + "=" * 60)
    print("Step 6b: Multi-Scenario Analysis")
    print("=" * 60)

    scenario_results = {}
    for sc_name in ['conservative', 'baseline', 'aggressive']:
        print(f"  Running {sc_name} scenario...")
        result = opt.evaluate_config_scenario(pv_cap, ess_cap, ess_pow, sc_name)
        scenario_results[sc_name] = result
        print(f"    NPC={result['npc_wan_yuan']:.1f}万元, "
              f"SSR={result['self_sufficiency']:.1%}, "
              f"Payback={result['payback_years']:.1f}yr")

    return scenario_results


def generate_all_figures(mc_scenarios, pv_gen, opt, opt_result, pareto_results,
                         sensitivity_data, scheme_results, calendar_ctx=None,
                         decision_result=None, scenario_results=None,
                         pareto_nsga2=None):
    """生成全部图表 (v5: 新增Fig8-10)"""
    print("\n" + "=" * 60)
    print("Generating Figures...")
    print("=" * 60)

    # 图1
    fig1_charging_load_probability(mc_scenarios)
    print("  [OK] fig1_charging_load_probability")

    # 图2
    fig2_scenario_comparison(mc_scenarios)
    print("  [OK] fig2_scenario_comparison")

    # 图3: 最优配置下的夏季晴天功率平衡 (含温度效应)
    pv_cap = opt_result['pv_capacity']
    # 使用PVGenerator获取温度修正后的出力
    temp_pv = PVGenerator(pv_capacity_kwp=pv_cap, seed=SEED, calendar_ctx=calendar_ctx)
    pv_profile = temp_pv.generate_daily_profile('summer', 'clear')
    load_profile = opt.get_total_load('summer', 'workday', 'clear')
    op_day = opt.simulate_daily_operation(pv_profile, load_profile)
    fig3_power_balance(pv_profile, load_profile, op_day, 'summer', 'clear')
    print("  [OK] fig3_power_balance (with temperature effect)")

    # 图4
    fig4_pareto_frontier(pareto_results)
    print("  [OK] fig4_pareto_frontier")

    # 图5
    fig5_sensitivity_heatmap(sensitivity_data)
    print("  [OK] fig5_sensitivity_heatmap")

    # 图6
    fig6_radar_chart(scheme_results)
    print("  [OK] fig6_radar_chart")

    # 图7: 月度能量平衡
    best_pv = PVGenerator(pv_capacity_kwp=opt_result['pv_capacity'], seed=SEED,
                          calendar_ctx=calendar_ctx)
    monthly_pv = best_pv.get_monthly_generation(best_pv.generate_annual())
    monthly_load = np.ones(12) * opt_result['annual_load_kwh'] / 12
    fig7_monthly_energy_balance(monthly_pv, monthly_load)
    print("  [OK] fig7_monthly_energy_balance")

    # --- v5 新增图表 ---
    # 图8: TOPSIS决策框架
    if decision_result is not None:
        fig8_decision_topsis(decision_result)
        print("  [OK] fig8_decision_topsis")

    # 图9: 多情景对比
    if scenario_results is not None:
        fig9_scenario_comparison(scenario_results)
        print("  [OK] fig9_scenario_comparison")

    # 图10: NSGA-II 3D Pareto
    if pareto_nsga2 is not None and len(pareto_nsga2) > 0:
        fig10_pareto_3d(pareto_nsga2)
        print("  [OK] fig10_pareto_3d")

    figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
    print(f"\nAll figures saved to: {os.path.abspath(figures_dir)}")


def save_results(mc_scenarios, pv_metrics, opt_result, pareto_results):
    """保存仿真结果到JSON"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mc_summary = {}
    for dt, res in mc_scenarios.items():
        mc_summary[dt] = {
            'peak_mean': float(res['peak_mean']),
            'peak_std': float(res['peak_std']),
            'peak_p95': float(res['p95'].max()),
            'daily_energy_mean': float(res['daily_energy_mean']),
            'hourly_p50': res['p50'].tolist(),
            'hourly_p95': res['p95'].tolist(),
        }
    with open(os.path.join(OUTPUT_DIR, 'mc_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(mc_summary, f, indent=2, ensure_ascii=False)

    opt_summary = {}
    for k, v in opt_result.items():
        if isinstance(v, (np.floating, np.integer)):
            opt_summary[k] = float(v)
        elif isinstance(v, (int, float)):
            opt_summary[k] = v
        elif isinstance(v, dict):
            # 嵌套字典 (如subsidy_detail)
            opt_summary[k] = {sk: (float(sv) if isinstance(sv, (np.floating, np.integer)) else sv)
                              for sk, sv in v.items()}
    # v5新增: 补贴和碳交易指标
    for k in ['annual_carbon_revenue', 'subsidy', 'net_capital_cost',
              'annual_ess_cycles']:
        if k in opt_result and k not in opt_summary:
            opt_summary[k] = float(opt_result[k]) if not isinstance(opt_result[k], dict) else opt_result[k]
    with open(os.path.join(OUTPUT_DIR, 'optimization_result.json'), 'w', encoding='utf-8') as f:
        json.dump(opt_summary, f, indent=2, ensure_ascii=False)

    pareto_list = []
    for r in pareto_results[::2]:
        pareto_list.append({
            'pv': float(r['pv_capacity']),
            'ess': float(r['ess_capacity']),
            'npc_wan': float(r['npc'] / 1e4),
            'ssr': float(r['self_sufficiency']),
        })
    with open(os.path.join(OUTPUT_DIR, 'pareto_results.json'), 'w', encoding='utf-8') as f:
        json.dump(pareto_list, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")


# ============================================================
# v6 新增流程函数
# ============================================================

def run_pv_resource_analysis(area_size='medium', zone='III'):
    """v6 Step 2a: 光伏资源区评估报告"""
    print("\n" + "=" * 60)
    print("Step 2a: PV Resource Zone Analysis (v6)")
    print("=" * 60)
    from pv_resource_analysis import (
        PVResourceAnalyzer, compare_all_zones, compute_service_area_pv_potential
    )
    compare_all_zones(area_size)
    analyzer = PVResourceAnalyzer(zone=zone, area_size=area_size, pv_capacity_kwp=1231)
    analyzer.print_potential_summary()
    return analyzer


def run_topology_comparison(area_size='medium', pv_cap=1231, ess_cap=2000, ess_pow=1000):
    """v6 Step 6c: 微网拓扑架构对比"""
    print("\n" + "=" * 60)
    print("Step 6c: Topology Architecture Comparison (v6)")
    print("=" * 60)
    from topology_comparison import TopologyAnalyzer
    ta = TopologyAnalyzer(area_size, pv_cap, ess_cap, ess_pow)
    ta.print_comparison()
    topo_names, topo_scores = ta.get_soi_scores()
    return ta, topo_names, topo_scores


def run_algorithm_comparison(opt):
    """v6 Step 3b: GA vs EGPSO 算法基准测试"""
    print("\n" + "=" * 60)
    print("Step 3b: Algorithm Benchmark (v6)")
    print("=" * 60)
    from optimization_comparison import AlgorithmBenchmark

    def evaluator(x):
        pv, ess_e, ess_p = x[0], x[1], x[2]
        if ess_e < 10:
            ess_e = 0; ess_p = 0
        ess_p = min(ess_p, max(ess_e * 0.5, 0.1))
        result = opt.evaluate_config(pv, ess_e, ess_p)
        npc_val = result['npc']
        ssr = result['self_sufficiency']
        penalty = 0
        if ssr < SELF_SUFFICIENCY_MIN:
            penalty += (SELF_SUFFICIENCY_MIN - ssr) * npc_val * 2.0
        return npc_val + penalty

    max_pv = opt.area_config['pv_area_m2'] / PV_AREA_RATIO
    bounds = np.array([[50, max_pv], [0, 3000], [0, 1000]])

    bench = AlgorithmBenchmark(evaluator, bounds, seed=SEED)
    results = bench.run_comparison(['GA', 'EGPSO'], n_runs=2)
    bench.print_comparison(results)
    return results


def run_cross_validation(schemes, indicator_names, values, directions,
                          weights=None, criteria_labels=None):
    """v6 Step 6b: VIKOR/GRA/Borda 多方法交叉验证"""
    print("\n" + "=" * 60)
    print("Step 6b: Multi-Method Decision Cross-Validation (v6)")
    print("=" * 60)
    from advanced_decision_methods import MultiMethodDecision
    mdm = MultiMethodDecision(schemes, indicator_names, values,
                               directions, weights, criteria_labels)
    mdm.run_all()
    mdm.print_comparison()
    best_idx, borda = mdm.get_best_by_consensus()
    print(f"\n  Borda共识最优: {schemes[best_idx]} (得分: {borda[best_idx]:.0f})")
    return mdm, best_idx, borda


def run_economic_detailed(opt, result, opt_result):
    """v6 Step 6d: 独立经济模型全生命周期评估"""
    print("\n" + "=" * 60)
    print("Step 6d: Detailed Economic Analysis (v6)")
    print("=" * 60)
    from economic_model import EconomicModel
    em = EconomicModel(
        pv_capacity=opt_result['pv_capacity'],
        ess_capacity=opt_result['ess_capacity'],
        ess_power=opt_result['ess_power'],
        n_piles_120kw=opt.n_piles_120,
        n_piles_480kw=opt.n_piles_480,
        scenario='baseline',
    )
    em.print_cost_breakdown()

    annual_saving = opt_result.get('annual_grid_cost', 300000) * 0.5
    npc_val = em.npc_simple(
        annual_grid_cost=opt_result.get('annual_grid_cost', 300000),
        annual_self_consumed_kwh=opt_result.get('annual_self_used_kwh', 0),
        annual_grid_export_kwh=opt_result.get('annual_grid_export_kwh', 0))
    lcoe_val = em.lcoe(npc_val, opt_result.get('annual_load_kwh', 1))
    pbp_val = em.payback_period(annual_saving)
    try:
        cash_flows = {y: annual_saving for y in range(1, PROJECT_LIFE + 1)}
        irr_val = em.irr(cash_flows)
    except Exception:
        irr_val = 0.0

    print(f"\n  关键经济指标:")
    print(f"  NPC (20年): {npc_val/1e4:.1f} 万元")
    print(f"  LCOE: {lcoe_val:.4f} 元/kWh")
    print(f"  动态回收期: {pbp_val:.1f} 年")
    print(f"  IRR: {irr_val:.1%}")

    print(f"\n  三情景NPC对比:")
    for sc in ['conservative', 'baseline', 'aggressive']:
        npc_sc = em.npc_with_scenario(
            np.zeros(8760), np.ones(8760) * 0.8,
            opt_result.get('annual_self_used_kwh', 0), 0, sc)
        print(f"    {sc}: NPC={npc_sc/1e4:.1f}万元")

    return em, npc_val, lcoe_val, pbp_val, irr_val


def run_dashboard_from_data(mc_scenarios, opt_result, opt=None, calendar_ctx=None):
    """v6 Step 7: 交互式仪表盘 (复用已有MC数据)"""
    print("\n" + "=" * 60)
    print("Step 7: Interactive Dashboard (v6)")
    print("=" * 60)
    from interactive_dashboard import build_8760h_data, create_dashboard
    pv_cap = opt_result['pv_capacity']
    ess_cap = opt_result['ess_capacity']
    ess_pow = opt_result['ess_power']

    (daily_results, pv_coeff_seq, load_seq, tou_seq,
     cal_ctx, summary) = build_8760h_data(mc_scenarios, pv_cap, ess_cap, ess_pow,
                                           opt=opt, calendar_ctx=calendar_ctx)

    print(f"  Self-sufficiency: {summary['self_sufficiency']:.1%}")
    print(f"  Annual PV Gen: {summary['total_pv_gen']/1000:.0f} MWh")
    print(f"  Grid Cost: {summary['grid_cost']/1e4:.1f} wan yuan")

    output = create_dashboard(daily_results, pv_coeff_seq, load_seq, tou_seq,
                               cal_ctx, summary, pv_cap, ess_cap, ess_pow)
    return output


def main_robust():
    """v6.5 两阶段鲁棒优化: PSO标称优化 + 鲁棒优化 + 对比分析"""
    print("=" * 60)
    print("Highway Service Area PV-Storage-Charging Simulation v6.5")
    print("Mode: Two-Stage Robust Optimization")
    print("=" * 60)

    from optimization_comparison import RobustOptimizationFramework

    SEED = 42
    calendar_ctx = CalendarContext(seed=SEED)
    calendar_ctx.print_summary()

    # Step 1: Monte Carlo (reduced runs for speed)
    print("\n[Step 1] Monte Carlo Charging Load Simulation...")
    mc_sim = MonteCarloChargingSimulator(service_area_size='medium', seed=SEED)
    mc_scenarios = mc_sim.simulate_all_scenarios(n_runs=2000)

    # Step 2: Standard PSO optimization
    print("\n[Step 2] Standard PSO Optimization (nominal scenario)...")
    opt = MicrogridOptimizer(size='medium', mc_scenarios=mc_scenarios, seed=SEED,
                             calendar_ctx=calendar_ctx, use_econ_model=True)
    nominal_result, nominal_x = opt.optimize_pso(pop_size=30, max_iter=20, verbose=True)

    print(f"\n  Nominal optimal: PV={nominal_x[0]:.0f}kWp, "
          f"ESS={nominal_x[1]:.0f}kWh, Power={nominal_x[2]:.0f}kW")
    print(f"  Nominal NPC: {nominal_result['npc']/1e4:.1f}万元, "
          f"SSR: {nominal_result['self_sufficiency']:.1%}")

    # Step 3: Robust optimization
    print("\n[Step 3] Two-Stage Robust Optimization (PV ±20%, Load ±15%)...")
    robust_result, robust_x, robust_info = opt.robust_optimize(
        uncertainty_pv=0.20, uncertainty_load=0.15,
        pop_size=30, max_iter=20, n_scenarios=3, verbose=True)

    # Step 4: Comparison
    print(f"\n{'='*55}")
    print(f"标称 vs 鲁棒 配置对比")
    print(f"{'='*55}")
    print(f"{'指标':<20} {'标称PSO':>15} {'鲁棒PSO':>15} {'差异':>15}")
    print(f"{'-'*65}")
    print(f"{'PV (kWp)':<20} {nominal_x[0]:>15.0f} {robust_x[0]:>15.0f} "
          f"{robust_x[0]-nominal_x[0]:>+15.0f}")
    print(f"{'ESS (kWh)':<20} {nominal_x[1]:>15.0f} {robust_x[1]:>15.0f} "
          f"{robust_x[1]-nominal_x[1]:>+15.0f}")
    print(f"{'ESS Power (kW)':<20} {nominal_x[2]:>15.0f} {robust_x[2]:>15.0f} "
          f"{robust_x[2]-nominal_x[2]:>+15.0f}")
    print(f"{'NPC (万元)':<20} {nominal_result['npc']/1e4:>15.1f} "
          f"{robust_info['robust_npc']/1e4:>15.1f} "
          f"{robust_info['npc_robustness_premium']/1e4:>+15.1f}")
    print(f"{'SSR':<20} {nominal_result['self_sufficiency']:>15.1%} "
          f"{robust_info['robust_npc']/1e4:>15}")

    print(f"\n  鲁棒性溢价: {robust_info['npc_robustness_premium']/1e4:.1f} 万元 "
          f"({robust_info['npc_robustness_premium_pct']:.1%})")
    print(f"  → 为抵御PV±20%/负荷±15%不确定性, 需额外投资")

    print("\n" + "=" * 60)
    print("Robust Optimization Complete!")
    print("=" * 60)


def main_v6():
    """v6 全流程: 全部6大研究任务 + 交叉验证 + 经济模型 + 仪表盘"""
    print("=" * 60)
    print("Highway Service Area PV-Storage-Charging Simulation v6")
    print("Full Pipeline: All 6 Research Tasks + Cross-Validation")
    print("=" * 60)

    calendar_ctx = CalendarContext(seed=SEED)
    calendar_ctx.print_summary()

    # Step 1: MC 充电负荷
    mc_scenarios = run_monte_carlo(n_runs=5000)

    # Step 2: PV 出力合成
    pv_gen, annual_pv, monthly_pv, pv_metrics = run_pv_synthesis(
        pv_capacity=500, calendar_ctx=calendar_ctx)

    # Step 2a: PV 资源区分析 [NEW]
    run_pv_resource_analysis(area_size='medium', zone='III')

    # Step 3: PSO 优化 (v6: 启用 EconomicModel + ABM)
    print("\n" + "=" * 60)
    print("Step 3: PSO Optimization (v6: EconomicModel + ABM enabled)")
    print("=" * 60)
    opt = MicrogridOptimizer(size='medium', mc_scenarios=mc_scenarios,
                              seed=SEED, calendar_ctx=calendar_ctx,
                              use_abm=True, use_econ_model=True, abm_seed=SEED)
    opt_result, best_x = opt.optimize_pso(pop_size=50, max_iter=40, verbose=True)
    print(f"\n--- PSO Optimal ---")
    print(f"  PV: {opt_result['pv_capacity']:.0f} kWp")
    print(f"  ESS Capacity: {opt_result['ess_capacity']:.0f} kWh")
    print(f"  ESS Power: {opt_result['ess_power']:.0f} kW")
    print(f"  NPC: {opt_result['npc']/1e4:.1f} wan yuan")
    print(f"  Self-sufficiency: {opt_result['self_sufficiency']:.1%}")
    print(f"  Payback: {opt_result['payback_years']:.1f} years")

    # 8760h 验证
    print(f"\n--- 8760h Full-year Verification ---")
    verify = opt.verify_8760h(opt_result['pv_capacity'],
                               opt_result['ess_capacity'],
                               opt_result['ess_power'])
    print(f"  Self-sufficiency (8760h): {verify['self_sufficiency']:.1%}")
    opt_result['verify_8760h'] = verify

    # Step 3a: NSGA-II
    pareto_nsga2, nsga2_best = run_nsga2_optimization(
        opt, pop_size=60, n_gen=30)

    # Step 3b: GA/EGPSO 算法对比 [NEW]
    run_algorithm_comparison(opt)

    # Step 4: Pareto
    pareto_results = run_pareto_analysis(opt, n_samples=120)

    # Step 5: Sensitivity
    sensitivity_data = run_sensitivity(opt)

    # Step 6: 四方案对比
    scheme_results = run_scheme_comparison(opt)

    # Step 6a: AHP-TOPSIS
    decision_result = run_decision_framework(opt)

    # Step 6b: VIKOR/GRA 交叉验证 [NEW]
    schemes_list = ['方案A:纯电网', '方案B:仅光伏', '方案C:光储均衡', '方案D:大光储']
    indicators_list = ['能源自洽率(%)', 'NPC(万元)', '碳减排(tCO2/年)',
                        '投资回收期(年)', '光伏消纳率(%)', '供电可靠率(%)']
    directions_list = ['benefit', 'cost', 'benefit', 'cost', 'benefit', 'benefit']

    # 复用 decision_result 的 values
    mdm, best_idx, borda = run_cross_validation(
        schemes_list, indicators_list,
        np.array(decision_result.values),
        directions_list,
        weights=decision_result.combined_weights)

    # Step 6c: 拓扑架构对比 [NEW]
    ta, topo_names, topo_scores = run_topology_comparison(
        area_size='medium',
        pv_cap=opt_result['pv_capacity'],
        ess_cap=opt_result['ess_capacity'],
        ess_pow=opt_result['ess_power'])

    # Step 6d: 经济模型评估 [NEW]
    em, npc_val, lcoe_val, pbp_val, irr_val = run_economic_detailed(
        opt, opt_result, opt_result)

    # 多情景分析
    scenario_results = run_multi_scenario(
        opt, opt_result['pv_capacity'],
        opt_result['ess_capacity'], opt_result['ess_power'])

    # Step 7: 交互仪表盘 [NEW]
    dashboard_path = run_dashboard_from_data(
        mc_scenarios, opt_result, opt=opt, calendar_ctx=calendar_ctx)

    # Save results
    save_results(mc_scenarios, pv_metrics, opt_result, pareto_results)

    # Step 8: 全部图表 (v5 + v6)
    print("\n" + "=" * 60)
    print("Generating All Figures (v5 + v6)...")
    print("=" * 60)

    generate_all_figures(mc_scenarios, pv_gen, opt, opt_result,
                         pareto_results, sensitivity_data, scheme_results,
                         calendar_ctx=calendar_ctx,
                         decision_result=decision_result,
                         scenario_results=scenario_results,
                         pareto_nsga2=pareto_nsga2)

    # v6 新增图表
    try:
        fig11_topology_radar(topo_names, topo_scores)
        print("  [OK] fig11_topology_radar")
    except Exception as e:
        print(f"  [SKIP] fig11_topology_radar: {e}")

    try:
        fig12_economic_breakdown(em, npc_val, lcoe_val, pbp_val, irr_val)
        print("  [OK] fig12_economic_breakdown")
    except Exception as e:
        print(f"  [SKIP] fig12_economic_breakdown: {e}")

    try:
        fig13_decision_cross_validation(decision_result, mdm)
        print("  [OK] fig13_decision_cross_validation")
    except Exception as e:
        print(f"  [SKIP] fig13_decision_cross_validation: {e}")

    # v6 增强输出
    print("\n" + "=" * 60)
    print("v6 Enhanced Output Summary")
    print("=" * 60)
    print(f"  PSO NPC (EconomicModel): {opt_result['npc']/1e4:.1f} wan yuan")
    print(f"  LCOE: {lcoe_val:.4f} yuan/kWh")
    print(f"  Payback: {pbp_val:.1f} yr  |  IRR: {irr_val:.1%}")
    print(f"  Topology: {topo_names[0]} recommended")
    print(f"  Decision Consensus: {schemes_list[best_idx]} (Borda: {borda[best_idx]:.0f})")
    print(f"  Dashboard: {dashboard_path}")
    if 'annual_carbon_revenue' in opt_result:
        print(f"  Annual Carbon Revenue: {opt_result['annual_carbon_revenue']:.0f} yuan")
    if 'subsidy' in opt_result:
        print(f"  Total Subsidy: {opt_result['subsidy']:.0f} yuan")

    print("\n" + "=" * 60)
    print("v6 Simulation Complete!")
    print("=" * 60)


def main(mode='full'):
    """主仿真流程

    Parameters
    ----------
    mode : str
        'fast' — 仅基础PSO (6步, 约2-5分钟)
        'nsga2' — PSO + NSGA-II多目标 (约5-15分钟)
        'full' — 全流程含决策框架 (约8-20分钟)
    """
    print("=" * 60)
    print("Highway Service Area PV-Storage-Charging Simulation v6")
    print(f"Mode: {mode}")
    print("=" * 60)

    if mode == 'v6':
        return main_v6()
    if mode == 'robust':
        return main_robust()

    # 创建统一日历上下文 (真实2025年日历 + Markov天气链)
    calendar_ctx = CalendarContext(seed=SEED)
    calendar_ctx.print_summary()

    # Step 1: Monte Carlo
    mc_n_runs = 5000 if mode != 'fast' else 2000
    mc_scenarios = run_monte_carlo(n_runs=mc_n_runs)

    # Step 2: PV (含温度效应)
    pv_gen, annual_pv, monthly_pv, pv_metrics = run_pv_synthesis(
        pv_capacity=500, calendar_ctx=calendar_ctx)

    # Step 3: PSO Optimization
    opt, opt_result, best_x = run_optimization(mc_scenarios, size='medium',
                                                calendar_ctx=calendar_ctx)

    # Step 3a: NSGA-II Multi-Objective (only for nsga2 / full modes)
    pareto_nsga2 = None
    nsga2_best = None
    if mode in ('nsga2', 'full'):
        pareto_nsga2, nsga2_best = run_nsga2_optimization(
            opt, pop_size=60 if mode == 'nsga2' else 40,
            n_gen=30 if mode == 'nsga2' else 20)

    # Step 4: Pareto (PSO-based)
    pareto_results = run_pareto_analysis(opt, n_samples=80 if mode != 'fast' else 40)

    # Step 5: Sensitivity
    sensitivity_data = run_sensitivity(opt)

    # Step 6: Scheme comparison
    scheme_results = run_scheme_comparison(opt)

    # Step 6a: AHP-TOPSIS Decision Framework
    decision_result = None
    scenario_results = None
    if mode == 'full':
        decision_result = run_decision_framework(opt)
        # Step 6b: Multi-scenario analysis
        scenario_results = run_multi_scenario(opt,
            opt_result['pv_capacity'],
            opt_result['ess_capacity'],
            opt_result['ess_power'])

    # Save results
    save_results(mc_scenarios, pv_metrics, opt_result, pareto_results)

    # Generate all figures
    generate_all_figures(mc_scenarios, pv_gen, opt, opt_result,
                         pareto_results, sensitivity_data, scheme_results,
                         calendar_ctx=calendar_ctx,
                         decision_result=decision_result,
                         scenario_results=scenario_results,
                         pareto_nsga2=pareto_nsga2)

    # v5 增强输出 (fast/nsga2/full 模式)
    print("\n" + "=" * 60)
    print("v5 Enhanced Economic Metrics")
    print("=" * 60)
    if 'annual_carbon_revenue' in opt_result:
        print(f"  Annual Carbon Revenue: {opt_result['annual_carbon_revenue']:.0f} yuan")
    if 'subsidy' in opt_result:
        print(f"  Total Subsidy: {opt_result['subsidy']:.0f} yuan")
    if 'net_capital_cost' in opt_result:
        print(f"  Net Capital Cost: {opt_result['net_capital_cost']/1e4:.1f} wan yuan")
    print(f"  Carbon Price: {70:.0f} yuan/tCO2 (CCER 2024-2025 avg)")

    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Highway Service Area PV-Storage-Charging Simulation v6')
    parser.add_argument('--mode', type=str, default='fast',
                       choices=['fast', 'nsga2', 'full', 'v6', 'robust'],
                       help='Simulation mode: fast (PSO only), '
                            'nsga2 (PSO + NSGA-II), full (all modules), '
                            'v6 (full pipeline + new modules), '
                            'robust (two-stage robust optimization)')
    args = parser.parse_args()
    main(mode=args.mode)
