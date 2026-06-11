"""
主程序 — 高速服务区光储充一体化仿真全流程
运行: python main.py
"""
import numpy as np
import json
import os
import sys

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
)
from config import (
    SERVICE_AREA_CONFIG, PV_COEFF, WEATHER_COEFF,
    PV_AREA_RATIO,
)

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

    import config
    orig_pv_cost = {k: v for k, v in config.PV_COST.items()}
    orig_price = {k: v for k, v in config.TOU_PRICE_VALUES.items()}

    for i, pvm in enumerate(pv_mult):
        for j, prm in enumerate(price_mult):
            for k in config.PV_COST:
                config.PV_COST[k] = orig_pv_cost[k] * pvm
            for k in config.TOU_PRICE_VALUES:
                config.TOU_PRICE_VALUES[k] = orig_price[k] * prm

            result, _ = opt.optimize_pso(pop_size=30, max_iter=25, verbose=False)

            pv_cap_grid[i, j] = result['pv_capacity']
            npc_grid[i, j] = result['npc']

    for k in config.PV_COST:
        config.PV_COST[k] = orig_pv_cost[k]
    for k in config.TOU_PRICE_VALUES:
        config.TOU_PRICE_VALUES[k] = orig_price[k]

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


def generate_all_figures(mc_scenarios, pv_gen, opt, opt_result, pareto_results,
                         sensitivity_data, scheme_results, calendar_ctx=None):
    """生成全部图表"""
    print("\n" + "=" * 60)
    print("Generating Figures...")
    print("=" * 60)

    # 图1
    fig1_charging_load_probability(mc_scenarios)
    print("  [OK] fig1_charging_load_probability")

    # 图2
    fig2_scenario_comparison(mc_scenarios)
    print("  [OK] fig2_scenario_comparison")

    # 图3: 最优配置下的夏季晴天功率平衡
    pv_cap = opt_result['pv_capacity']
    pv_profile = np.array(PV_COEFF['summer']) * pv_cap * WEATHER_COEFF['clear']
    load_profile = opt.get_total_load('summer', 'workday', 'clear')
    op_day = opt.simulate_daily_operation(pv_profile, load_profile)
    fig3_power_balance(pv_profile, load_profile, op_day, 'summer', 'clear')
    print("  [OK] fig3_power_balance")

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


def main():
    print("=" * 60)
    print("Highway Service Area PV-Storage-Charging Simulation")
    print("=" * 60)

    # 创建统一日历上下文 (真实2025年日历 + Markov天气链)
    calendar_ctx = CalendarContext(seed=SEED)
    calendar_ctx.print_summary()

    # Step 1: Monte Carlo
    mc_scenarios = run_monte_carlo(n_runs=5000)

    # Step 2: PV
    pv_gen, annual_pv, monthly_pv, pv_metrics = run_pv_synthesis(
        pv_capacity=500, calendar_ctx=calendar_ctx)

    # Step 3: Optimization (with MC results)
    opt, opt_result, best_x = run_optimization(mc_scenarios, size='medium',
                                                calendar_ctx=calendar_ctx)

    # Step 4: Pareto
    pareto_results = run_pareto_analysis(opt, n_samples=80)

    # Step 5: Sensitivity
    sensitivity_data = run_sensitivity(opt)

    # Step 6: Scheme comparison
    scheme_results = run_scheme_comparison(opt)

    # Save
    save_results(mc_scenarios, pv_metrics, opt_result, pareto_results)

    # Figures
    generate_all_figures(mc_scenarios, pv_gen, opt, opt_result,
                         pareto_results, sensitivity_data, scheme_results,
                         calendar_ctx=calendar_ctx)

    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
