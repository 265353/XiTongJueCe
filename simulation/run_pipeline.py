"""
一键仿真全流程 — 输出所有 JSON 结果供前端渲染
运行: python run_pipeline.py
"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from calendar_utils import CalendarContext
from mc_charging_load import MonteCarloChargingSimulator
from capacity_optimization import MicrogridOptimizer
from pv_generation import PVGenerator
from decision_framework import DecisionFramework
from config import PV_AREA_RATIO, load_pvgis_tmy

SEED = 42
MODE = 'medium'
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 55)
print("Highway Service Area Simulation — Full Pipeline")
print("=" * 55)

# ============================================================
# [1] MC Charging Load (v6.6: physics model)
# ============================================================
print("\n[1/6] Monte Carlo EV Charging Load (n=1000, physics model)...")
sim = MonteCarloChargingSimulator(service_area_size=MODE, seed=SEED, year=2025)
mc_scenarios = sim.simulate_all_scenarios(n_runs=1000)

mc_summary = {}
for dt, res_dict in mc_scenarios.items():
    mc_summary[dt] = {
        'peak_mean': float(res_dict['peak_mean']),
        'peak_std': float(res_dict['peak_std']),
        'peak_p95': float(res_dict['p95'].max()),
        'daily_energy_mean': float(res_dict['daily_energy_mean']),
        'hourly_p50': res_dict['p50'].tolist(),
        'hourly_p95': res_dict['p95'].tolist(),
    }
with open(os.path.join(RESULTS_DIR, 'mc_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(mc_summary, f, indent=2, ensure_ascii=False)
print(f"  OK: {len(mc_scenarios)} day types, "
      f"workday peak={mc_summary['workday']['peak_mean']:.0f}kW")

# ============================================================
# [2] PV Generation (v6.6: PVGIS真实TMY)
# ============================================================
print("\n[2/6] PV Generation Synthesis...")
calendar_ctx = CalendarContext(seed=SEED)
# 加载PVGIS实测TMY
tmy = load_pvgis_tmy('wuhan')
print(f"  TMY source: {tmy['source']}")
pv_gen = PVGenerator(pv_capacity_kwp=500, seed=SEED, calendar_ctx=calendar_ctx, tmy_data=tmy)
annual = pv_gen.generate_annual()
pv_metrics = pv_gen.compute_annual_metrics(annual)
monthly_pv = pv_gen.get_monthly_generation(annual)
print(f"  OK: annual={pv_metrics['annual_generation_kwh']:.0f}kWh, "
      f"eq_hours={pv_metrics['equivalent_hours']:.1f}h")

# ============================================================
# [3] PSO Capacity Optimization
# ============================================================
print("\n[3/6] PSO Capacity Optimization (pop=40, iter=30)...")
opt = MicrogridOptimizer(size=MODE, mc_scenarios=mc_scenarios,
                          seed=SEED, calendar_ctx=calendar_ctx, tmy_data=tmy,
                          use_abm=True)
opt_result, best_x = opt.optimize_pso(pop_size=40, max_iter=30, verbose=False)
print(f"  Optimal: PV={opt_result['pv_capacity']:.0f}kWp, "
      f"ESS={opt_result['ess_capacity']:.0f}kWh, "
      f"ESS_P={opt_result['ess_power']:.0f}kW")
print(f"  NPC={opt_result['npc']/1e4:.1f}万元, "
      f"SSR={opt_result['self_sufficiency']:.1%}, "
      f"Carbon={opt_result['carbon_reduction_t']:.1f}t/yr")

# 8760h verification
print("  Running 8760h verification...")
verify = opt.verify_8760h(opt_result['pv_capacity'],
                           opt_result['ess_capacity'],
                           opt_result['ess_power'])
opt_result['verify_8760h'] = verify
print(f"  Verify: SSR_8760={verify['self_sufficiency']:.1%}, "
      f"LOLP={verify['loss_of_load_pct']:.3%}, "
      f"ESS_cycles={verify['ess_cycles']:.0f}")

opt_json = {}
for k, v in opt_result.items():
    if isinstance(v, (np.floating, np.integer)):
        opt_json[k] = float(v)
    elif isinstance(v, (int, float)):
        opt_json[k] = v
    elif isinstance(v, dict):
        opt_json[k] = {
            sk: (float(sv) if isinstance(sv, (np.floating, np.integer)) else sv)
            for sk, sv in v.items()
        }
    elif hasattr(v, 'tolist'):
        opt_json[k] = v.tolist()
with open(os.path.join(RESULTS_DIR, 'optimization_result.json'), 'w', encoding='utf-8') as f:
    json.dump(opt_json, f, indent=2, ensure_ascii=False)

# ============================================================
# [4] Pareto Frontier
# ============================================================
print("\n[4/6] Pareto Frontier Analysis (n=200)...")
rng = np.random.RandomState(SEED)
max_pv = opt.area_config['pv_area_m2'] / PV_AREA_RATIO
results = []
for i in range(200):
    pv = rng.uniform(50, max_pv * 0.95)
    ess_e = rng.uniform(0, 3000)
    ess_p = rng.uniform(0, min(ess_e * 0.5, 800))
    res = opt.evaluate_config(pv, ess_e if ess_e > 20 else 0,
                               ess_p if ess_e > 20 else 0)
    results.append(res)

pareto_front = []
for i, ri in enumerate(results):
    dominated = False
    for j, rj in enumerate(results):
        if i == j:
            continue
        if rj['npc'] <= ri['npc'] and rj['self_sufficiency'] >= ri['self_sufficiency']:
            if rj['npc'] < ri['npc'] or rj['self_sufficiency'] > ri['self_sufficiency']:
                dominated = True
                break
    if not dominated:
        pareto_front.append(ri)

pareto_front.sort(key=lambda x: x['self_sufficiency'])
pareto_json = [
    {'pv': float(r['pv_capacity']),
     'ess': float(r['ess_capacity']),
     'npc_wan': float(r['npc'] / 1e4),
     'ssr': float(r['self_sufficiency'])}
    for r in pareto_front[::2]
]
with open(os.path.join(RESULTS_DIR, 'pareto_results.json'), 'w', encoding='utf-8') as f:
    json.dump(pareto_json, f, indent=2, ensure_ascii=False)
print(f"  OK: {len(pareto_front)} Pareto-optimal solutions")

# ============================================================
# [5] Decision Framework
# ============================================================
print("\n[5/6] AHP-TOPSIS Decision Framework...")
s1 = opt.evaluate_config(0, 0, 0)
s2 = opt.evaluate_config(max_pv * 0.8, 0, 0)
s3 = opt_result
s4 = opt.evaluate_config(max_pv * 0.95, 2500, 1200)

schemes = ['方案A:基础型', '方案B:均衡型', '方案C:激进型', '方案D:离网型']
indicators = ['能源自洽率', 'NPC(万元)', '投资回收期', '供电可靠率', '年碳减排']
directions = ['benefit', 'cost', 'cost', 'benefit', 'benefit']

vals = np.array([
    [s1['self_sufficiency'] * 100, s1['npc'] / 1e4, s1['payback_years'], 99.5, s1['carbon_reduction_t']],
    [s2['self_sufficiency'] * 100, s2['npc'] / 1e4, s2['payback_years'], 99.5, s2['carbon_reduction_t']],
    [s3['self_sufficiency'] * 100, s3['npc'] / 1e4, s3['payback_years'], 99.8, s3['carbon_reduction_t']],
    [s4['self_sufficiency'] * 100, s4['npc'] / 1e4, s4['payback_years'], 99.0, s4['carbon_reduction_t']],
])

df = DecisionFramework()
dec_result = df.evaluate(
    schemes=schemes, indicators=indicators, values=vals,
    directions=directions,
    criteria_labels=['能源性', '经济性', '可靠性', '环境性'],
    indicator_criteria_map=[0, 1, 1, 2, 3])
dec_result.save()
print(f"  Best: {dec_result.get_best_scheme()}")
ranking = dec_result.get_ranking()
if ranking:
    print(f"  Ranking: {ranking[:3]}")

# ============================================================
# [6] 8760h Daily Data
# ============================================================
print("\n[6/6] Building 8760h daily dispatch data...")
opt.pv_capacity = opt_result['pv_capacity']
opt.ess_capacity = opt_result['ess_capacity']
opt.ess_power = opt_result['ess_power']
pv_coeff_seq, load_seq, tou_seq, seasons_seq, temp_seq = opt._build_8760h_sequence()

daily_list = []
for d in range(365):
    season = seasons_seq[d * 24]
    pv_profile = pv_coeff_seq[d * 24:(d + 1) * 24] * opt_result['pv_capacity']
    load_profile = load_seq[d * 24:(d + 1) * 24]
    tou_hourly = tou_seq[d * 24:(d + 1) * 24]
    day_res = opt.simulate_daily_operation(pv_profile, load_profile, season,
                                            tou_prices=tou_hourly)
    pv_day = float(pv_profile.sum())
    load_day = float(load_profile.sum())
    grid_exp = float(day_res['grid_export'].sum())
    daily_list.append({
        'day': d,
        'month': calendar_ctx.month_of_day[d],
        'season': season,
        'day_type': calendar_ctx.day_types[d],
        'weather': calendar_ctx.weather_seq[d],
        'pv_gen': round(pv_day, 1),
        'load': round(load_day, 1),
        'grid_import': round(float(day_res['grid_import'].sum()), 1),
        'grid_export': round(grid_exp, 1),
        'grid_cost': round(float((day_res['grid_import'] * tou_hourly).sum()), 1),
        'soc_end': round(float(day_res['soc_curve'][-1]), 3),
        'ssr': round((pv_day - grid_exp) / max(load_day, 1), 4),
    })

with open(os.path.join(RESULTS_DIR, 'daily_8760h.json'), 'w', encoding='utf-8') as f:
    json.dump(daily_list, f, indent=1, ensure_ascii=False)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 55)
print("FULL PIPELINE COMPLETE")
print("=" * 55)
print(f"  PV capacity:       {opt_result['pv_capacity']:.0f} kWp")
print(f"  ESS capacity:      {opt_result['ess_capacity']:.0f} kWh")
print(f"  ESS power:         {opt_result['ess_power']:.0f} kW")
print(f"  NPC (20yr):        {opt_result['npc']/1e4:.1f} 万元")
print(f"  Self-sufficiency:  {opt_result['self_sufficiency']:.1%}")
print(f"  Carbon reduction:  {opt_result['carbon_reduction_t']:.1f} tCO2/yr")
print(f"  Payback:           {opt_result['payback_years']:.1f} years")
print(f"  Best scheme:       {dec_result.get_best_scheme()}")
print(f"  Results saved to:  {os.path.abspath(RESULTS_DIR)}")
