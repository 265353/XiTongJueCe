"""
接续仿真 — 步4-6 (基于已完成的步1-3结果)
"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from calendar_utils import CalendarContext
from mc_charging_load import MonteCarloChargingSimulator
from capacity_optimization import MicrogridOptimizer
from decision_framework import DecisionFramework
from config import PV_AREA_RATIO

SEED = 42
MODE = 'medium'
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# 加载已完成的步1-3结果
with open(os.path.join(RESULTS_DIR, 'optimization_result.json'), 'r', encoding='utf-8') as f:
    opt_json = json.load(f)

print("=" * 55)
print("Continuation Pipeline — Steps 4-6")
print("=" * 55)

# 重建 MC 和 Calendar (轻量)
print("\n[Setup] Rebuilding simulation context...")
calendar_ctx = CalendarContext(seed=SEED)
mc_sim = MonteCarloChargingSimulator(service_area_size=MODE, seed=SEED)
mc_scenarios = mc_sim.simulate_all_scenarios(n_runs=1000)  # 快速模式

opt = MicrogridOptimizer(size=MODE, mc_scenarios=mc_scenarios, seed=SEED,
                          calendar_ctx=calendar_ctx)
opt.pv_capacity = opt_json['pv_capacity']
opt.ess_capacity = opt_json['ess_capacity']
opt.ess_power = opt_json['ess_power']

# ============================================================
# [4] Pareto Frontier
# ============================================================
print("\n[4/6] Pareto Frontier Analysis (n=200)...")
rng = np.random.RandomState(SEED)
max_pv = opt.area_config['pv_area_m2'] / PV_AREA_RATIO
all_results = []
for i in range(200):
    pv = rng.uniform(50, max_pv * 0.95)
    ess_e = rng.uniform(0, 3000)
    ess_p = rng.uniform(0, min(ess_e * 0.5, 800))
    try:
        res = opt.evaluate_config(pv, ess_e if ess_e > 20 else 0,
                                   ess_p if ess_e > 20 else 0)
        all_results.append(res)
    except Exception:
        pass

pareto_front = []
for i, ri in enumerate(all_results):
    dominated = False
    for j, rj in enumerate(all_results):
        if i == j: continue
        if rj['npc'] <= ri['npc'] and rj['self_sufficiency'] >= ri['self_sufficiency']:
            if rj['npc'] < ri['npc'] or rj['self_sufficiency'] > ri['self_sufficiency']:
                dominated = True; break
    if not dominated:
        pareto_front.append(ri)

pareto_front.sort(key=lambda x: x['self_sufficiency'])
pareto_json = [{
    'pv': float(r['pv_capacity']),
    'ess': float(r['ess_capacity']),
    'npc_wan': float(r['npc'] / 1e4),
    'ssr': float(r['self_sufficiency']),
} for r in pareto_front[::2]]
with open(os.path.join(RESULTS_DIR, 'pareto_results.json'), 'w', encoding='utf-8') as f:
    json.dump(pareto_json, f, indent=2, ensure_ascii=False)
print(f"  OK: {len(pareto_front)} Pareto-optimal, saved {len(pareto_json)}")

# ============================================================
# [5] Decision Framework
# ============================================================
print("\n[5/6] AHP-TOPSIS Decision Framework...")
s1 = opt.evaluate_config(0, 0, 0)
s2 = opt.evaluate_config(max_pv * 0.8, 0, 0)
s3_dict = {'pv_capacity': opt_json['pv_capacity'],
           'ess_capacity': opt_json['ess_capacity'],
           'ess_power': opt_json['ess_power'],
           'npc': opt_json['npc'],
           'self_sufficiency': opt_json['self_sufficiency'],
           'carbon_reduction_t': opt_json.get('carbon_reduction_t', 687),
           'payback_years': opt_json.get('payback_years', 12),
           'annual_grid_export_kwh': opt_json.get('annual_grid_export_kwh', 0),
           'annual_pv_gen_kwh': opt_json.get('annual_pv_gen_kwh', 1200000)}
s4 = opt.evaluate_config(max_pv * 0.95, 2500, 1200)

schemes = ['方案A:基础型', '方案B:均衡型', '方案C:激进型', '方案D:离网型']
indicators = ['能源自洽率', 'NPC(万元)', '投资回收期', '供电可靠率', '年碳减排']
directions = ['benefit', 'cost', 'cost', 'benefit', 'benefit']

vals = np.array([
    [s1['self_sufficiency'] * 100, s1['npc'] / 1e4, s1['payback_years'], 99.5, s1['carbon_reduction_t']],
    [s2['self_sufficiency'] * 100, s2['npc'] / 1e4, s2['payback_years'], 99.5, s2['carbon_reduction_t']],
    [s3_dict['self_sufficiency'] * 100, s3_dict['npc'] / 1e4, s3_dict['payback_years'], 99.8, s3_dict.get('carbon_reduction_t', 687)],
    [s4['self_sufficiency'] * 100, s4['npc'] / 1e4, s4['payback_years'], 99.0, s4['carbon_reduction_t']],
])

criteria = ['能源性', '经济性', '可靠性', '环境性']
df = DecisionFramework()
dec_result = df.evaluate(
    schemes=schemes, indicators=indicators, values=vals,
    directions=directions, criteria_labels=criteria,
    indicator_criteria_map=[0, 1, 1, 2, 3])
dec_result.save()
print(f"  Best: {dec_result.get_best_scheme()}")

# ============================================================
# [6] 8760h Daily Data
# ============================================================
print("\n[6/6] Building 8760h daily dispatch data...")
pv_coeff_seq, load_seq, tou_seq, seasons_seq, _ = opt._build_8760h_sequence()

daily_list = []
for d in range(365):
    season = seasons_seq[d * 24]
    pv_profile = pv_coeff_seq[d * 24:(d + 1) * 24] * opt_json['pv_capacity']
    load_profile = load_seq[d * 24:(d + 1) * 24]
    tou_hourly = tou_seq[d * 24:(d + 1) * 24]
    day_res = opt.simulate_daily_operation(pv_profile, load_profile, season,
                                            tou_prices=tou_hourly)
    pv_day = float(pv_profile.sum())
    load_day = float(load_profile.sum())
    grid_exp = float(day_res['grid_export'].sum())
    daily_list.append({
        'day': d, 'month': calendar_ctx.month_of_day[d], 'season': season,
        'day_type': calendar_ctx.day_types[d], 'weather': calendar_ctx.weather_seq[d],
        'pv_gen': round(pv_day, 1), 'load': round(load_day, 1),
        'grid_import': round(float(day_res['grid_import'].sum()), 1),
        'grid_export': round(grid_exp, 1),
        'grid_cost': round(float((day_res['grid_import'] * tou_hourly).sum()), 1),
        'soc_end': round(float(day_res['soc_curve'][-1]), 3),
        'ssr': round((pv_day - grid_exp) / max(load_day, 1), 4),
    })
    if (d + 1) % 100 == 0:
        print(f"  ... {d + 1}/365 days")

with open(os.path.join(RESULTS_DIR, 'daily_8760h.json'), 'w', encoding='utf-8') as f:
    json.dump(daily_list, f, indent=1, ensure_ascii=False)
print(f"  OK: 365 days saved")

print("\n" + "=" * 55)
print("PIPELINE COMPLETE!")
print("=" * 55)
print(f"  Pareto: {len(pareto_front)} solutions in pareto_results.json")
print(f"  Decision: {dec_result.get_best_scheme()} in decision_result.json")
print(f"  Daily: 365 days in daily_8760h.json")
