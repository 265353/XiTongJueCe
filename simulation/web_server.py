"""
高速服务区光储充一体化微网 — Web 服务器 (FastAPI)
提供 REST API 接口 + 静态前端服务
启动: python web_server.py [--port 8760]
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from config import (
    SERVICE_AREA_CONFIG, TOU_PRICE_VALUES, CARBON_FACTOR_GRID,
    SOC_MIN, SOC_MAX, PROJECT_LIFE, PV_COEFF, WEATHER_COEFF,
    BUILDING_LOAD, get_season, get_tou_price_array,
    FEED_IN_PRICE, DAY_TYPE_COEFF, WEATHER_CHARGING_COEFF,
    WEATHER_BUILDING_COEFF, HOLIDAY_TOU_FACTOR, STATION_AUX_DAILY_KWH,
)
from calendar_utils import CalendarContext
from mc_charging_load import MonteCarloChargingSimulator
from capacity_optimization import MicrogridOptimizer

app = FastAPI(title="高速服务区光储充微网仿真系统", version="6.5")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
SEED = 42

# 全局缓存: 启动时加载已有结果
_cache = {}


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


@app.on_event("startup")
def startup():
    _cache['mc_summary'] = load_json('mc_summary.json')
    _cache['optimization'] = load_json('optimization_result.json')
    _cache['pareto'] = load_json('pareto_results.json')
    _cache['decision'] = load_json('decision_result.json')


# ============================================================
# API: 仿真结果
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": "6.5",
        "data_available": {
            "mc_summary": _cache['mc_summary'] is not None,
            "optimization": _cache['optimization'] is not None,
            "pareto": _cache['pareto'] is not None,
            "decision": _cache['decision'] is not None,
        }
    }


@app.get("/api/results/mc-summary")
def get_mc_summary():
    """MC充电负荷仿真结果"""
    return JSONResponse(_cache.get('mc_summary', {}))


@app.get("/api/results/optimization")
def get_optimization():
    """PSO容量优化结果"""
    return JSONResponse(_cache.get('optimization', {}))


@app.get("/api/results/pareto")
def get_pareto():
    """Pareto前沿结果"""
    return JSONResponse(_cache.get('pareto', []))


@app.get("/api/results/decision")
def get_decision():
    """AHP-TOPSIS决策结果"""
    return JSONResponse(_cache.get('decision', {}))


# ============================================================
# API: 系统配置
# ============================================================

@app.get("/api/config")
def get_config():
    """返回关键配置参数"""
    medium = SERVICE_AREA_CONFIG['medium']
    return {
        "service_area": {
            "name": medium['name'],
            "daily_charge_kwh": medium['daily_charge_kwh'],
            "n_piles_120kw": medium['n_piles_120kw'],
            "n_piles_480kw": medium['n_piles_480kw'],
            "building_area_m2": medium['building_area_m2'],
            "pv_area_m2": medium['pv_area_m2'],
            "peak_building_kw": medium['peak_building_kw'],
        },
        "tou_prices": {
            season: {
                "peak": v.get('peak', 0),
                "flat": v.get('flat', 0),
                "valley": v.get('valley', 0),
            }
            for season, v in TOU_PRICE_VALUES.items()
        },
        "feed_in_price": FEED_IN_PRICE,
        "carbon_factor_grid": CARBON_FACTOR_GRID,
        "soc_range": [SOC_MIN, SOC_MAX],
        "project_life": PROJECT_LIFE,
        "device_specs": {
            "pv_modules": 1986, "pv_inverters": 5, "pv_inverter_kw": 225,
            "ess_cabinets": 10, "ess_cabinet_kwh": 215,
            "ev_120kw": 16, "ev_480kw": 2,
            "transformer_kva": 2500, "svg_kvar": 50,
            "building_peak_kw": 147,
        },
    }


# ============================================================
# API: 实时仿真 (轻量)
# ============================================================

@app.get("/api/simulate/dispatch-day")
def simulate_dispatch_day(
    pv_cap: float = Query(1231, description="光伏容量 kWp"),
    ess_cap: float = Query(2000, description="储能容量 kWh"),
    ess_pow: float = Query(1000, description="储能力率 kW"),
    day: int = Query(200, description="一年中的第几天 (0-364)"),
):
    """仿真单日调度, 返回24h逐时数据"""
    calendar_ctx = CalendarContext(seed=SEED)
    mc_sim = MonteCarloChargingSimulator(service_area_size='medium', seed=SEED)
    mc_scenarios = mc_sim.simulate_all_scenarios(n_runs=500)

    opt = MicrogridOptimizer(size='medium', mc_scenarios=mc_scenarios, seed=SEED,
                              calendar_ctx=calendar_ctx)
    opt.pv_capacity = pv_cap
    opt.ess_capacity = ess_cap
    opt.ess_power = ess_pow

    pv_coeff_seq, load_seq, tou_seq, seasons_seq = opt._build_8760h_sequence()
    season = seasons_seq[day * 24]
    pv_profile = pv_coeff_seq[day * 24:(day + 1) * 24] * pv_cap
    load_profile = load_seq[day * 24:(day + 1) * 24]
    tou_hourly = tou_seq[day * 24:(day + 1) * 24]

    result = opt.simulate_daily_operation(pv_profile, load_profile, season, tou_prices=tou_hourly)

    hourly = []
    for h in range(24):
        hourly.append({
            'h': h,
            'pv': round(float(pv_profile[h]), 1),
            'load': round(float(load_profile[h]), 1),
            'grid_import': round(float(result['grid_import'][h]), 1),
            'grid_export': round(float(result['grid_export'][h]), 1),
            'ess_ch': round(float(result['ess_charge'][h]), 1),
            'ess_disch': round(float(result['ess_discharge'][h]), 1),
            'soc': round(float(result['soc_curve'][h]), 3),
            'tou': round(float(tou_hourly[h]), 3),
            'net': round(float(pv_profile[h]) - float(load_profile[h]), 1),
        })

    total_pv = float(pv_profile.sum())
    total_load = float(load_profile.sum())
    self_use = total_pv - float(result['grid_export'].sum())

    return {
        'config': {'pv_cap': pv_cap, 'ess_cap': ess_cap, 'ess_pow': ess_pow},
        'day_info': {
            'day': day,
            'month': calendar_ctx.month_of_day[day],
            'day_type': calendar_ctx.day_types[day],
            'weather': calendar_ctx.weather_seq[day],
            'season': season,
        },
        'hourly': hourly,
        'summary': {
            'total_pv': round(total_pv, 1),
            'total_load': round(total_load, 1),
            'total_grid_import': round(float(result['grid_import'].sum()), 1),
            'total_grid_export': round(float(result['grid_export'].sum()), 1),
            'self_sufficiency': round(self_use / max(total_load, 1), 4),
            'grid_cost': round(float((result['grid_import'] * tou_hourly).sum()), 1),
            'carbon_reduction': round(self_use / 1000 * CARBON_FACTOR_GRID, 2),
            'ess_cycles': round(float(result['ess_discharge'].sum()) / max(ess_cap, 1), 2),
        },
    }


@app.get("/api/simulate/dispatch-year-summary")
def simulate_year_summary(
    pv_cap: float = Query(1231),
    ess_cap: float = Query(2000),
    ess_pow: float = Query(1000),
):
    """仿真全年365天调度汇总"""
    calendar_ctx = CalendarContext(seed=SEED)
    mc_sim = MonteCarloChargingSimulator(service_area_size='medium', seed=SEED)
    mc_scenarios = mc_sim.simulate_all_scenarios(n_runs=1000)

    opt = MicrogridOptimizer(size='medium', mc_scenarios=mc_scenarios, seed=SEED,
                              calendar_ctx=calendar_ctx)
    opt.pv_capacity = pv_cap
    opt.ess_capacity = ess_cap
    opt.ess_power = ess_pow

    pv_coeff_seq, load_seq, tou_seq, seasons_seq = opt._build_8760h_sequence()

    daily_results = []
    ts = {'pv_gen': 0, 'load': 0, 'grid_import': 0, 'grid_export': 0, 'grid_cost': 0}
    monthly_pv = np.zeros(12)
    monthly_load = np.zeros(12)
    monthly_grid = np.zeros(12)
    monthly_ssr = np.zeros(12)
    monthly_days = np.zeros(12)

    for d in range(365):
        season = seasons_seq[d * 24]
        pv_profile = pv_coeff_seq[d * 24:(d + 1) * 24] * pv_cap
        load_profile = load_seq[d * 24:(d + 1) * 24]
        tou_hourly = tou_seq[d * 24:(d + 1) * 24]
        result = opt.simulate_daily_operation(pv_profile, load_profile, season, tou_prices=tou_hourly)

        pv_day = float(pv_profile.sum())
        load_day = float(load_profile.sum())
        grid_imp = float(result['grid_import'].sum())
        grid_exp = float(result['grid_export'].sum())
        cost = float((result['grid_import'] * tou_hourly).sum())
        self_use = pv_day - grid_exp

        daily_results.append({
            'day': d,
            'month': calendar_ctx.month_of_day[d],
            'season': season,
            'day_type': calendar_ctx.day_types[d],
            'weather': calendar_ctx.weather_seq[d],
            'pv_gen': round(pv_day, 1),
            'load': round(load_day, 1),
            'grid_import': round(grid_imp, 1),
            'grid_export': round(grid_exp, 1),
            'grid_cost': round(cost, 1),
            'soc_end': round(float(result['soc_curve'][-1]), 3),
            'ssr': round(self_use / max(load_day, 1), 4),
        })

        ts['pv_gen'] += pv_day
        ts['load'] += load_day
        ts['grid_import'] += grid_imp
        ts['grid_export'] += grid_exp
        ts['grid_cost'] += cost

        m = calendar_ctx.month_of_day[d] - 1
        monthly_pv[m] += pv_day
        monthly_load[m] += load_day
        monthly_grid[m] += grid_imp
        monthly_days[m] += 1
        monthly_ssr[m] += self_use / max(load_day, 1)

    for m in range(12):
        if monthly_days[m] > 0:
            monthly_ssr[m] /= monthly_days[m]

    return {
        'config': {'pv_cap': pv_cap, 'ess_cap': ess_cap, 'ess_pow': ess_pow},
        'totals': {k: round(v, 1) for k, v in ts.items()},
        'self_sufficiency': round(
            (ts['pv_gen'] - ts['grid_export']) / max(ts['load'], 1), 4),
        'monthly': {
            'pv': monthly_pv.tolist(),
            'load': monthly_load.tolist(),
            'grid': monthly_grid.tolist(),
            'ssr': [round(float(v), 4) for v in monthly_ssr],
        },
        'daily': daily_results[::7],  # 每周采样
    }


@app.get("/api/simulate/mc-distribution")
def simulate_mc_distribution(
    day_type: str = Query('workday', description="日类型: workday/weekend/holiday/spring_festival"),
    weather: str = Query('clear', description="天气: clear/partly_cloudy/cloudy/overcast/rain"),
    month: int = Query(6, description="月份 1-12"),
    n_runs: int = Query(1000, description="MC仿真次数"),
):
    """MC充电负荷分布仿真"""
    sim = MonteCarloChargingSimulator(service_area_size='medium', seed=SEED)
    result = sim.simulate_scenario(day_type, weather, month, n_runs=n_runs)

    return {
        'params': {'day_type': day_type, 'weather': weather, 'month': month},
        'hourly_p50': result['p50'].tolist(),
        'hourly_p95': result['p95'].tolist(),
        'hourly_p5': result['p5'].tolist() if 'p5' in result else [0]*24,
        'hourly_p25': result['p25'].tolist() if 'p25' in result else [0]*24,
        'hourly_p75': result['p75'].tolist() if 'p75' in result else [0]*24,
        'peak_mean': float(result['peak_mean']),
        'peak_std': float(result['peak_std']),
        'daily_energy_mean': float(result['daily_energy_mean']),
    }


@app.get("/api/simulate/charging-station-live")
def simulate_charging_station_live(
    hour: int = Query(12, ge=0, le=23, description="仿真小时 (0-23)"),
    day_type: str = Query('workday'),
    month: int = Query(6, ge=1, le=12),
    update_ms: int = Query(5000, description="更新间隔 ms"),
):
    """生成充电站实景数据 — 模拟20个充电终端的逐分钟车辆到达/充电/离开事件

    返回:
        - 当前小时逐分钟 (60个时间步) 每个终端的充电事件
        - 每辆车的 SOC / 充电功率 / 剩余时间
        - 总功率曲线
    """
    import random as _random
    _rng = _random.Random(SEED + hour * 100 + month)

    # 获取该小时的基础到达率
    arrival_rates = {
        'workday':  [1,0,0,0,0,0,2,4,8,13,19,21,19,18,20,21,19,16,14,12,8,6,4,2],
        'weekend':  [1,1,0,0,0,1,3,7,12,18,24,25,24,21,24,25,22,19,17,14,10,7,5,3],
        'holiday':  [1,1,0,0,0,1,4,9,15,22,28,30,27,25,28,30,27,23,20,17,12,9,6,4],
    }
    rates = arrival_rates.get(day_type, arrival_rates['workday'])
    base_rate = rates[min(hour, 23)]
    month_factor = {1:0.88,2:0.80,3:1.00,4:1.03,5:1.15,6:1.02,7:1.08,8:1.08,9:1.02,10:1.20,11:0.97,12:0.92}.get(month, 1.0)
    effective_rate = base_rate * month_factor

    # 充电桩配置
    n_120kw = 16
    n_480kw = 2  # 超充堆, 每个可同时服务2辆车
    total_piles = n_120kw + n_480kw

    pile_types = (
        [{'type': '120kW', 'power': 120, 'id': f'CP-{i+1:02d}', 'x': i} for i in range(n_120kw)]
        + [{'type': '480kW', 'power': 480, 'id': f'HP-{i+1:02d}', 'x': n_120kw + i} for i in range(n_480kw)]
    )

    # 车型
    car_types = [
        {'name': '微型', 'battery': 25, 'color': '#60d040', 'pct': 0.08},
        {'name': '小型', 'battery': 35, 'color': '#40c0f0', 'pct': 0.12},
        {'name': '紧凑型', 'battery': 50, 'color': '#f0a020', 'pct': 0.25},
        {'name': '中型', 'battery': 65, 'color': '#e8e8f0', 'pct': 0.25},
        {'name': 'SUV', 'battery': 85, 'color': '#f08030', 'pct': 0.18},
        {'name': '豪华', 'battery': 100, 'color': '#c090d0', 'pct': 0.07},
        {'name': '物流', 'battery': 80, 'color': '#8090a0', 'pct': 0.05},
    ]

    # 逐分钟仿真
    minutes = []
    active_sessions = []  # [{pile_idx, car_type, start_soc, target_soc, power, start_min, est_duration}]
    waiting_queue = []
    next_car_id = 1

    for minute in range(60):
        # 车辆到达 (Poisson过程)
        arrival_prob = effective_rate / 60.0
        n_arrivals = 0
        if _rng.random() < arrival_prob * 0.7:
            n_arrivals = 1
        if effective_rate > 12 and _rng.random() < arrival_prob * 0.3:
            n_arrivals = 2

        for _ in range(n_arrivals):
            # 选择车型
            r = _rng.random()
            cum = 0
            car_type = car_types[0]
            for ct in car_types:
                cum += ct['pct']
                if r <= cum:
                    car_type = ct
                    break

            start_soc = max(0.05, _rng.gauss(0.21, 0.08))
            target_soc = min(0.95, _rng.gauss(0.85, 0.06))
            needed_kwh = car_type['battery'] * (target_soc - start_soc)
            needed_kwh = max(5, needed_kwh)

            car = {
                'id': next_car_id,
                'type': car_type['name'],
                'battery_kwh': car_type['battery'],
                'color': car_type['color'],
                'start_soc': round(start_soc, 2),
                'target_soc': round(target_soc, 2),
                'needed_kwh': round(needed_kwh, 1),
                'arrive_min': minute,
            }
            next_car_id += 1

            # 找空闲桩
            used_piles = {s['pile_idx'] for s in active_sessions}
            free_120 = [i for i in range(n_120kw) if i not in used_piles]
            free_480 = [i for i in range(n_120kw, total_piles) if i not in used_piles]

            if free_120:
                pile_idx = _rng.choice(free_120)
            elif free_480:
                pile_idx = _rng.choice(free_480)
            else:
                waiting_queue.append(car)
                continue

            charge_power = min(pile_types[pile_idx]['power'], car_type['battery'] * 1.5)
            duration_min = needed_kwh / charge_power * 60

            active_sessions.append({
                'pile_idx': pile_idx,
                'car': car,
                'charge_power': round(charge_power, 1),
                'start_min': minute,
                'est_duration': round(duration_min, 0),
                'soc_history': [round(start_soc, 2)],
            })

        # 更新充电进度
        active_now = []
        for s in active_sessions:
            elapsed = minute - s['start_min']
            progress = min(1.0, elapsed / max(s['est_duration'], 1))
            current_soc = s['car']['start_soc'] + (s['car']['target_soc'] - s['car']['start_soc']) * progress
            s['car']['current_soc'] = round(current_soc, 2)
            s['car']['remaining_min'] = max(0, round(s['est_duration'] - elapsed, 0))

            if progress < 1.0:
                active_now.append(s)

        # 检查等待队列
        used_piles_now = {s['pile_idx'] for s in active_now}
        free_now = [i for i in range(total_piles) if i not in used_piles_now]
        while waiting_queue and free_now:
            car = waiting_queue.pop(0)
            pile_idx = _rng.choice(free_now)
            free_now.remove(pile_idx)
            charge_power = min(pile_types[pile_idx]['power'], car['battery_kwh'] * 1.5)
            duration_min = car['needed_kwh'] / charge_power * 60
            active_now.append({
                'pile_idx': pile_idx, 'car': car,
                'charge_power': round(charge_power, 1), 'start_min': minute,
                'est_duration': round(duration_min, 0),
            })
            used_piles_now.add(pile_idx)

        active_sessions = active_now

        # 统计总功率
        total_power = sum(s['charge_power'] for s in active_sessions)

        # 每5分钟输出一次快照
        if minute % 5 == 0 or minute == 59:
            snap = []
            for s in active_sessions:
                snap.append({
                    'pile': pile_types[s['pile_idx']]['id'],
                    'pile_idx': s['pile_idx'],
                    'pile_type': pile_types[s['pile_idx']]['type'],
                    'car_id': s['car']['id'],
                    'car_type': s['car']['type'],
                    'car_color': s['car']['color'],
                    'soc': s['car'].get('current_soc', s['car']['start_soc']),
                    'remaining_min': s['car'].get('remaining_min', s['est_duration']),
                    'charge_power': s['charge_power'],
                })
            minutes.append({
                'minute': minute,
                'active_sessions': len(active_sessions),
                'waiting': len(waiting_queue),
                'total_power_kw': round(total_power, 1),
                'piles_detail': snap,
            })

    # 小时总览
    total_energy = sum(m['total_power_kw'] for m in minutes) / 12  # 5分钟间隔 -> kWh近似

    return {
        'hour': hour,
        'day_type': day_type,
        'month': month,
        'total_piles': total_piles,
        'pile_config': [{'id': p['id'], 'type': p['type'], 'x': p['x']} for p in pile_types],
        'total_energy_kwh': round(total_energy, 1),
        'peak_power_kw': max((m['total_power_kw'] for m in minutes), default=0),
        'avg_active_sessions': round(sum(m['active_sessions'] for m in minutes) / max(len(minutes), 1), 1),
        'timeline': minutes,
    }


# ============================================================
# 静态文件
# ============================================================

os.makedirs(STATIC_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def index():
    """主页"""
    index_path = os.path.join(STATIC_DIR, 'index.html')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Frontend not built yet. Run web_server.py and ensure static/index.html exists.</h1>")


# 挂载静态文件目录
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Web Server for Microgrid Simulation')
    parser.add_argument('--port', type=int, default=8760, help='Server port (default: 8760)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    args = parser.parse_args()

    print("=" * 55)
    print("  高速服务区光储充微网仿真系统 — Web 服务器 v6.5")
    print("=" * 55)
    print(f"  http://{args.host}:{args.port}")
    print("=" * 55)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == '__main__':
    main()
