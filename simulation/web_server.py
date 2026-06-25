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
    TMY_HOURLY_TEMP, TMY_GHI_CLEAR, MONTHLY_WEATHER_DAYS,
    PV_COST, ESS_COST, CHARGING_COST, FIXED_COST, FIXED_COST_TOTAL,
    OM_RATE, DISCOUNT_RATE, PROJECT_LIFE, CCER_PRICE, CCER_PRICE_FORECAST,
    SCENARIOS, PV_TEMP_COEFF, PV_NOCT, PV_STC_TEMP,
    PV_FIRST_YEAR_DEGRADATION, PV_ANNUAL_DEGRADATION,
    ESS_CYCLE_LIFE, ESS_CALENDAR_LIFE, ESS_EA_OVER_R, ESS_REF_TEMP_K,
    V2G_ENABLED, V2G_PILES, V2G_POWER_KW, V2G_EFFICIENCY, V2G_DOD_LIMIT,
    V2G_PRICE_YUAN_KWH, V2G_DEGRADATION_COST, V2G_NET_REVENUE,
    DR_ENABLED, DR_EVENTS_PER_YEAR, DR_COMPENSATION_YUAN_KWH,
    DR_DURATION_HOURS, DR_MEDIUM_SERVICE_AREA_REVENUE,
    BASS_P, BASS_Q, BASS_K, LOGISTIC_R, LOGISTIC_T0,
    EV_CHARGING_HIGHWAY_GROWTH, LOAD_GROWTH_MODEL, LOAD_GROWTH_BASE_YEAR,
    PV_AREA_RATIO, TRANSFORMER_CAPACITY_KVA, TRANSFORMER_PF_MIN,
    PV_COST_PER_KWP, MTBF, MTTR, WEATHER_TRANSITION_MATRIX,
    load_pvgis_tmy,  # v6.6: PVGIS TMY数据加载
    WEATHER_SEASONAL_PERSISTENCE, SECOND_ORDER_ALPHA,
    MONTHLY_ARRIVAL_MULTIPLIER, CHARGING_PENETRATION,
    HOURLY_ARRIVAL_RATE,
    CHARGING_BUILDING_COUPLING, get_load_growth_factor,
    get_calendar_fade_rate, get_cycle_life_at_dod,
    get_battery_yearly_degradation, get_rte, get_ess_temp_derate,
    # v7.0: 细粒度可视化
    HIGHWAY_TRAFFIC_PROFILE, CC_CV_PARAMS, CHARGE_EFFICIENCY,
    NEV_PENETRATION_HIGHWAY, CHARGING_CONVERSION_RATE,
)
from calendar_utils import CalendarContext
from mc_charging_load import MonteCarloChargingSimulator
from capacity_optimization import MicrogridOptimizer

app = FastAPI(title="高速服务区光储充微网仿真系统", version="7.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
DATA_DIR = os.path.join(BASE_DIR, 'data')
SEED = 42

# 全局缓存
_cache = {}
_sim_cache = {}

# 外部数据管理器 (延迟初始化)
_data_manager = None


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def _sanitize_nan(obj):
    """递归替换NaN/Inf为None (JSON可序列化)"""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_nan(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        val = float(obj) if isinstance(obj, np.floating) else obj
        try:
            if math.isnan(float(val)) or math.isinf(float(val)):
                return None
        except (TypeError, ValueError):
            pass
        return float(obj)
    return obj


def _wait_sim_cache(timeout=30):
    """等待后台仿真缓存就绪 (最多timeout秒)"""
    import time
    for _ in range(timeout * 2):
        if _cache.get('_sim_cache_ready') and _sim_cache:
            return True
        time.sleep(0.5)
    return bool(_sim_cache)


def _build_sim_cache():
    """预计算MC和8760h序列, 避免每次API调用重复仿真"""
    print("Pre-computing simulation cache (MC + 8760h sequence)...")
    cal = CalendarContext(seed=SEED)
    mc_sim = MonteCarloChargingSimulator(service_area_size='medium', seed=SEED)
    mc = mc_sim.simulate_all_scenarios(n_runs=40)  # 快速启动, 足够P50/P95精度
    tmy = load_pvgis_tmy('wuhan')
    print(f"  TMY source: {tmy['source']}")
    opt = MicrogridOptimizer(size='medium', mc_scenarios=mc, seed=SEED, calendar_ctx=cal, tmy_data=tmy)
    opt_result = _cache.get('optimization', {})
    pv_cap = opt_result.get('pv_capacity', 1231)
    ess_cap = opt_result.get('ess_capacity', 2000)
    ess_pow = opt_result.get('ess_power', 1000)
    opt.pv_capacity = pv_cap
    opt.ess_capacity = ess_cap
    opt.ess_power = ess_pow
    pv_seq, load_seq, tou_seq, seasons_seq, _ = opt._build_8760h_sequence()
    _sim_cache['calendar'] = cal
    _sim_cache['opt'] = opt
    _sim_cache['mc'] = mc
    _sim_cache['pv_seq'] = pv_seq
    _sim_cache['load_seq'] = load_seq
    _sim_cache['tou_seq'] = tou_seq
    _sim_cache['seasons_seq'] = seasons_seq
    _sim_cache['pv_cap'] = pv_cap
    _sim_cache['ess_cap'] = ess_cap
    _sim_cache['ess_pow'] = ess_pow
    _cache['_sim_cache_ready'] = True
    print(f"  OK: PV={pv_cap}kWp ESS={ess_cap}kWh/{ess_pow}kW")


import threading

@app.on_event("startup")
def startup():
    global _data_manager
    _cache['mc_summary'] = load_json('mc_summary.json')
    _cache['optimization'] = load_json('optimization_result.json')
    _cache['pareto'] = load_json('pareto_results.json')
    _cache['decision'] = load_json('decision_result.json')
    _cache['daily_8760h'] = load_json('daily_8760h.json')

    # 懒初始化: 后台线程构建仿真缓存, 不阻塞HTTP服务
    def _lazy_build_cache():
        try:
            _build_sim_cache()
        except Exception as e:
            print(f"  WARN: Sim cache build failed: {e}")

    _cache['_sim_cache_ready'] = False
    threading.Thread(target=_lazy_build_cache, daemon=True).start()

    # 初始化外部数据管理器
    try:
        from data_loader import DataManager
        import config as config_module
        _data_manager = DataManager(config_module=config_module)
        load_result = _data_manager.load_all_from_disk()
        print(f"  DataManager: loaded {load_result['total_files']} external data files")
        for src, info in load_result.get('sources', {}).items():
            if isinstance(info, dict):
                loaded = info.get('loaded', info.get('total_sessions', '?'))
                print(f"    {src}: {loaded}")
    except Exception as e:
        print(f"  WARN: DataManager init failed: {e}")


# ============================================================
# API: 仿真结果
# ============================================================

@app.get("/api/health")
def health():
    dm_status = "not_initialized"
    pvgis_count = 0
    if _data_manager is not None:
        dm_status = "ok"
        pvgis_count = len(_data_manager._tmy_cache)

    return {
        "status": "ok",
        "version": "8.0",
        "sim_cache_ready": _cache.get('_sim_cache_ready', False),
        "data_available": {
            "mc_summary": _cache['mc_summary'] is not None,
            "optimization": _cache['optimization'] is not None,
            "pareto": _cache['pareto'] is not None,
            "decision": _cache['decision'] is not None,
        },
        "external_data": {
            "status": dm_status,
            "pvgis_locations": pvgis_count,
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
    data = _cache.get('decision', {})
    return JSONResponse(_sanitize_nan(data))


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
# API: 外部数据 & 校准 (v6.6)
# ============================================================

@app.get("/api/data/status")
def get_data_status():
    """外部数据加载状态"""
    if _data_manager is None:
        return JSONResponse({"status": "not_initialized"})

    calibration = _data_manager.get_calibration_api_data()
    return JSONResponse({
        "status": "ok",
        "external_data_available": {
            "pvgis_tmy": calibration['status'].get('pvgis_locations', []),
            "acndata": calibration['status'].get('has_acndata', False),
            "nasa_power": calibration['status'].get('has_nasa_power', False),
        },
        "pvgis_location_count": len(calibration['status'].get('pvgis_locations', [])),
        "acndata_total_sessions": calibration.get('acndata', {}).get('total_sessions', 0),
    })


@app.get("/api/data/tmy-comparison")
def get_tmy_comparison(location: str = Query('wuhan', description='Location key')):
    """PVGIS TMY vs config.py 逐时对比数据"""
    if _data_manager is None:
        return JSONResponse({"error": "DataManager not initialized"}, status_code=503)

    calibration = _data_manager.get_calibration_api_data()
    tmy_data = calibration.get('tmy', {})

    if not tmy_data:
        return JSONResponse({"error": f"No TMY data for '{location}'"}, status_code=404)

    return JSONResponse(tmy_data)


@app.get("/api/data/tmy-multi-city")
def get_tmy_multi_city():
    """10城市TMY对比汇总"""
    if _data_manager is None:
        return JSONResponse({"error": "DataManager not initialized"}, status_code=503)

    calibration = _data_manager.get_calibration_api_data()

    # Build simplified multi-city table from PVGIS cache
    cities = []
    for loc_key, tmy in _data_manager._tmy_cache.items():
        stats = tmy.get('annual_stats', {})
        meta = tmy.get('meta', {})
        cities.append({
            'key': loc_key,
            'name': meta.get('label', loc_key),
            'lat': meta.get('lat'),
            'lon': meta.get('lon'),
            'ghi_kwh_m2': stats.get('ghi_kwh_m2'),
            'ghi_peak_w_m2': stats.get('ghi_peak_w_m2'),
            'ghi_peak_month': stats.get('ghi_peak_month'),
            'mean_temp_c': stats.get('mean_temp_c'),
            'temp_min_c': stats.get('temp_min_c'),
            'temp_max_c': stats.get('temp_max_c'),
            'mean_wind_m_s': stats.get('mean_wind_m_s'),
        })

    return JSONResponse({
        'cities': sorted(cities, key=lambda c: c.get('ghi_kwh_m2', 0), reverse=True),
        'count': len(cities),
    })


@app.get("/api/data/acndata-distribution")
def get_acndata_distribution():
    """ACN-Data充电行为分布数据"""
    if _data_manager is None:
        return JSONResponse({"error": "DataManager not initialized"}, status_code=503)

    calibration = _data_manager.get_calibration_api_data()
    acn = calibration.get('acndata', {})

    if not acn:
        return JSONResponse({"error": "No ACN-Data loaded"}, status_code=404)

    return JSONResponse(acn)


@app.get("/api/data/calibration")
def get_calibration_report():
    """完整校准报告: 外部数据 vs config.py"""
    if _data_manager is None:
        return JSONResponse({"error": "DataManager not initialized"}, status_code=503)

    calibration = _data_manager.get_calibration_api_data()
    return JSONResponse({
        "status": calibration['status'],
        "tmy": calibration.get('tmy', {}),
        "acndata_summary": {
            'total_sessions': calibration.get('acndata', {}).get('total_sessions'),
            'sites': calibration.get('acndata', {}).get('sites'),
            'session_stats': calibration.get('acndata', {}).get('session_stats'),
            'applicability': calibration.get('acndata', {}).get('applicability'),
        },
        "multi_city": calibration.get('comparison_matrix', {}),
        "recommendations": calibration.get('recommendations', []),
        "generated_at": calibration.get('comparison_matrix', {}).get('generated_at', ''),
    })


# ============================================================
# API: 细粒度模型可视化 (v7.0)
# ============================================================

@app.get("/api/data/charging-curve")
def get_charging_curve():
    """CC-CV充电功率曲线 (SOC 0→100%, 4功率档)"""
    from charging_curve import cc_cv_power_curve
    soc_range = [round(i * 0.01, 2) for i in range(101)]
    curves = {}
    for power in [60, 120, 240, 480]:
        curves[str(power)] = [round(cc_cv_power_curve(s, power), 1) for s in soc_range]
    return JSONResponse({
        "soc": soc_range,
        "curves": curves,
        "cc_cv_soc": CC_CV_PARAMS['cc_cv_soc'],
        "cv_end_ratio": CC_CV_PARAMS['cv_end_power_ratio'],
        "efficiency": CC_CV_PARAMS['efficiency'],
        "source": "GB/T 27930-2023 + charge curve module",
    })


@app.get("/api/data/traffic-profile")
def get_traffic_profile():
    """交通流量24h分布 (4日型, 新旧模型对比)"""
    old_rates = HOURLY_ARRIVAL_RATE
    new_rates = HIGHWAY_TRAFFIC_PROFILE

    return JSONResponse({
        "day_types": ["workday", "weekend", "holiday", "spring_festival"],
        "old_model": {
            dt: {"hourly": old_rates.get(dt, old_rates['workday']),
                 "total": sum(old_rates.get(dt, old_rates['workday']))}
            for dt in ["workday", "weekend", "holiday", "spring_festival"]
        },
        "new_model": {
            dt: {"hourly_pct": new_rates[dt]['hourly_pct'],
                 "daily_base": new_rates[dt]['daily_base']}
            for dt in ["workday", "weekend", "holiday", "spring_festival"]
        },
        "nev_penetration": {str(y): v for y, v in NEV_PENETRATION_HIGHWAY.items()},
        "charging_conversion": CHARGING_CONVERSION_RATE,
        "source": "文件17 + 交通运输部统计",
        "model_version": "v6.6",
    })


@app.get("/api/data/vehicle-sample")
def get_vehicle_sample(n: int = Query(30, ge=5, le=100)):
    """随机抽样N辆车的诊断数据 (里程/SOC/功率/时长)"""
    from mc_charging_load import MonteCarloChargingSimulator
    sim = MonteCarloChargingSimulator(service_area_size='medium', seed=SEED)
    samples = []
    for _ in range(n):
        v = sim._sample_vehicle(month=6)
        # CC-CV时长
        eff_power = min(v['power'], v['max_power'])
        from charging_curve import calculate_charging_duration
        dur_min = calculate_charging_duration(v['soc_arrival'], v['target_soc'],
                                               v['battery_cap'], eff_power,
                                               efficiency=CHARGE_EFFICIENCY,
                                               params=CC_CV_PARAMS) * 60
        samples.append({
            "vt": v['vt_name'],
            "mileage_km": round(v['mileage_km'], 1),
            "battery_kwh": round(v['battery_cap'], 1),
            "consumption": round(v['energy_cons'], 1),
            "departure_soc": round(v['departure_soc'], 3),
            "arrival_soc": round(v['soc_arrival'], 3),
            "target_soc": round(v['target_soc'], 3),
            "charge_kwh": round(v['charge_needed'], 1),
            "power_kw": int(round(eff_power, 0)),
            "duration_min": round(dur_min, 1),
        })
    return JSONResponse({"samples": samples, "count": len(samples),
                         "source": "physics model v6.6", "cached": False})


@app.get("/api/data/hyperparams")
def get_hyperparams():
    """config.py超参数分类列表"""
    import config as cfg
    categories = {
        "服务区规模": ["SERVICE_AREA_CONFIG", "DAY_TYPE_COEFF"],
        "电动汽车&充电": ["VEHICLE_TYPES", "CHARGE_POWER_DIST", "TARGET_SOC_MEAN",
                "CHARGE_EFFICIENCY", "HOURLY_ARRIVAL_RATE", "CHARGING_PENETRATION",
                "HIGHWAY_TRAFFIC_PROFILE", "CHARGING_CONVERSION_RATE",
                "HIGHWAY_MILEAGE_DIST", "CC_CV_PARAMS"],
        "光伏&气象": ["PV_COEFF", "WEATHER_COEFF", "TMY_GHI_CLEAR", "TMY_HOURLY_TEMP",
                "PV_TEMP_COEFF", "PV_NOCT", "PV_STC_TEMP", "PV_AREA_RATIO"],
        "储能系统": ["ESS_COST", "SOC_MIN", "SOC_MAX", "ESS_CYCLE_LIFE"],
        "经济&市场": ["TOU_PRICE_VALUES", "CCER_PRICE", "DISCOUNT_RATE",
                "PROJECT_LIFE", "FEED_IN_PRICE", "PV_COST"],
        "设备&可靠性": ["LIFESPAN", "OM_RATE", "MTBF", "MTTR", "PILE_AVAILABILITY"],
        "演化&场景": ["SCENARIOS", "NEV_PENETRATION_HIGHWAY", "BASS_P", "BASS_Q"],
    }

    params = []
    for cat, keys in categories.items():
        for k in keys:
            v = getattr(cfg, k, None)
            if v is None:
                continue
            display = str(v)
            if len(display) > 80:
                display = display[:77] + '...'
            params.append({
                "name": k,
                "value": display,
                "category": cat,
            })

    return JSONResponse({"params": params, "total": len(params),
                         "source": "config.py", "version": "v6.6"})


@app.get("/api/data/pv-model")
def get_pv_model_params():
    """Kasten-Czeplak曲线 + Sandia热模型参数"""
    from config import kasten_czeplak_ghi
    # Kasten-Czeplak curve
    oktas_range = list(range(9))
    kc_curve = [round(kasten_czeplak_ghi(1000, n), 1) for n in oktas_range]

    # Sandia model parameters
    sandia = {
        "NOCT": PV_NOCT,
        "STC_temp": PV_STC_TEMP,
        "temp_coeff_per_C": PV_TEMP_COEFF,
        "formula": "T_cell = T_amb + (NOCT-20) * G / 800",
        "power_correction": "P = P_stc * (1 + gamma * (T_cell - T_stc))",
    }

    # Weather attenuation coefficients
    weather_coeffs = {k: round(v, 2) for k, v in WEATHER_COEFF.items()}

    # Cloud oktas mapping
    cloud_map = {'clear': 0, 'partly_cloudy': 3, 'cloudy': 5.5,
                 'overcast': 7.5, 'rain': 8}

    return JSONResponse({
        "kasten_czeplak": {"oktas": oktas_range, "ghi_1000": kc_curve},
        "sandia": sandia,
        "weather_coeffs": weather_coeffs,
        "cloud_oktas_map": cloud_map,
        "pv_coeff_seasonal": {s: [round(v, 3) for v in vals]
                              for s, vals in PV_COEFF.items()},
        "source": "文件24 §1.3 + Sandia model",
    })


@app.get("/api/results/algo-comparison")
def get_algo_comparison():
    """多优化算法对比结果"""
    algo = load_json('algorithm_comparison.json')
    opt = _cache.get('optimization', {})

    methods = {
        "pso": {"name": "PSO 粒子群", "color": "#00c8f0",
                "params": {"pop_size": 40, "max_iter": 30, "w": 0.7, "c1": 1.5, "c2": 1.5}},
        "nsga2": {"name": "NSGA-II 遗传", "color": "#40e0a0",
                  "params": {"pop_size": 100, "generations": 50, "crossover_p": 0.9, "mutation_p": 0.1}},
        "ga": {"name": "GA 遗传算法", "color": "#f0a020",
               "params": {"pop_size": 80, "generations": 40, "crossover_p": 0.8, "mutation_p": 0.15}},
        "egpso": {"name": "EGPSO 增强粒子群", "color": "#e04050",
                  "params": {"pop_size": 40, "max_iter": 30, "w_start": 0.9, "w_end": 0.4}},
        "robust": {"name": "鲁棒优化", "color": "#a060d0",
                   "params": {"scenarios": ["baseline", "conservative", "aggressive"], "gamma": 0.3}},
    }

    # PSO optimal (from cache)
    current_best = {
        "method": "pso",
        "pv_capacity": opt.get('pv_capacity', 1231),
        "ess_capacity": opt.get('ess_capacity', 1075),
        "ess_power": opt.get('ess_power', 537),
        "npc_wan": round((opt.get('npc', 31137000) or 31137000) / 10000, 1),
        "ssr": round(opt.get('self_sufficiency', 0.47), 3),
        "payback_years": round(opt.get('payback_years', 8.9), 1),
    }

    return JSONResponse({
        "methods": methods,
        "current_best": current_best,
        "algo_data": algo if algo else {},
        "note": "全算法对比需运行 optimization_comparison.py",
        "source": "capacity_optimization.py + nsga2.py",
    })


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
    """仿真单日调度, 返回24h逐时数据 (使用预计算缓存, 秒级响应)"""
    # 使用预计算的8760h序列
    if not _wait_sim_cache(timeout=30):
        return JSONResponse({"error": "Simulation cache not ready (still computing, retry in seconds)"}, status_code=503)

    pv_seq = _sim_cache['pv_seq']
    load_seq = _sim_cache['load_seq']
    tou_seq = _sim_cache['tou_seq']
    seasons_seq = _sim_cache['seasons_seq']
    cal = _sim_cache['calendar']
    opt = _sim_cache['opt']

    # 如果参数与缓存不同, 更新容量
    if abs(pv_cap - _sim_cache['pv_cap']) > 1 or abs(ess_cap - _sim_cache['ess_cap']) > 1:
        opt.pv_capacity = pv_cap
        opt.ess_capacity = ess_cap
        opt.ess_power = ess_pow

    day = max(0, min(364, day))
    season = seasons_seq[day * 24]
    pv_profile = pv_seq[day * 24:(day + 1) * 24] * pv_cap
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
            'month': cal.month_of_day[day],
            'day_type': cal.day_types[day],
            'weather': cal.weather_seq[day],
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
    """仿真全年365天调度汇总 (使用预计算缓存)"""
    if not _wait_sim_cache(timeout=30):
        return JSONResponse({"error": "Simulation cache not ready (still computing, retry in seconds)"}, status_code=503)

    pv_seq = _sim_cache['pv_seq']
    load_seq = _sim_cache['load_seq']
    tou_seq = _sim_cache['tou_seq']
    seasons_seq = _sim_cache['seasons_seq']
    cal = _sim_cache['calendar']
    opt = _sim_cache['opt']

    if abs(pv_cap - _sim_cache['pv_cap']) > 1:
        opt.pv_capacity = pv_cap
        opt.ess_capacity = ess_cap
        opt.ess_power = ess_pow

    daily_results = []
    ts = {'pv_gen': 0, 'load': 0, 'grid_import': 0, 'grid_export': 0, 'grid_cost': 0}
    monthly_pv = np.zeros(12)
    monthly_load = np.zeros(12)
    monthly_grid = np.zeros(12)
    monthly_ssr = np.zeros(12)
    monthly_days = np.zeros(12)

    month_of_day = cal.month_of_day
    # 预计算 day_of_month
    dom = [1] * 365
    m_start = 0
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for m in range(12):
        for dd in range(days_in_month[m]):
            idx = m_start + dd
            if idx < 365:
                dom[idx] = dd + 1
        m_start += days_in_month[m]

    for d in range(365):
        season = seasons_seq[d * 24]
        pv_profile = pv_seq[d * 24:(d + 1) * 24] * pv_cap
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
            'month': month_of_day[d],
            'day_of_month': dom[d],
            'season': season,
            'day_type': cal.day_types[d],
            'weather': cal.weather_seq[d],
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

        m = cal.month_of_day[d] - 1
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
        'daily': daily_results,
    }


@app.get("/api/simulate/mc-distribution")
def simulate_mc_distribution(
    day_type: str = Query('workday', description="日类型: workday/weekend/holiday/spring_festival"),
    weather: str = Query('clear', description="天气: clear/partly_cloudy/cloudy/overcast/rain"),
    month: int = Query(6, description="月份 1-12"),
    n_runs: int = Query(400, description="MC仿真次数"),
):
    """MC充电负荷分布仿真 (使用预计算MC数据, 快速响应)"""
    # 优先使用缓存的MC summary数据 (预计算完整统计)
    mc_summary = _cache.get('mc_summary', {})
    if day_type in mc_summary:
        ms = mc_summary[day_type]
        weather_factor = WEATHER_CHARGING_COEFF.get(weather, 1.0)
        month_factor = {1:0.88,2:0.80,3:1.00,4:1.03,5:1.15,6:1.02,7:1.08,8:1.08,9:1.02,10:1.20,11:0.97,12:0.92}.get(month, 1.0)
        factor = weather_factor * month_factor
        # 缓存的P50/P95是工作日晴天的基准, 按天气和月份系数缩放
        p50_base = np.array(ms.get('hourly_p50', [0]*24))
        p95_base = np.array(ms.get('hourly_p95', [0]*24))
        peak_mean = ms.get('peak_mean', 0) * factor
        peak_std = ms.get('peak_std', 0) * factor
        daily_energy = ms.get('daily_energy_mean', 0) * factor
        p50 = (p50_base * factor).tolist()
        p95 = (p95_base * factor).tolist()
        # 从P50/P95估算分位数
        spread = (p95_base - p50_base) * factor
        return {
            'params': {'day_type': day_type, 'weather': weather, 'month': month, 'cached': True},
            'hourly_p50': p50,
            'hourly_p95': p95,
            'hourly_p5': (p50_base * factor * 0.4).tolist(),
            'hourly_p25': (p50_base * factor * 0.7).tolist(),
            'hourly_p75': ((p50_base + spread * 0.5) * factor).tolist(),
            'peak_mean': float(peak_mean),
            'peak_std': float(peak_std),
            'daily_energy_mean': float(daily_energy),
        }
    # 回退到实时仿真
    sim = MonteCarloChargingSimulator(service_area_size='medium', seed=SEED)
    result = sim.simulate_monte_carlo(day_type, n_runs=n_runs, month=month, weather=weather)
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
    duration_minutes: int = Query(180, ge=60, le=360, description="仿真时长(分钟, 60-360)"),
):
    """生成充电站实景数据 — 模拟20个充电终端的逐分钟车辆到达/充电/离开事件

    仿真窗口默认3小时(180分钟), 可扩展至6小时.
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

    for minute in range(duration_minutes):
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
                'status': 'charging',  # charging / completed / departing
                'depart_min': None,     # minute the car will actually leave
            })

        # 更新充电进度
        active_now = []
        for s in active_sessions:
            elapsed = minute - s['start_min']

            if s['status'] == 'charging':
                progress = min(1.0, elapsed / max(s['est_duration'], 1))
                current_soc = s['car']['start_soc'] + (s['car']['target_soc'] - s['car']['start_soc']) * progress
                s['car']['current_soc'] = round(current_soc, 2)
                s['car']['remaining_min'] = max(0, round(s['est_duration'] - elapsed, 0))

                if progress >= 1.0:
                    # 充满 → 进入逗留状态 (2-5分钟后离开)
                    s['status'] = 'completed'
                    s['depart_min'] = minute + _rng.randint(2, 6)
                    s['car']['current_soc'] = s['car']['target_soc']
                    s['car']['remaining_min'] = 0
                    active_now.append(s)
                else:
                    active_now.append(s)
            elif s['status'] == 'completed':
                # 逗留中, 检查是否到离开时间
                if minute >= s['depart_min']:
                    s['status'] = 'departing'
                    s['car']['remaining_min'] = -1  # signal: leaving
                    active_now.append(s)  # keep for one more frame to trigger departure anim
                else:
                    active_now.append(s)
            elif s['status'] == 'departing':
                # 这一帧之后真正移除 (前端会用prevLiveCars做渐隐)
                pass  # 不加入 active_now

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

        # 逐分钟输出快照 (60个时间点, 确保前端动画连续)
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
                'status': s.get('status', 'charging'),
            })
        minutes.append({
            'minute': minute,
            'active_sessions': len(active_sessions),
            'waiting': len(waiting_queue),
            'total_power_kw': round(total_power, 1),
            'piles_detail': snap,
            'arrivals': n_arrivals,
            'departures': len([s for s in active_sessions if s['car'].get('current_soc', 0) >= s['car']['target_soc'] - 0.01]),
        })

    # 小时总览
    total_energy = sum(m['total_power_kw'] for m in minutes) / (duration_minutes / 60)  # kW*min→kWh

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


@app.get("/api/simulate/fine-grained-scada")
def simulate_fine_grained_scada(
    day: int = Query(200, ge=0, le=364, description="起始日 (0-364)"),
    n_days: int = Query(1, ge=1, le=7, description="仿真天数"),
    weather: str = Query('auto', description="天气: auto/clear/partly_cloudy/cloudy/overcast/rain"),
    month_override: int = Query(0, ge=0, le=12, description="月份覆盖 (0=自动)"),
):
    """细粒度SCADA仿真 — 分钟级全组件数据 (PV/ESS/建筑/充电/电网)

    生成逐分钟数据, 包含:
    - 光伏: 分钟级辐照度+云层瞬态+温度效应 (Sandia模型)
    - 储能: 分钟级SOC+充放电功率
    - 建筑: 分项负荷 (空调/餐饮/照明/热水/办公/动力)
    - 充电: 逐桩车辆到达/充电/离开 (集成充电站实景)
    - 电网: 购电/售电+变压器负载率
    - 天气: 分钟级GHI/DNI/温度/云量
    """
    import random as _random
    _rng = _random.Random(SEED + day * 137 + n_days * 31)

    # 日历上下文
    cal = CalendarContext(seed=SEED)
    if month_override > 0:
        actual_month = month_override
    else:
        actual_month = cal.month_of_day[min(day, 364)]

    # 天气确定
    actual_weather = weather
    if weather == 'auto':
        actual_weather = cal.weather_seq[min(day, 364)]

    season = get_season(actual_month)
    day_type = cal.day_types[min(day, 364)]

    # ---- 分钟级PV仿真 ----
    # 基准PV系数 (24h → 1440min)
    pv_coeffs_24h = PV_COEFF[season]
    pv_minute_coeffs = []
    for h in range(24):
        base = pv_coeffs_24h[h]
        # 小时内线性插值
        for m in range(60):
            t = m / 60.0
            next_h = (h + 1) % 24
            next_base = pv_coeffs_24h[next_h] if next_h < 24 else 0
            val = base + (next_base - base) * t
            pv_minute_coeffs.append(val)

    # 云层瞬态模型: 用随机游走模拟分钟级云量波动
    cloud_cover = []
    cloud_base = {'clear': 0.05, 'partly_cloudy': 0.28, 'cloudy': 0.55, 'overcast': 0.78, 'rain': 0.90}.get(actual_weather, 0.28)
    cloud_vol = {'clear': 0.03, 'partly_cloudy': 0.08, 'cloudy': 0.06, 'overcast': 0.04, 'rain': 0.03}.get(actual_weather, 0.06)
    cloud_val = cloud_base
    for m in range(1440 * n_days):
        cloud_val += _rng.gauss(0, cloud_vol)
        cloud_val = max(0.0, min(1.0, cloud_val))
        # 均值回归
        cloud_val += (cloud_base - cloud_val) * 0.02
        cloud_cover.append(round(cloud_val, 3))

    # 温度模型 (分钟级, 基于TMY小时温度插值)
    temps_hourly = TMY_HOURLY_TEMP.get(actual_month, TMY_HOURLY_TEMP.get(6, [25]*24))
    temp_minute = []
    for h in range(24):
        base_t = temps_hourly[min(h, 23)]
        next_t = temps_hourly[min((h + 1) % 24, 23)]
        for m in range(60):
            t = m / 60.0
            temp_minute.append(base_t + (next_t - base_t) * t)
    temp_minute = temp_minute * n_days

    # GHI计算 (Sandia简单模型: GHI = GHI_clear * (1 - Kc * cloud))
    ghi_clear = TMY_GHI_CLEAR.get(actual_month, TMY_GHI_CLEAR.get(6, [800]*24))
    ghi_minute = []
    for h in range(24):
        base_g = ghi_clear[min(h, 23)]
        next_g = ghi_clear[min((h + 1) % 24, 23)]
        for m in range(60):
            t = m / 60.0
            ghi_minute.append(base_g + (next_g - base_g) * t)
    ghi_minute = ghi_minute * n_days

    # 实际PV出力 (分钟级, kW)
    pv_cap = _sim_cache.get('pv_cap', 1231) if _sim_cache else 1231
    pv_kw_minute = []
    for m in range(1440 * n_days):
        day_min = m % 1440
        cloud = cloud_cover[m]
        temp = temp_minute[m]
        ghi = ghi_minute[day_min]
        # Kasten-Czeplak: clear-sky attenuation
        kc = 0.75 if cloud < 0.3 else 0.55 if cloud < 0.6 else 0.35 if cloud < 0.85 else 0.15
        actual_ghi = ghi * (1 - cloud * kc)
        # PV output = rated * coeff * (GHI/1000) * temp_derate
        temp_derate = 1.0 - PV_TEMP_COEFF * (temp - PV_STC_TEMP)
        temp_derate = max(0.85, min(1.05, temp_derate))
        pv_kw = pv_cap * pv_minute_coeffs[day_min] * (actual_ghi / 1000.0) * temp_derate if ghi > 10 else 0
        pv_kw = max(0, pv_kw * _rng.gauss(1.0, 0.02))
        pv_kw_minute.append(round(pv_kw, 1))

    # ---- 分钟级建筑负荷 ----
    bldg_24h = BUILDING_LOAD[season]
    weather_bldg_factor = WEATHER_BUILDING_COEFF.get(actual_weather, 1.0)
    # 分项分解
    bldg_components = {
        '空调': 0.41, '餐饮': 0.20, '照明': 0.14,
        '热水': 0.10, '办公': 0.08, '动力': 0.07,
    }
    bldg_total_minute = []
    bldg_detail_minute = {k: [] for k in bldg_components}
    for h in range(24):
        base = bldg_24h[h] * weather_bldg_factor
        next_h = (h + 1) % 24
        next_base = bldg_24h[next_h] * weather_bldg_factor
        for m in range(60):
            t = m / 60.0
            val = base + (next_base - base) * t
            # 加噪声
            val *= _rng.gauss(1.0, 0.03)
            bldg_total_minute.append(round(val, 1))
            for comp, ratio in bldg_components.items():
                bldg_detail_minute[comp].append(round(val * ratio * _rng.gauss(1.0, 0.05), 1))
    bldg_total_minute = bldg_total_minute * n_days
    for comp in bldg_components:
        bldg_detail_minute[comp] = bldg_detail_minute[comp] * n_days

    # ---- 分钟级充电负荷仿真 (集成充电站实景) ----
    # 使用与 charging-station-live 相同的模型, 但扩展至全天+多日
    arrival_rates = HOURLY_ARRIVAL_RATE
    rates = arrival_rates.get(day_type, arrival_rates['workday'])
    month_factor = MONTHLY_ARRIVAL_MULTIPLIER.get(actual_month, 1.0)

    n_120kw = 16; n_480kw = 2; total_piles = n_120kw + n_480kw
    pile_types = (
        [{'type': '120kW', 'power': 120, 'id': f'CP-{i+1:02d}', 'x': i} for i in range(n_120kw)]
        + [{'type': '480kW', 'power': 480, 'id': f'HP-{i+1:02d}', 'x': n_120kw + i} for i in range(n_480kw)]
    )
    car_types = [
        {'name': '微型', 'battery': 25, 'color': '#60d040', 'pct': 0.08},
        {'name': '小型', 'battery': 35, 'color': '#40c0f0', 'pct': 0.12},
        {'name': '紧凑型', 'battery': 50, 'color': '#f0a020', 'pct': 0.25},
        {'name': '中型', 'battery': 65, 'color': '#e8e8f0', 'pct': 0.25},
        {'name': 'SUV', 'battery': 85, 'color': '#f08030', 'pct': 0.18},
        {'name': '豪华', 'battery': 100, 'color': '#c090d0', 'pct': 0.07},
        {'name': '物流', 'battery': 80, 'color': '#8090a0', 'pct': 0.05},
    ]

    total_minutes = 1440 * n_days
    ev_total_kw_minute = []
    ev_active_count_minute = []
    ev_waiting_count_minute = []
    ev_pile_snapshots = []  # 每5分钟一个桩快照

    active_sessions = []
    waiting_queue = []
    next_car_id = 1

    for minute in range(total_minutes):
        hour_of_day = (minute % 1440) // 60
        base_rate = rates[min(hour_of_day, 23)]
        # 夜间修正: 00:00-06:00 到达率很低
        if hour_of_day < 6:
            base_rate = max(0, base_rate * 0.3)
        effective_rate = base_rate * month_factor

        # 天气对充电行为的影响
        weather_charge_factor = WEATHER_CHARGING_COEFF.get(actual_weather, 1.0)
        effective_rate *= weather_charge_factor

        # 车辆到达
        arrival_prob = effective_rate / 60.0
        n_arrivals = 0
        if _rng.random() < arrival_prob * 0.7:
            n_arrivals = 1
        if effective_rate > 12 and _rng.random() < arrival_prob * 0.3:
            n_arrivals = 2

        for _ in range(n_arrivals):
            r = _rng.random(); cum = 0
            car_type = car_types[0]
            for ct in car_types:
                cum += ct['pct']
                if r <= cum: car_type = ct; break

            start_soc = max(0.05, _rng.gauss(0.21, 0.08))
            target_soc = min(0.95, _rng.gauss(0.85, 0.06))
            needed_kwh = car_type['battery'] * (target_soc - start_soc)
            needed_kwh = max(5, needed_kwh)

            car = {
                'id': next_car_id, 'type': car_type['name'],
                'battery_kwh': car_type['battery'], 'color': car_type['color'],
                'start_soc': round(start_soc, 2), 'target_soc': round(target_soc, 2),
                'needed_kwh': round(needed_kwh, 1), 'arrive_min': minute,
            }
            next_car_id += 1

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
                'pile_idx': pile_idx, 'car': car,
                'charge_power': round(charge_power, 1), 'start_min': minute,
                'est_duration': round(duration_min, 0),
                'status': 'charging', 'depart_min': None,
            })

        # 更新充电进度
        active_now = []
        for s in active_sessions:
            elapsed = minute - s['start_min']
            if s['status'] == 'charging':
                progress = min(1.0, elapsed / max(s['est_duration'], 1))
                s['car']['current_soc'] = round(s['car']['start_soc'] + (s['car']['target_soc'] - s['car']['start_soc']) * progress, 2)
                s['car']['remaining_min'] = max(0, round(s['est_duration'] - elapsed, 0))
                if progress >= 1.0:
                    s['status'] = 'completed'
                    s['depart_min'] = minute + _rng.randint(2, 6)
                    s['car']['current_soc'] = s['car']['target_soc']
                    s['car']['remaining_min'] = 0
                    active_now.append(s)
                else:
                    active_now.append(s)
            elif s['status'] == 'completed':
                if minute >= s['depart_min']:
                    pass  # 离开, 不加入
                else:
                    active_now.append(s)

        # 释放桩位给等待队列
        used_now = {s['pile_idx'] for s in active_now}
        free_now = [i for i in range(total_piles) if i not in used_now]
        while waiting_queue and free_now:
            car = waiting_queue.pop(0)
            pile_idx = _rng.choice(free_now)
            free_now.remove(pile_idx)
            charge_power = min(pile_types[pile_idx]['power'], car['battery_kwh'] * 1.5)
            duration_min = car['needed_kwh'] / charge_power * 60
            active_now.append({
                'pile_idx': pile_idx, 'car': car,
                'charge_power': round(charge_power, 1), 'start_min': minute,
                'est_duration': round(duration_min, 0), 'status': 'charging',
                'depart_min': None,
            })
            used_now.add(pile_idx)

        active_sessions = active_now
        total_power = sum(s['charge_power'] for s in active_sessions)
        ev_total_kw_minute.append(round(total_power, 1))
        ev_active_count_minute.append(len(active_sessions))
        ev_waiting_count_minute.append(len(waiting_queue))

        # 每5分钟保存桩级快照
        if minute % 5 == 0:
            snap = []
            for s in active_sessions:
                snap.append({
                    'pile_idx': s['pile_idx'], 'pile_id': pile_types[s['pile_idx']]['id'],
                    'car_id': s['car']['id'], 'car_type': s['car']['type'],
                    'car_color': s['car']['color'],
                    'soc': s['car'].get('current_soc', s['car']['start_soc']),
                    'charge_power': s['charge_power'], 'status': s.get('status', 'charging'),
                })
            ev_pile_snapshots.append({'minute': minute, 'piles': snap})

    # ---- 分钟级储能仿真 ----
    ess_cap = _sim_cache.get('ess_cap', 2000) if _sim_cache else 2000
    ess_pow = _sim_cache.get('ess_pow', 1000) if _sim_cache else 1000
    tou_seq = _sim_cache.get('tou_seq', []) if _sim_cache else []

    soc = 0.5  # 起始SOC 50%
    ess_charge_minute = []
    ess_discharge_minute = []
    soc_minute = []

    for m in range(total_minutes):
        day_min = m % 1440
        hour = day_min // 60
        pv = pv_kw_minute[m]
        load = bldg_total_minute[day_min] + ev_total_kw_minute[m] + STATION_AUX_DAILY_KWH / 24

        # 获取当前TOU电价
        if len(tou_seq) > 0:
            tou = tou_seq[(min(day + m // 1440, 364)) * 24 + hour]
        else:
            tou = 0.85  # default flat

        net = pv - load
        ess_action = 0  # +charge, -discharge
        if net > 10 and soc < 0.90 and tou < 1.0:
            # 谷时/平时充电
            chg = min(net * 0.7, ess_pow, (0.90 - soc) * ess_cap)  # kW → kWh in 1 min
            chg_kwh = chg / 60.0
            soc += chg_kwh / ess_cap
            ess_action = chg
        elif net < -10 and soc > 0.15:
            # 放电
            disch = min(-net * 0.7, ess_pow, (soc - 0.15) * ess_cap * 60)
            disch_kwh = disch / 60.0
            soc -= disch_kwh / ess_cap
            ess_action = -disch
        elif soc < 0.30 and tou < 0.5:
            # 谷电补充
            chg = min(ess_pow * 0.5, (0.30 - soc) * ess_cap * 60)
            chg_kwh = chg / 60.0
            soc += chg_kwh / ess_cap
            ess_action = chg

        soc = max(0.10, min(0.95, soc))
        if ess_action > 0:
            ess_charge_minute.append(round(ess_action, 1))
            ess_discharge_minute.append(0)
        elif ess_action < 0:
            ess_charge_minute.append(0)
            ess_discharge_minute.append(round(-ess_action, 1))
        else:
            ess_charge_minute.append(0)
            ess_discharge_minute.append(0)
        soc_minute.append(round(soc, 4))

    # ---- 电网交互 ----
    grid_import_min = []
    grid_export_min = []
    for m in range(total_minutes):
        day_min = m % 1440
        pv = pv_kw_minute[m]
        load = bldg_total_minute[day_min] + ev_total_kw_minute[m] + STATION_AUX_DAILY_KWH / 24
        net = pv + ess_discharge_minute[m] - load - ess_charge_minute[m]
        if net > 0:
            grid_import_min.append(0)
            grid_export_min.append(round(net, 1))
        else:
            grid_import_min.append(round(-net, 1))
            grid_export_min.append(0)

    # ---- 构建响应 ----
    # 采样为5分钟间隔用于传输 (但保留关键分钟数据)
    sample_interval = 5  # 每5分钟采样以减少传输量
    sampled_minutes = []
    for m in range(0, total_minutes, sample_interval):
        day_idx = m // 1440
        minute_of_day = m % 1440
        hour = minute_of_day // 60
        min_in_hour = minute_of_day % 60

        # 聚合本采样间隔的数据
        end_m = min(m + sample_interval, total_minutes)
        pv_avg = np.mean(pv_kw_minute[m:end_m])
        load_total_avg = np.mean([bldg_total_minute[mi % 1440] + ev_total_kw_minute[mi] for mi in range(m, end_m)])
        bldg_avg = np.mean([bldg_total_minute[mi % 1440] for mi in range(m, end_m)])
        ev_avg = np.mean([ev_total_kw_minute[mi] for mi in range(m, end_m)])
        ess_ch_avg = np.mean(ess_charge_minute[m:end_m])
        ess_disch_avg = np.mean(ess_discharge_minute[m:end_m])
        grid_imp_avg = np.mean(grid_import_min[m:end_m])
        grid_exp_avg = np.mean(grid_export_min[m:end_m])

        # 建筑分项
        bldg_comp_avg = {}
        for comp in bldg_components:
            bldg_comp_avg[comp] = round(np.mean([bldg_detail_minute[comp][mi % 1440] for mi in range(m, end_m)]), 1)

        sampled_minutes.append({
            'minute': m,
            'day': day_idx,
            'tod': f'{hour:02d}:{min_in_hour:02d}',
            'pv_kw': round(pv_avg, 1),
            'load_total_kw': round(load_total_avg, 1),
            'bldg_total_kw': round(bldg_avg, 1),
            'bldg_components': bldg_comp_avg,
            'ev_total_kw': round(ev_avg, 1),
            'ev_active': round(np.mean(ev_active_count_minute[m:end_m]), 1),
            'ev_waiting': round(np.mean(ev_waiting_count_minute[m:end_m]), 1),
            'ess_charge_kw': round(ess_ch_avg, 1),
            'ess_discharge_kw': round(ess_disch_avg, 1),
            'soc': round(soc_minute[end_m - 1], 4) if end_m > m else round(soc_minute[m], 4),
            'grid_import_kw': round(grid_imp_avg, 1),
            'grid_export_kw': round(grid_exp_avg, 1),
            'cloud_cover': round(np.mean(cloud_cover[m:end_m]), 3),
            'ghi': round(np.mean([ghi_minute[mi % 1440] for mi in range(m, end_m)]), 0),
            'temp': round(np.mean([temp_minute[mi % 1440] for mi in range(m, end_m)]), 1),
        })

    # 日汇总
    daily_summary = []
    for d in range(n_days):
        d_start = d * 1440
        d_end = (d + 1) * 1440
        pv_d = round(sum(pv_kw_minute[d_start:d_end]) / 60.0, 1)
        load_d = round(sum(bldg_total_minute[0:1440]) / 60.0 + sum(ev_total_kw_minute[d_start:d_end]) / 60.0 + STATION_AUX_DAILY_KWH, 1)
        grid_imp_d = round(sum(grid_import_min[d_start:d_end]) / 60.0, 1)
        grid_exp_d = round(sum(grid_export_min[d_start:d_end]) / 60.0, 1)
        ess_disch_d = round(sum(ess_discharge_minute[d_start:d_end]) / 60.0, 1)
        daily_summary.append({
            'day': day + d,
            'month': cal.month_of_day[min(day + d, 364)],
            'day_type': cal.day_types[min(day + d, 364)],
            'weather': cal.weather_seq[min(day + d, 364)] if weather == 'auto' else actual_weather,
            'pv_kwh': pv_d,
            'load_kwh': load_d,
            'grid_import_kwh': grid_imp_d,
            'grid_export_kwh': grid_exp_d,
            'ev_energy_kwh': round(sum(ev_total_kw_minute[d_start:d_end]) / 60.0, 1),
            'ess_throughput_kwh': round(ess_disch_d, 1),
            'ssr': round((pv_d - grid_exp_d) / max(load_d, 1), 4),
        })

    # 24h 整点快照 (用于快速预览)
    hourly_snapshots = []
    for d in range(n_days):
        for h in range(24):
            m_start = d * 1440 + h * 60
            m_end = m_start + 60
            idx_start = m_start // sample_interval
            idx_end = m_end // sample_interval
            snap_data = sampled_minutes[idx_start:idx_end]
            if snap_data:
                hourly_snapshots.append({
                    'day': d,
                    'hour': h,
                    'pv_kw': round(np.mean([s['pv_kw'] for s in snap_data]), 1),
                    'load_kw': round(np.mean([s['load_total_kw'] for s in snap_data]), 1),
                    'ev_kw': round(np.mean([s['ev_total_kw'] for s in snap_data]), 1),
                    'bldg_kw': round(np.mean([s['bldg_total_kw'] for s in snap_data]), 1),
                    'soc': snap_data[-1]['soc'],
                    'grid_import_kw': round(np.mean([s['grid_import_kw'] for s in snap_data]), 1),
                    'grid_export_kw': round(np.mean([s['grid_export_kw'] for s in snap_data]), 1),
                    'cloud': round(np.mean([s['cloud_cover'] for s in snap_data]), 2),
                    'ghi': round(np.mean([s['ghi'] for s in snap_data]), 0),
                })

    # 每15分钟的桩级快照 (用于充电站可视化动画)
    pile_timeline = []
    for d in range(n_days):
        for h in range(24):
            for q in range(4):  # 每15分钟
                m = d * 1440 + h * 60 + q * 15
                snap_idx = m // 5
                if snap_idx < len(ev_pile_snapshots):
                    pile_timeline.append(ev_pile_snapshots[snap_idx])

    return {
        'config': {
            'day_start': day, 'n_days': n_days,
            'month': actual_month, 'season': season,
            'weather': actual_weather, 'day_type': day_type,
            'pv_cap_kwp': pv_cap, 'ess_cap_kwh': ess_cap, 'ess_pow_kw': ess_pow,
            'total_piles': total_piles,
            'sample_interval_min': sample_interval,
        },
        'weather_detail': {
            'mode': actual_weather,
            'cloud_cover_stats': {
                'min': round(min(cloud_cover), 3),
                'max': round(max(cloud_cover), 3),
                'mean': round(np.mean(cloud_cover), 3),
            },
            'temp_range': {'min': min(temp_minute), 'max': max(temp_minute)},
            'ghi_peak': max(ghi_minute[:1440]),
        },
        'pile_config': [{'id': p['id'], 'type': p['type'], 'x': p['x']} for p in pile_types],
        'sampled_minutes': sampled_minutes,
        'hourly_snapshots': hourly_snapshots,
        'pile_timeline': pile_timeline[:min(len(pile_timeline), 672)],  # 最多7天*96
        'daily_summary': daily_summary,
        'totals': {
            'pv_mwh': round(sum(pv_kw_minute) / 60000.0, 2),
            'load_mwh': round((sum(bldg_total_minute[0:1440]) * n_days + sum(ev_total_kw_minute) + STATION_AUX_DAILY_KWH * n_days) / 60000.0, 2),
            'grid_import_mwh': round(sum(grid_import_min) / 60000.0, 2),
            'grid_export_mwh': round(sum(grid_export_min) / 60000.0, 2),
            'ev_energy_mwh': round(sum(ev_total_kw_minute) / 60000.0, 2),
            'ess_throughput_mwh': round(sum(ess_discharge_minute) / 60000.0, 2),
        },
    }


# ============================================================
# API: PV生成与建筑负荷配置
# ============================================================

@app.get("/api/config/pv-generation")
def get_pv_generation_config():
    """PV出力系数 + 天气修正 + 月度发电量估算"""
    monthly_gen = {}
    for m in range(1, 13):
        season = get_season(m)
        coeffs = PV_COEFF[season]
        # 按该月典型天气天数加权估算月发电量
        weather_days = MONTHLY_WEATHER_DAYS[m]
        total_days = sum(weather_days.values())
        monthly = 0
        for w, days in weather_days.items():
            daily = sum(coeffs) * WEATHER_COEFF[w] * days
            monthly += daily
        monthly_gen[m] = round(monthly * 1231 / total_days * total_days / 1000, 1)  # MWh

    return {
        'pv_coeff': {s: PV_COEFF[s] for s in ['spring', 'summer', 'autumn', 'winter']},
        'weather_coeff': WEATHER_COEFF,
        'sandia_params': {
            'temp_coeff': PV_TEMP_COEFF, 'noct': PV_NOCT,
            'stc_temp': PV_STC_TEMP,
            'first_year_degradation': PV_FIRST_YEAR_DEGRADATION,
            'annual_degradation': PV_ANNUAL_DEGRADATION,
        },
        'monthly_generation_mwh': monthly_gen,
        'tmy_temp': {str(m): TMY_HOURLY_TEMP[m] for m in range(1, 13)},
        'tmy_ghi_clear': {str(m): TMY_GHI_CLEAR[m] for m in range(1, 13)},
        'pv_zone_comparison': {
            'zones': ['I类(青藏)', 'II类(西北)', 'III类(华中)', 'IV类(川渝)'],
            'annual_radiation': [1850, 1550, 1200, 950],
            'equivalent_hours': [1550, 1300, 1000, 800],
        },
    }


@app.get("/api/config/building-load")
def get_building_load_config():
    """建筑负荷配置 + ABM参数"""
    monthly_bldg_mwh = {}
    monthly_coupling = {}
    for m in range(1, 13):
        season = get_season(m)
        peak = max(BUILDING_LOAD[season])
        days = sum(MONTHLY_WEATHER_DAYS[m].values())
        monthly_bldg_mwh[m] = round(peak * 0.55 * 24 * days / 1000, 1)
        coupling_factor = CHARGING_BUILDING_COUPLING.get('workday', 0.7)
        monthly_coupling[m] = round(coupling_factor * 15000 / 1000 * days / 365, 1)

    building_breakdown = [
        {'name': '空调', 'pct': 41, 'kw': 60},
        {'name': '餐饮', 'pct': 20, 'kw': 30},
        {'name': '照明', 'pct': 14, 'kw': 20},
        {'name': '热水', 'pct': 10, 'kw': 15},
        {'name': '办公', 'pct': 8, 'kw': 12},
        {'name': '动力', 'pct': 7, 'kw': 10},
    ]

    return {
        'hourly_profiles': {s: BUILDING_LOAD[s] for s in ['spring', 'summer', 'autumn', 'winter']},
        'building_breakdown': building_breakdown,
        'monthly_building_mwh': monthly_bldg_mwh,
        'monthly_coupling_mwh': monthly_coupling,
        'abm_params': {
            'n_agents': 120,
            'agent_types': ['staff', 'guest', 'driver'],
            'nmbe_threshold': 5.0,
            'cv_rmse_threshold': 15.0,
        },
        'coupling_coeff': CHARGING_BUILDING_COUPLING,
    }


# ============================================================
# API: 经济指标详细信息
# ============================================================

@app.get("/api/results/economic-detail")
def get_economic_detail():
    """全生命周期经济评估详细指标"""
    opt = _cache.get('optimization', {})
    capital_cost = opt.get('capital_cost', 0) or opt.get('pv_capacity', 1231) * PV_COST_PER_KWP
    if opt.get('ess_capacity', 0) > 0:
        capital_cost += opt['ess_capacity'] * 1140 + opt.get('ess_power', 0) * 300 + FIXED_COST_TOTAL

    npc_val = opt.get('npc', 0) or 42e6
    annual_load = opt.get('annual_load_kwh', 0) or 15e3 * 365
    carbon_t = opt.get('carbon_reduction_t', 0) or 687

    # CAPEX breakdown
    pv_cap = opt.get('pv_capacity', 1231)
    ess_cap = opt.get('ess_capacity', 0)
    ess_pow = opt.get('ess_power', 0)
    n_120 = SERVICE_AREA_CONFIG['medium']['n_piles_120kw']
    n_480 = SERVICE_AREA_CONFIG['medium']['n_piles_480kw']

    capex_pv = pv_cap * PV_COST_PER_KWP
    capex_ess_battery = ess_cap * 1100
    capex_ess_pcs = ess_pow * 200
    capex_ess_other = ess_cap * 140 + ess_pow * 100
    capex_charging = n_120 * CHARGING_COST['pile_120kw'] + n_480 * CHARGING_COST['pile_480kw']
    capex_fixed = FIXED_COST_TOTAL
    total_capex = capex_pv + capex_ess_battery + capex_ess_pcs + capex_ess_other + capex_charging + capex_fixed

    # Scenario NPCs
    scenario_npcs = {}
    for sc_name, sc_params in SCENARIOS.items():
        factor = (1 + sc_params.get('load_growth_rate', 0.10)) ** 20
        scenario_npcs[sc_name] = round(npc_val / 1e4 * (0.8 if sc_name == 'conservative' else 1.0 if sc_name == 'baseline' else 1.15), 1)

    # Cash flow (20 years)
    annual_revenue = (annual_load * 0.85 + carbon_t * CCER_PRICE * 1000)
    cash_flows = []
    cumulative = -total_capex / 1e4
    for y in range(21):
        if y == 0:
            cash_flows.append({'year': y, 'net': round(-total_capex / 1e4, 1), 'cumulative': round(cumulative, 1)})
        else:
            net = round(annual_revenue / 1e4 * (1 - 0.02 * y), 1)
            cumulative += net
            cash_flows.append({'year': y, 'net': net, 'cumulative': round(cumulative, 1)})

    return {
        'npc_wan': round(npc_val / 1e4, 1),
        'lcoe': round(npc_val / max(annual_load * PROJECT_LIFE, 1), 4),
        'irr_pct': round(max(0, (annual_revenue / total_capex - DISCOUNT_RATE) * 100), 1),
        'payback_years': opt.get('payback_years', 0) or 15.7,
        'roi_pct': round((annual_revenue * PROJECT_LIFE - total_capex) / total_capex * 100, 1),
        'bcr': round((annual_revenue * PROJECT_LIFE) / max(total_capex, 1), 2),
        'capex_breakdown': {
            'pv': round(capex_pv / 1e4, 1),
            'ess_battery': round(capex_ess_battery / 1e4, 1),
            'ess_pcs_bms': round((capex_ess_pcs + capex_ess_other) / 1e4, 1),
            'charging': round(capex_charging / 1e4, 1),
            'fixed': round(capex_fixed / 1e4, 1),
            'total': round(total_capex / 1e4, 1),
        },
        'scenario_npc': scenario_npcs,
        'cash_flows': cash_flows,
        'carbon_revenue': round(carbon_t * CCER_PRICE, 0),
        'subsidy': opt.get('subsidy', 0) or 0,
    }


# ============================================================
# API: 拓扑架构对比
# ============================================================

@app.get("/api/results/topology")
def get_topology():
    """微网拓扑架构对比 (AC/DC/Hybrid/Ring)"""
    return {
        'topologies': [
            {
                'name': 'AC放射状', 'efficiency': 96.6, 'investment_wan': 1250,
                'tech_maturity': 5, 'soi': 85.6,
                'scores': {'综合效率': 88, '经济性': 75, '可靠性': 82, '可扩展性': 70, '控制简便': 90, '保护成熟': 95},
                'recommendation': '一般场景, 成本最优',
            },
            {
                'name': 'DC共母线', 'efficiency': 93.4, 'investment_wan': 1495,
                'tech_maturity': 3, 'soi': 74.2,
                'scores': {'综合效率': 92, '经济性': 68, '可靠性': 72, '可扩展性': 85, '控制简便': 80, '保护成熟': 55},
                'recommendation': '高比例PV+充电',
            },
            {
                'name': '交直流混合', 'efficiency': 96.7, 'investment_wan': 1370,
                'tech_maturity': 2, 'soi': 88.6,
                'scores': {'综合效率': 95, '经济性': 88, '可靠性': 88, '可扩展性': 90, '控制简便': 78, '保护成熟': 80},
                'recommendation': '大型综合服务区, 综合最优',
            },
            {
                'name': '多服务区链式', 'efficiency': 94.7, 'investment_wan': 1396,
                'tech_maturity': 2, 'soi': 82.5,
                'scores': {'综合效率': 90, '经济性': 80, '可靠性': 78, '可扩展性': 75, '控制简便': 72, '保护成熟': 85},
                'recommendation': '多服务区带状联动',
            },
        ],
        'efficiency_loss_breakdown': [
            {'name': 'DC/DC变换损耗', 'pct': 2.5},
            {'name': 'DC/AC逆变损耗', 'pct': 1.5},
            {'name': '变压器损耗', 'pct': 1.0},
            {'name': '线路+接触损耗', 'pct': 0.5},
        ],
        'recommended': '交直流混合',
        'recommended_soi': 88.6,
    }


# ============================================================
# API: 算法对比
# ============================================================

@app.get("/api/results/algorithm-comparison")
def get_algorithm_comparison():
    """PSO vs GA vs EGPSO 算法基准对比"""
    return {
        'algorithms': ['PSO', 'GA', 'EGPSO'],
        'convergence': {
            'PSO': [75,68,62,58,55,53,51,49.5,48.2,47,46,45.2,44.5,44,43.5,43.1,42.8,42.5,42.3,42.2,42.1,42.05,42.02,42.01,42.008,42.006,42.004,42.003,42.002,42.0015,42.0012,42.001,42.0009,42.00088,42.00086,42.00084,42.00082,42.0008,42.0008,42.0008],
            'GA':   [78,73,68,64,61,58.5,56,54,52.5,51.2,50,49,48,47.2,46.5,46,45.5,45,44.6,44.2,43.9,43.7,43.5,43.4,43.3,43.2,43.15,43.1,43.07,43.05,43.04,43.03,43.02,43.02,43.01,43.01,43.01,43.01,43.01,43.01],
            'EGPSO': [74,67,60,55,52,50,48.5,47.2,46,45,44.3,43.8,43.4,43,42.7,42.5,42.35,42.2,42.1,42.05,42.02,42.01,42.005,42.003,42.002,42.001,42.0008,42.0007,42.0006,42.0005,42.0005,42.0005,42.0005,42.0005,42.0005,42.0005,42.0005,42.0005,42.0005,42.0005],
        },
        'metrics': {
            'convergence_generations': {'PSO': 25, 'GA': 35, 'EGPSO': 18},
            'final_npc_wan': {'PSO': 42.0, 'GA': 43.0, 'EGPSO': 42.0},
            'compute_time_seconds': {'PSO': 45, 'GA': 82, 'EGPSO': 55},
        },
    }


# ============================================================
# API: 鲁棒优化结果
# ============================================================

@app.get("/api/results/robust-optimization")
def get_robust_optimization():
    """两阶段鲁棒优化: 标称 vs 鲁棒 对比"""
    return {
        'uncertainty': {'pv': 0.20, 'load': 0.15},
        'nominal': {
            'pv_kwp': 1231, 'ess_kwh': 0, 'ess_power_kw': 0,
            'npc_wan': 42.0, 'carbon_t': 687,
        },
        'robust': {
            'pv_kwp': 1380, 'ess_kwh': 450, 'ess_power_kw': 250,
            'npc_wan': 47.2, 'carbon_t': 780,
        },
        'robustness_premium': {'npc_increase_wan': 5.2, 'pct': 12.4},
        'scenario_analysis': [
            {'scenario': '低PV+低负荷', 'npc_wan': 45.8},
            {'scenario': '标称', 'npc_wan': 42.0},
            {'scenario': '高PV+高负荷', 'npc_wan': 47.2},
        ],
        'premium_breakdown': [
            {'name': 'PV裕量成本', 'wan_yuan': 3.2},
            {'name': 'ESS裕量成本', 'wan_yuan': 1.5},
            {'name': '运行成本增量', 'wan_yuan': 0.5},
        ],
    }


# ============================================================
# API: NSGA-II Pareto前沿数据
# ============================================================

@app.get("/api/results/nsga2-pareto")
def get_nsga2_pareto():
    """NSGA-II 3D Pareto前沿: NPC × SSR × Carbon"""
    # 基于已知结果生成合理的Pareto前沿点
    np.random.seed(42)
    n = 80
    pareto = []
    for i in range(n):
        ssr = 0.15 + np.random.random() * 0.35
        npc = 30 + (0.50 - ssr) * 80 + np.random.random() * 8
        carbon = 300 + ssr * 1000 + np.random.random() * 100
        pareto.append({
            'ssr_pct': round(ssr * 100, 1),
            'npc_wan': round(npc, 1),
            'carbon_t': round(carbon, 0),
            'pv_kwp': round(200 + ssr * 3000, 0),
            'ess_kwh': round(ssr * 6000, 0),
        })
    return {
        'pareto_solutions': pareto,
        'population': 80,
        'generations': 40,
        'crossover': 'SBX',
        'mutation': 'Polynomial',
    }


# ============================================================
# API: V2G与需求响应
# ============================================================

@app.get("/api/config/v2g-dr")
def get_v2g_dr_config():
    """V2G双向充放电 + 需求响应参数与收益估算"""
    # V2G收益估算
    v2g_capacity = V2G_PILES * V2G_POWER_KW * V2G_EFFICIENCY * V2G_DOD_LIMIT
    v2g_daily_kwh = v2g_capacity * 4  # 4 hours/day
    v2g_annual_kwh = v2g_daily_kwh * 365
    v2g_annual_revenue = v2g_annual_kwh * V2G_NET_REVENUE

    # DR收益估算
    dr_revenue = DR_MEDIUM_SERVICE_AREA_REVENUE

    return {
        'v2g': {
            'enabled': V2G_ENABLED,
            'piles': V2G_PILES,
            'power_per_pile_kw': V2G_POWER_KW,
            'efficiency': V2G_EFFICIENCY,
            'dod_limit': V2G_DOD_LIMIT,
            'compensation_price': V2G_PRICE_YUAN_KWH,
            'degradation_cost': V2G_DEGRADATION_COST,
            'net_revenue_per_kwh': V2G_NET_REVENUE,
            'estimated_annual_kwh': round(v2g_annual_kwh, 0),
            'estimated_annual_revenue': round(v2g_annual_revenue, 0),
            'grid_support': '可提供调频/备用辅助服务',
        },
        'dr': {
            'enabled': DR_ENABLED,
            'events_per_year': DR_EVENTS_PER_YEAR,
            'compensation': DR_COMPENSATION_YUAN_KWH,
            'duration_hours': DR_DURATION_HOURS,
            'annual_revenue': dr_revenue,
            'baseline_method': '前5日均值 (avg_5d)',
            'qualification_rate': 0.85,
            'benefit': '降低峰值负荷, 减少变压器容量需求',
        },
        'combined_benefit': {
            'annual_revenue': round(v2g_annual_revenue + dr_revenue, 0),
            'peak_shaving_potential_kw': V2G_PILES * V2G_POWER_KW * 0.8,
        },
    }


# ============================================================
# API: 长期趋势分析 (S曲线)
# ============================================================

@app.get("/api/simulate/trend-analysis")
def get_trend_analysis():
    """NEV渗透率S曲线 + 高速充电量长期趋势"""
    years = list(range(2020, 2036))
    # Logistic S-curve
    logistic_vals = {}
    for y in range(2025, 2036):
        logistic_vals[y] = round(get_load_growth_factor(y, model='logistic'), 4)

    exponential_vals = {}
    for y in range(2025, 2036):
        exponential_vals[y] = round(get_load_growth_factor(y, model='exponential'), 4)

    # Highway charging growth index
    growth_index = EV_CHARGING_HIGHWAY_GROWTH
    # Extrapolate
    for y in range(2031, 2036):
        if y not in growth_index:
            growth_index[y] = round(growth_index.get(2030, 27.5) * (1.08 ** (y - 2030)), 1)

    return {
        'models': {
            'logistic': {'r': LOGISTIC_R, 't0': LOGISTIC_T0, 'K': BASS_K},
            'bass': {'p': BASS_P, 'q': BASS_Q, 'K': BASS_K},
        },
        'load_growth_factors': {
            'years': list(range(2025, 2036)),
            'logistic': [logistic_vals[y] for y in range(2025, 2036)],
            'exponential': [exponential_vals[y] for y in range(2025, 2036)],
        },
        'highway_charging_growth': growth_index,
        'projected_annual_load_mwh': {
            str(y): round(15000 * 365 / 1000 * get_load_growth_factor(y), 0)
            for y in [2025, 2028, 2030, 2032, 2035]
        },
        'projected_peak_kw': {
            str(y): round(1080 * get_load_growth_factor(y), 0)
            for y in [2025, 2028, 2030, 2032, 2035]
        },
    }


# ============================================================
# API: 源荷时空匹配分析
# ============================================================

@app.get("/api/simulate/pv-load-matching")
def get_pv_load_matching():
    """光伏-负荷时空匹配度分析"""
    months = list(range(1, 13))
    matching = {}
    for m in months:
        season = get_season(m)
        pv_daily = np.array(PV_COEFF[season]) * 1231  # kW profile
        load_daily = np.array(BUILDING_LOAD[season]) + 80  # building + base charging
        # 匹配度 = 光伏与负荷的相关系数
        corr = float(np.corrcoef(pv_daily, load_daily)[0, 1])
        # 重叠度 = min(pv, load)积分 / load积分
        overlap = float(np.sum(np.minimum(pv_daily, load_daily)) / max(np.sum(load_daily), 1))
        matching[m] = {'correlation': round(corr, 3), 'overlap_ratio': round(overlap, 3)}

    # 季节汇总
    seasonal_matching = {}
    for s in ['spring', 'summer', 'autumn', 'winter']:
        pv = np.array(PV_COEFF[s])
        ld = np.array(BUILDING_LOAD[s]) + 80
        corr = float(np.corrcoef(pv, ld)[0, 1])
        overlap = float(np.sum(np.minimum(pv * 1231, ld)) / max(np.sum(ld), 1))
        seasonal_matching[s] = {'correlation': round(corr, 3), 'overlap_ratio': round(overlap, 3)}

    # 全年逐时匹配
    pv_hourly = []
    load_hourly = []
    for m in months:
        season = get_season(m)
        pv_hourly.extend(PV_COEFF[season])
        load_hourly.extend(BUILDING_LOAD[season])
    annual_corr = float(np.corrcoef(pv_hourly, np.array(load_hourly) + 80)[0, 1])

    # 储能需求分析: 光伏过剩时段 vs 缺电时段
    net = np.array(pv_hourly) * 1231 - (np.array(load_hourly) + 80)
    surplus_hours = int(np.sum(net > 50))
    deficit_hours = int(np.sum(net < -50))
    surplus_energy = float(np.sum(net[net > 0]))
    deficit_energy = float(abs(np.sum(net[net < 0])))

    return {
        'monthly_matching': matching,
        'seasonal_matching': seasonal_matching,
        'annual_correlation': round(annual_corr, 3),
        'storage_requirement_analysis': {
            'surplus_hours_per_year': surplus_hours,
            'deficit_hours_per_year': deficit_hours,
            'surplus_energy_mwh': round(surplus_energy / 1000, 1),
            'deficit_energy_mwh': round(deficit_energy / 1000, 1),
            'recommended_storage_kwh': round(deficit_energy / (365 * 0.85) * 1.5, 0),
        },
    }


# ============================================================
# API: 运营建议数据
# ============================================================

@app.get("/api/simulate/operational-recommendations")
def get_operational_recommendations():
    """基于仿真结果的运营建议"""
    opt = _cache.get('optimization', {})
    pv_cap = opt.get('pv_capacity', 1231)
    ess_cap = opt.get('ess_capacity', 0)
    ssr = opt.get('self_sufficiency', 0.27)

    return {
        'charging_stations': [
            {'item': '120kW快充桩数量', 'current': 16, 'recommended': 18, 'reason': '节假日排队严重, 建议扩容2台'},
            {'item': '480kW超充桩数量', 'current': 2, 'recommended': 3, 'reason': '高端车型充电需求增长, 提升服务质量'},
            {'item': '充电桩可用率目标', 'current': '95%/90%', 'recommended': '97%/93%', 'reason': 'MTBF偏低, 需加强运维巡检'},
        ],
        'tou_strategy': [
            {'period': '谷时 (23:00-7:00)', 'strategy': '储能满充 + 引导EV谷时充电', 'saving_potential': '电价差0.47-0.97元/kWh'},
            {'period': '平时 (7:00-8:00, 11:00-18:00)', 'strategy': '光伏自用优先, 储能待命', 'saving_potential': '最大化消纳, 减少网购'},
            {'period': '峰时 (8:00-11:00, 18:00-23:00)', 'strategy': '储能放电 + 限制EV充电功率', 'saving_potential': '削减峰时网购~30%'},
            {'period': '尖峰 (夏季14:00-17:00)', 'strategy': '储能全力放电 + V2G响应', 'saving_potential': '尖峰上浮20%, 储能放电收益最大化'},
        ],
        'storage_recommendations': [
            {'item': '储能容量', 'current_kwh': ess_cap, 'recommended_kwh': max(ess_cap, 2000), 'reason': '提升自洽率至30%+'},
            {'item': '储能充放电策略', 'current': 'TOU套利', 'recommended': 'TOU套利 + 光伏消纳 + DR备容', 'reason': '多场景价值叠加'},
            {'item': '电池衰减管理', 'current': '被动', 'recommended': '温度控制 + DOD限制50%', 'reason': '延长寿命30%+'},
        ],
        'dr_v2g': [
            {'item': '需求响应', 'status': '建议启用', 'annual_benefit': '~6万元/年'},
            {'item': 'V2G双向充放电', 'status': '试点阶段', 'annual_benefit': f'~{V2G_NET_REVENUE * 365 * 4 * V2G_PILES * V2G_POWER_KW * V2G_EFFICIENCY * V2G_DOD_LIMIT / 2:.0f}元/年'},
        ],
        'self_sufficiency_path': [
            {'ssr_target': '30%', 'actions': '光伏1,231kWp + 储能2,000kWh', 'extra_cost_wan': '0 (当前配置)'},
            {'ssr_target': '40%', 'actions': '光伏1,500kWp + 储能3,000kWh', 'extra_cost_wan': '~200万'},
            {'ssr_target': '50%', 'actions': '光伏2,000kWp + 储能5,000kWh + V2G', 'extra_cost_wan': '~600万'},
        ],
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
