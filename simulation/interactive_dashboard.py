"""
交互式可视化仪表盘 — 基于Plotly的HTML看板
生成包含年度热力图、每日调度、月度平衡的交互式图表
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    SERVICE_AREA_CONFIG, PV_COEFF, WEATHER_COEFF, MONTHLY_WEATHER_DAYS,
    BUILDING_LOAD, WEATHER_CHARGING_COEFF, WEATHER_BUILDING_COEFF,
    HOLIDAY_TOU_FACTOR, TOU_PRICE_VALUES, TOU_PRICE, FEED_IN_PRICE,
    SOC_MIN, SOC_MAX, CARBON_FACTOR_GRID, STATION_AUX_DAILY_KWH,
    DAY_TYPE_COEFF, get_season, get_tou_price_array,
)
from calendar_utils import CalendarContext
from capacity_optimization import MicrogridOptimizer, BUILDING_DAY_TYPE_COEFF
from mc_charging_load import MonteCarloChargingSimulator

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
SEED = 42


def build_8760h_data(mc_scenarios, pv_cap=1231, ess_cap=2000, ess_pow=1000,
                      opt=None, calendar_ctx=None):
    """构建全年8760h仿真数据

    Parameters
    ----------
    opt : MicrogridOptimizer or None
        已有优化器实例, 传入则复用其 calendar_ctx.
    calendar_ctx : CalendarContext or None
        若 opt 和 calendar_ctx 都未传入, 则创建新的 CalendarContext.
    """
    if opt is not None:
        calendar_ctx = getattr(opt, 'calendar_ctx', calendar_ctx)
    if calendar_ctx is None:
        calendar_ctx = CalendarContext(seed=SEED)
    if opt is None:
        opt = MicrogridOptimizer(size='medium', mc_scenarios=mc_scenarios, seed=SEED,
                                  calendar_ctx=calendar_ctx)
    opt.pv_capacity = pv_cap
    opt.ess_capacity = ess_cap
    opt.ess_power = ess_pow

    pv_coeff_seq, load_seq, tou_seq, seasons_seq, _ = opt._build_8760h_sequence()

    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # 逐日仿真
    daily_results = []
    total_grid_import = 0
    total_grid_export = 0
    total_pv_gen = 0
    total_load = 0
    total_self_used = 0
    total_grid_cost = 0
    total_ess_cycles = 0
    total_loss_of_load = 0

    for d in range(365):
        season = seasons_seq[d * 24]
        pv_profile = pv_coeff_seq[d * 24:(d + 1) * 24] * pv_cap
        load_profile = load_seq[d * 24:(d + 1) * 24]
        tou_hourly = tou_seq[d * 24:(d + 1) * 24]

        result = opt.simulate_daily_operation(pv_profile, load_profile, season,
                                               tou_prices=tou_hourly)

        daily_results.append({
            'day': d,
            'month': calendar_ctx.month_of_day[d],
            'day_type': calendar_ctx.day_types[d],
            'weather': calendar_ctx.weather_seq[d],
            'season': season,
            'pv_gen': pv_profile.sum(),
            'load': load_profile.sum(),
            'grid_import': result['grid_import'].sum(),
            'grid_export': result['grid_export'].sum(),
            'grid_cost': float(np.sum(result['grid_import'] * tou_hourly)),
            'soc_curve': result['soc_curve'].copy(),
            'ess_ch': result['ess_charge'].sum(),
            'ess_disch': result['ess_discharge'].sum(),
            'net_load_hourly': load_profile - pv_profile,
            'loss_of_load': result['loss_of_load'].sum(),
        })

        total_grid_import += result['grid_import'].sum()
        total_grid_export += result['grid_export'].sum()
        total_pv_gen += pv_profile.sum()
        total_load += load_profile.sum()
        total_self_used += pv_profile.sum() - result['grid_export'].sum()
        total_grid_cost += float(np.sum(result['grid_import'] * tou_hourly))
        total_grid_cost -= result['grid_export'].sum() * FEED_IN_PRICE
        total_loss_of_load += result['loss_of_load'].sum()
        if ess_cap > 0:
            total_ess_cycles += result['ess_discharge'].sum() / ess_cap

    station_aux = STATION_AUX_DAILY_KWH * 365
    total_grid_import += station_aux
    flat_price = np.mean([v['flat'] for v in TOU_PRICE_VALUES.values()])
    total_grid_cost += station_aux * flat_price

    summary = {
        'pv_cap': pv_cap, 'ess_cap': ess_cap, 'ess_pow': ess_pow,
        'total_pv_gen': total_pv_gen, 'total_load': total_load,
        'total_grid_import': total_grid_import, 'total_grid_export': total_grid_export,
        'total_self_used': total_self_used,
        'self_sufficiency': total_self_used / max(total_load, 1),
        'grid_cost': total_grid_cost,
        'ess_cycles': total_ess_cycles,
        'loss_of_load': total_loss_of_load,
    }

    return daily_results, pv_coeff_seq, load_seq, tou_seq, calendar_ctx, summary


def create_dashboard(daily_results, pv_coeff_seq, load_seq, tou_seq, calendar_ctx, summary,
                     pv_cap, ess_cap, ess_pow):
    """生成交互式仪表盘HTML"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_labels_cn = ['1月', '2月', '3月', '4月', '5月', '6月',
                       '7月', '8月', '9月', '10月', '11月', '12月']

    # ---- Heatmap data ----
    heatmap_data = np.zeros((365, 24))
    for d in range(365):
        net_load = daily_results[d]['net_load_hourly']
        heatmap_data[d, :] = net_load

    vmax = max(abs(heatmap_data.min()), abs(heatmap_data.max()))

    # Y-axis tick positions (first day of each month)
    month_starts = [sum(days_in_month[:i]) for i in range(12)]

    # ---- Monthly aggregation ----
    monthly_pv = np.zeros(12)
    monthly_load = np.zeros(12)
    monthly_grid_import = np.zeros(12)
    monthly_grid_export = np.zeros(12)
    for d in range(365):
        m = daily_results[d]['month'] - 1
        monthly_pv[m] += daily_results[d]['pv_gen']
        monthly_load[m] += daily_results[d]['load']
        monthly_grid_import[m] += daily_results[d]['grid_import']
        monthly_grid_export[m] += daily_results[d]['grid_export']

    # ---- Build daily operation for 4 seasonal typical days ----
    season_days = {'spring': None, 'summer': None, 'autumn': None, 'winter': None}
    for d in range(365):
        s = daily_results[d]['season']
        if season_days[s] is None and daily_results[d]['weather'] == 'clear' and \
           daily_results[d]['day_type'] == 'workday':
            season_days[s] = d

    # ---- Create HTML with multiple sections ----
    html_parts = []

    # CSS
    html_parts.append('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>高速服务区光储充微网 — 交互式运行仪表盘</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif; background: #0f1923; color: #e0e0e0; }
  .header { background: linear-gradient(135deg, #1a3a4a 0%, #0d2137 100%); padding: 24px 40px;
            border-bottom: 2px solid #2a5a7a; }
  .header h1 { font-size: 24px; color: #4fc3f7; margin-bottom: 4px; }
  .header .subtitle { font-size: 13px; color: #90a4ae; }
  .metrics-row { display: flex; gap: 16px; padding: 20px 40px; flex-wrap: wrap; }
  .metric-card { background: #1a2d3d; border: 1px solid #2a4a5a; border-radius: 10px;
                 padding: 16px 20px; min-width: 140px; flex: 1; }
  .metric-card .label { font-size: 11px; color: #78909c; text-transform: uppercase; letter-spacing: 1px; }
  .metric-card .value { font-size: 28px; font-weight: 700; color: #4fc3f7; margin-top: 4px; }
  .metric-card .unit { font-size: 12px; color: #78909c; }
  .metric-card.accent .value { color: #ffb74d; }
  .metric-card.green .value { color: #81c784; }
  .metric-card.warn .value { color: #ef5350; }
  .section { padding: 0 40px 24px; }
  .section-title { font-size: 16px; color: #b0bec5; padding: 8px 0 12px; border-bottom: 1px solid #1a3a4a;
                   margin-bottom: 12px; }
  .chart-container { background: #152838; border: 1px solid #1a3a4a; border-radius: 10px;
                     padding: 12px; margin-bottom: 20px; }
  .footer { text-align: center; padding: 20px; color: #546e7a; font-size: 11px; }
</style>
</head>
<body>
<div class="header">
  <h1>高速公路服务区光储充一体化微网 — 交互式运行仪表盘</h1>
  <div class="subtitle">中型服务区 | 光伏 ''' + f'{pv_cap:.0f} kWp' + ''' | 储能 ''' + f'{ess_cap:.0f} kWh' + ''' | 8760h全时序仿真</div>
</div>
''')

    # Metrics row
    ssr = summary['self_sufficiency']
    html_parts.append(f'''<div class="metrics-row">
  <div class="metric-card"><div class="label">新能源自洽率</div><div class="value">{ssr:.1%}</div><div class="unit">全年平均</div></div>
  <div class="metric-card accent"><div class="label">年光伏发电</div><div class="value">{summary['total_pv_gen']/1000:,.0f}</div><div class="unit">MWh</div></div>
  <div class="metric-card"><div class="label">年总用电</div><div class="value">{summary['total_load']/1000:,.0f}</div><div class="unit">MWh</div></div>
  <div class="metric-card green"><div class="label">年网购电</div><div class="value">{summary['total_grid_import']/1000:,.0f}</div><div class="unit">MWh</div></div>
  <div class="metric-card accent"><div class="label">年网购电成本</div><div class="value">{summary['grid_cost']/1e4:.1f}</div><div class="unit">万元</div></div>
  <div class="metric-card"><div class="label">ESS年等效循环</div><div class="value">{summary['ess_cycles']:.0f}</div><div class="unit">次</div></div>
</div>''')

    html_parts.append('<div class="section"><div class="section-title">年度运行热力图 (8760h) — 横轴: 小时(0-23) | 纵轴: 天数(1-365) | 颜色: 净负荷 (正=缺电, 负=余电)</div>')
    html_parts.append('<div class="chart-container"><div id="heatmap"></div></div>')

    html_parts.append('<div class="section"><div class="section-title">四季典型日调度对比 (工作日/晴天) — 含分时电价曲线与储能SOC</div>')
    html_parts.append('<div class="chart-container"><div id="seasonal"></div></div>')

    html_parts.append('<div class="section"><div class="section-title">月度能量平衡</div>')
    html_parts.append('<div class="chart-container"><div id="monthly"></div></div>')

    html_parts.append('<div class="section"><div class="section-title">日类型负荷与电网成本分布</div>')
    html_parts.append('<div class="chart-container"><div id="daytype"></div></div>')

    html_parts.append('<div class="footer">高速公路服务区光储充一体化仿真系统 | 基于Monte Carlo充电负荷预测 + PSO容量优化 + 8760h验证</div>')
    html_parts.append('</body>')

    # ---- Plotly charts as embedded JS ----
    html_parts.append('<script>')

    # Chart 1: Annual heatmap
    html_parts.append(f'''
// --- Annual heatmap ---
(function() {{
  var data = [{{
    z: {heatmap_data.tolist()},
    type: 'heatmap',
    colorscale: [
      [0.0, '#1565c0'], [0.25, '#42a5f5'], [0.45, '#e3f2fd'],
      [0.5, '#ffffff'], [0.55, '#fff3e0'],
      [0.75, '#ff9800'], [1.0, '#d84315']
    ],
    zmin: -{vmax:.0f}, zmax: {vmax:.0f},
    colorbar: {{ title: {{ text: '净负荷 (kW)' }}, titleside: 'right' }},
    hovertemplate: 'Day %{{y}} Hr %{{x}}:00<br>Net Load: %{{z:.0f}} kW<extra></extra>',
  }}];
  var layout = {{
    title: '年度8760h净负荷热力图 (PV出力 - 总用电负荷)',
    xaxis: {{ title: '小时', dtick: 2, side: 'top' }},
    yaxis: {{
      title: '天数', autorange: 'reversed',
      tickvals: {month_starts},
      ticktext: {month_labels_cn},
      dtick: 1,
    }},
    height: 650,
    paper_bgcolor: '#152838', plot_bgcolor: '#152838',
    font: {{ color: '#b0bec5' }},
    margin: {{ l: 60, r: 40, t: 40, b: 40 }},
  }};
  var config = {{ responsive: true, displayModeBar: false }};
  Plotly.newPlot('heatmap', data, layout, config);
}})();
''')

    # Chart 2: Seasonal typical day comparison
    season_names = {'spring': '春季(4月)', 'summer': '夏季(7月)', 'autumn': '秋季(10月)', 'winter': '冬季(1月)'}
    season_colors = {'spring': '#81c784', 'summer': '#ef5350', 'autumn': '#ffb74d', 'winter': '#64b5f6'}

    seasonal_traces = []
    for si, (season, d) in enumerate(season_days.items()):
        if d is None:
            continue
        hours = list(range(24))
        dr = daily_results[d]
        pv_hourly = np.array(PV_COEFF[season]) * pv_cap * WEATHER_COEFF.get(dr['weather'], 1.0)
        season_idx = d
        pv_coeff = np.array(PV_COEFF[season]) * WEATHER_COEFF.get(dr['weather'], 1.0)
        load_hourly = load_seq[season_idx*24:(season_idx+1)*24].tolist()
        net_hourly = (np.array(pv_hourly) - np.array(load_hourly)).tolist()
        soc = dr['soc_curve'].tolist()
        tou_h = tou_seq[season_idx*24:(season_idx+1)*24].tolist()

        # PV trace
        seasonal_traces.append(f'''
  {{
    x: {hours}, y: {pv_hourly.tolist()},
    type: 'scatter', mode: 'none', fill: 'tozeroy', fillcolor: 'rgba(255,183,77,0.3)',
    name: '{season_names[season]} 光伏', legendgroup: '{season}',
    xaxis: 'x{si+1}', yaxis: 'y{si+1}',
    hovertemplate: 'Hr %{{x}}: PV=%{{y:.0f}} kW'
  }}''')

        # Load trace
        seasonal_traces.append(f'''
  {{
    x: {hours}, y: {load_hourly},
    type: 'scatter', mode: 'none', fill: 'tozeroy', fillcolor: 'rgba(239,83,80,0.25)',
    name: '{season_names[season]} 负荷', legendgroup: '{season}',
    xaxis: 'x{si+1}', yaxis: 'y{si+1}',
    hovertemplate: 'Hr %{{x}}: Load=%{{y:.0f}} kW'
  }}''')

        # SOC trace
        seasonal_traces.append(f'''
  {{
    x: {hours}, y: {soc},
    type: 'scatter', mode: 'lines+markers', line: {{ color: '#4fc3f7', width: 2 }},
    marker: {{ size: 4 }},
    name: '{season_names[season]} SOC', legendgroup: '{season}',
    xaxis: 'x{si+1}', yaxis: 'y{si+1}',
    yaxis: 'y{si+1}',
    hovertemplate: 'Hr %{{x}}: SOC=%{{y:.1%}}'
  }}''')

        # TOU price (secondary y)
        seasonal_traces.append(f'''
  {{
    x: {hours}, y: {tou_h},
    type: 'scatter', mode: 'lines', line: {{ color: '#ce93d8', width: 1.5, dash: 'dot' }},
    name: 'TOU电价', legendgroup: 'tou{si}',
    xaxis: 'x{si+1}', yaxis: 'y{si+1}_2',
    hovertemplate: 'Hr %{{x}}: TOU=%{{y:.3f}} yuan/kWh'
  }}''')

    # Build axis dicts for each subplot
    domain_positions = [
        [0.0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]
    ]
    axis_configs = []
    for si in range(4):
        x_dom = domain_positions[si]
        axis_configs.append(f'''
    xaxis{si+1}: {{ domain: {x_dom}, title: '小时', dtick: 4 }},
    yaxis{si+1}: {{ title: '功率 (kW)', side: 'left' }},
    yaxis{si+1}_2: {{ title: '电价 (元/kWh)', side: 'right', overlaying: 'y{si+1}',
                     range: [0, 1.6], showgrid: false }}''')

    axis_str = ','.join(axis_configs)

    html_parts.append(f'''
// --- Seasonal dispatch ---
(function() {{
  var data = [{','.join(seasonal_traces)}];
  var layout = {{
    title: '四季典型日调度对比 (工作日/晴天)',
    {axis_str},
    grid: {{ rows: 1, columns: 4, pattern: 'independent' }},
    height: 500,
    paper_bgcolor: '#152838', plot_bgcolor: '#152838',
    font: {{ color: '#b0bec5' }},
    margin: {{ l: 50, r: 50, t: 50, b: 40 }},
    showlegend: false,
  }};
  Plotly.newPlot('seasonal', data, layout, {{ responsive: true, displayModeBar: false }});
}})();
''')

    # Chart 3: Monthly energy balance
    html_parts.append(f'''
// --- Monthly balance ---
(function() {{
  var pvBar = {{
    x: {month_labels_cn}, y: {(monthly_pv/1000).tolist()},
    type: 'bar', name: '光伏发电', marker: {{ color: '#ffb74d' }},
    hovertemplate: '%{{x}}: PV=%{{y:.0f}} MWh'
  }};
  var loadBar = {{
    x: {month_labels_cn}, y: {(monthly_load/1000).tolist()},
    type: 'bar', name: '总用电', marker: {{ color: '#ef5350' }},
    hovertemplate: '%{{x}}: Load=%{{y:.0f}} MWh'
  }};
  var gridBar = {{
    x: {month_labels_cn}, y: {(monthly_grid_import/1000).tolist()},
    type: 'bar', name: '网购电', marker: {{ color: '#42a5f5' }},
    hovertemplate: '%{{x}}: Grid=%{{y:.0f}} MWh'
  }};
  var ssrLine = {{
    x: {month_labels_cn},
    y: {(np.divide(monthly_pv, monthly_load, where=(monthly_load>0), out=np.zeros(12))*100).tolist()},
    type: 'scatter', mode: 'lines+markers', name: '自洽率(%)', yaxis: 'y2',
    line: {{ color: '#81c784', width: 3 }}, marker: {{ size: 8 }},
    hovertemplate: '%{{x}}: SSR=%{{y:.1f}}%'
  }};
  var layout = {{
    title: '月度能量平衡与自洽率',
    barmode: 'group',
    xaxis: {{ title: '' }},
    yaxis: {{ title: '电量 (MWh)', side: 'left' }},
    yaxis2: {{ title: '自洽率 (%)', side: 'right', overlaying: 'y', range: [0, 50], showgrid: false }},
    height: 450,
    paper_bgcolor: '#152838', plot_bgcolor: '#152838',
    font: {{ color: '#b0bec5' }},
    legend: {{ x: 0.01, y: 0.99 }},
  }};
  Plotly.newPlot('monthly', [pvBar, loadBar, gridBar, ssrLine], layout, {{ responsive: true, displayModeBar: false }});
}})();
''')

    # Chart 4: Day type cost distribution
    day_type_categories = ['workday', 'weekend', 'holiday', 'spring_festival']
    day_type_labels = ['工作日', '双休日', '节假日', '春运']
    dt_grid_costs = {dt: [] for dt in day_type_categories}
    dt_loads = {dt: [] for dt in day_type_categories}
    for d in range(365):
        dt = daily_results[d]['day_type']
        if dt in dt_grid_costs:
            dt_grid_costs[dt].append(daily_results[d]['grid_cost'])
            dt_loads[dt].append(daily_results[d]['load'])

    dt_cost_mean = [np.mean(dt_grid_costs[dt]) if dt_grid_costs[dt] else 0 for dt in day_type_categories]
    dt_load_mean = [np.mean(dt_loads[dt]) if dt_loads[dt] else 0 for dt in day_type_categories]

    html_parts.append(f'''
// --- Day type analysis ---
(function() {{
  var costBar = {{
    x: {day_type_labels}, y: {dt_cost_mean},
    type: 'bar', name: '日均网购电成本', marker: {{ color: '#ef5350' }},
    text: {[f'{v:.0f}' for v in dt_cost_mean]}, textposition: 'outside',
    hovertemplate: '%{{x}}: %{{y:.0f}} yuan'
  }};
  var loadBar2 = {{
    x: {day_type_labels}, y: {dt_load_mean},
    type: 'bar', name: '日均用电负荷', marker: {{ color: '#42a5f5' }}, yaxis: 'y2',
    text: {[f'{v:.0f}' for v in dt_load_mean]}, textposition: 'outside',
    hovertemplate: '%{{x}}: %{{y:.0f}} kWh'
  }};
  var layout = {{
    title: '不同日类型 — 日均用电负荷与网购电成本',
    barmode: 'group',
    xaxis: {{ title: '' }},
    yaxis: {{ title: '日均网购电成本 (元)', side: 'left' }},
    yaxis2: {{ title: '日均用电 (kWh)', side: 'right', overlaying: 'y', showgrid: false }},
    height: 400,
    paper_bgcolor: '#152838', plot_bgcolor: '#152838',
    font: {{ color: '#b0bec5' }},
    legend: {{ x: 0.01, y: 0.99 }},
  }};
  Plotly.newPlot('daytype', [costBar, loadBar2], layout, {{ responsive: true, displayModeBar: false }});
}})();
''')

    html_parts.append('</script></html>')

    html_content = '\n'.join(html_parts)
    output_path = os.path.join(OUTPUT_DIR, 'interactive_dashboard.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Dashboard saved to: {os.path.abspath(output_path)}")
    return output_path


def main():
    print("=" * 55)
    print("Building Interactive Dashboard...")
    print("=" * 55)

    # 配置 (基于v4最优结果)
    pv_cap = 1231   # kWp
    ess_cap = 2000  # kWh
    ess_pow = 1000  # kW

    print(f"Config: PV={pv_cap}kWp, ESS={ess_cap}kWh/{ess_pow}kW")

    # Step 1: Monte Carlo (快速模式)
    print("\n[1/3] Running Monte Carlo simulation (2000 runs)...")
    sim = MonteCarloChargingSimulator(service_area_size='medium', seed=SEED)
    mc_scenarios = sim.simulate_all_scenarios(n_runs=2000)

    # Step 2: Build 8760h data
    print("\n[2/3] Building 8760h simulation data...")
    (daily_results, pv_coeff_seq, load_seq, tou_seq,
     calendar_ctx, summary) = build_8760h_data(mc_scenarios, pv_cap, ess_cap, ess_pow)

    print(f"  Self-sufficiency: {summary['self_sufficiency']:.1%}")
    print(f"  Annual PV Gen: {summary['total_pv_gen']/1000:.0f} MWh")
    print(f"  Annual Load: {summary['total_load']/1000:.0f} MWh")
    print(f"  Grid Cost: {summary['grid_cost']/1e4:.1f} wan yuan")
    print(f"  ESS Cycles: {summary['ess_cycles']:.0f}")

    # Step 3: Generate dashboard
    print("\n[3/3] Generating interactive dashboard HTML...")
    output = create_dashboard(daily_results, pv_coeff_seq, load_seq, tou_seq,
                               calendar_ctx, summary, pv_cap, ess_cap, ess_pow)

    print(f"\nOpen in browser: {output}")


if __name__ == '__main__':
    main()
