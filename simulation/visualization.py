"""
可视化模块 — 生成课程报告用图表
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')

# 供 v6 图表使用的配置常量
_FIXED_COST_TOTAL = None


def _get_fixed_cost():
    global _FIXED_COST_TOTAL
    if _FIXED_COST_TOTAL is None:
        from config import FIXED_COST_TOTAL
        _FIXED_COST_TOTAL = FIXED_COST_TOTAL
    return _FIXED_COST_TOTAL


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig1_charging_load_probability(mc_scenarios, save=True):
    """图1: 充电负荷概率分布曲线 (工作日)"""
    ensure_output_dir()
    result = mc_scenarios['workday']

    fig, ax = plt.subplots(figsize=(10, 5))
    hours = np.arange(24)

    ax.fill_between(hours, result['p5'], result['p95'], alpha=0.2,
                    color='steelblue', label='P5-P95 置信区间')
    ax.fill_between(hours, result['p25'], result['p75'], alpha=0.3,
                    color='steelblue', label='P25-P75 置信区间')
    ax.plot(hours, result['p50'], 'b-', linewidth=2, label='P50 中位数')
    ax.plot(hours, result['mean'], 'r--', linewidth=1.5, label='均值')

    ax.set_xlabel('时刻 (h)', fontsize=12)
    ax.set_ylabel('充电负荷 (kW)', fontsize=12)
    ax.set_title('高速服务区充电负荷概率分布 (工作日, 中型服务区)', fontsize=14)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_charging_load_probability.pdf'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_charging_load_probability.png'),
                    dpi=150, bbox_inches='tight')
    return fig


def fig2_scenario_comparison(mc_scenarios, save=True):
    """图2: 不同日类型充电负荷对比 (P50)"""
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(10, 5))
    hours = np.arange(24)
    colors = {'workday': '#2ecc71', 'weekend': '#3498db',
              'holiday': '#e74c3c', 'spring_festival': '#e67e22'}
    labels_cn = {'workday': '工作日', 'weekend': '双休日',
                 'holiday': '节假日', 'spring_festival': '春运'}

    for dt in ['workday', 'weekend', 'holiday', 'spring_festival']:
        if dt in mc_scenarios:
            ax.plot(hours, mc_scenarios[dt]['p50'], color=colors[dt],
                    linewidth=2, label=f"{labels_cn[dt]}")

    ax.set_xlabel('时刻 (h)', fontsize=12)
    ax.set_ylabel('充电负荷 (kW)', fontsize=12)
    ax.set_title('不同日类型充电负荷P50曲线对比', fontsize=14)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_scenario_comparison.pdf'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_scenario_comparison.png'),
                    dpi=150, bbox_inches='tight')
    return fig


def fig3_power_balance(pv_profile, load_profile, op_result, season='summer',
                       weather='clear', save=True):
    """图3: 典型日功率平衡堆叠面积图"""
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(12, 5))
    hours = np.arange(24)

    # 堆叠面积: PV → 负荷(负) → 储能充放 → 电网交互
    ax.fill_between(hours, 0, pv_profile, alpha=0.4, color='#f39c12',
                    label='光伏出力')
    ax.fill_between(hours, 0, -load_profile, alpha=0.3, color='#e74c3c',
                    label='总负荷')

    # 净负荷
    net_load = pv_profile - load_profile
    ax.plot(hours, net_load, 'k-', linewidth=1.5, label='净负荷 (PV-负荷)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 储能和电网交互
    if op_result is not None:
        ax.plot(hours, op_result['soc_curve'] * 100, 'g-', linewidth=1.5,
                label='储能SOC (%)', alpha=0.7)

    ax.set_xlabel('时刻 (h)', fontsize=12)
    ax.set_ylabel('功率 (kW)', fontsize=12)
    ax.set_title(f'典型日功率平衡 ({season} {weather})', fontsize=14)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc='lower left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_power_balance.pdf'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_power_balance.png'),
                    dpi=150, bbox_inches='tight')
    return fig


def fig4_pareto_frontier(pareto_results, save=True):
    """图4: 自洽率 vs NPC Pareto前沿"""
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(8, 6))

    ssr = [r['self_sufficiency'] * 100 for r in pareto_results]
    npc = [r['npc'] / 1e4 for r in pareto_results]
    pv = [r['pv_capacity'] for r in pareto_results]

    scatter = ax.scatter(ssr, npc, c=pv, cmap='RdYlGn_r', s=80,
                         edgecolors='black', linewidth=0.5, alpha=0.8)

    ax.set_xlabel('新能源自洽率 (%)', fontsize=12)
    ax.set_ylabel('NPC (万元)', fontsize=12)
    ax.set_title('Pareto前沿: 自洽率 vs 全生命周期成本', fontsize=14)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('光伏容量 (kWp)', fontsize=10)

    # 标注最优区域
    best_idx = np.argmin(npc)
    ax.annotate(f'最优\nPV={pv[best_idx]:.0f}kWp\nSSR={ssr[best_idx]:.1f}%',
                xy=(ssr[best_idx], npc[best_idx]),
                xytext=(ssr[best_idx] + 5, npc[best_idx] + 5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

    ax.grid(True, alpha=0.3)

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_pareto_frontier.pdf'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_pareto_frontier.png'),
                    dpi=150, bbox_inches='tight')
    return fig


def fig5_sensitivity_heatmap(sensitivity_data, save=True):
    """图5: 敏感性分析热力图"""
    ensure_output_dir()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1: 电价-光伏成本敏感性 (对最优PV容量的影响)
    param1 = sensitivity_data.get('pv_cost_mult', [0.6, 0.8, 1.0, 1.2, 1.4])
    param2 = sensitivity_data.get('grid_price_mult', [0.6, 0.8, 1.0, 1.2, 1.4])
    data1 = sensitivity_data.get('pv_capacity', np.zeros((5, 5)))

    im1 = axes[0].imshow(data1, cmap='YlOrRd', aspect='auto', origin='lower')
    axes[0].set_xticks(range(len(param2)))
    axes[0].set_xticklabels([f'{x:.0%}' for x in param2])
    axes[0].set_yticks(range(len(param1)))
    axes[0].set_yticklabels([f'{x:.0%}' for x in param1])
    axes[0].set_xlabel('电网电价系数', fontsize=11)
    axes[0].set_ylabel('光伏成本系数', fontsize=11)
    axes[0].set_title('最优光伏容量 (kWp)', fontsize=12)
    for i in range(len(param1)):
        for j in range(len(param2)):
            axes[0].text(j, i, f'{data1[i, j]:.0f}', ha='center', va='center',
                        fontsize=9, color='white' if data1[i, j] > np.median(data1) else 'black')
    fig.colorbar(im1, ax=axes[0])

    # 子图2: 电价-储能成本敏感性 (对NPC的影响)
    data2 = sensitivity_data.get('npc', np.zeros((5, 5)))
    im2 = axes[1].imshow(data2 / 1e4, cmap='YlOrRd_r', aspect='auto', origin='lower')
    axes[1].set_xticks(range(len(param2)))
    axes[1].set_xticklabels([f'{x:.0%}' for x in param2])
    axes[1].set_yticks(range(len(param1)))
    axes[1].set_yticklabels([f'{x:.0%}' for x in param1])
    axes[1].set_xlabel('电网电价系数', fontsize=11)
    axes[1].set_ylabel('光伏成本系数', fontsize=11)
    axes[1].set_title('全生命周期NPC (万元)', fontsize=12)
    for i in range(len(param1)):
        for j in range(len(param2)):
            axes[1].text(j, i, f'{data2[i, j]/1e4:.1f}', ha='center', va='center',
                        fontsize=9, color='white' if data2[i, j] > np.median(data2) else 'black')
    fig.colorbar(im2, ax=axes[1])

    fig.suptitle('敏感性分析: 电价与光伏成本对优化结果的影响', fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_sensitivity_heatmap.pdf'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_sensitivity_heatmap.png'),
                    dpi=150, bbox_inches='tight')
    return fig


def fig6_radar_chart(scheme_results, save=True):
    """图6: 多方案雷达图对比"""
    ensure_output_dir()

    categories = ['能源自洽率', '经济性\n(NPC倒数)', '碳减排', '投资回收期\n(倒数)', '光伏消纳率', '供电可靠率']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for idx, (name, values) in enumerate(scheme_results.items()):
        values_norm = values + values[:1]
        ax.fill(angles, values_norm, alpha=0.1, color=colors[idx])
        ax.plot(angles, values_norm, 'o-', linewidth=2, color=colors[idx],
                label=name, markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8)
    ax.set_title('多方案综合对比雷达图', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_radar_chart.pdf'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_radar_chart.png'),
                    dpi=150, bbox_inches='tight')
    return fig


def fig7_monthly_energy_balance(monthly_pv, monthly_load, save=True):
    """图7: 月度能量平衡柱状图"""
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(12, 5))
    months = np.arange(1, 13)
    width = 0.35

    bars1 = ax.bar(months - width/2, monthly_pv / 1000, width,
                   label='光伏发电', color='#f39c12', alpha=0.8)
    bars2 = ax.bar(months + width/2, monthly_load / 1000, width,
                   label='总用电负荷', color='#3498db', alpha=0.8)

    ax.set_xlabel('月份', fontsize=12)
    ax.set_ylabel('电量 (MWh)', fontsize=12)
    ax.set_title('月度能量平衡分析', fontsize=14)
    ax.set_xticks(months)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # 标注自给率
    for m in months:
        ratio = monthly_pv[m-1] / max(monthly_load[m-1], 1) * 100
        ax.annotate(f'{ratio:.0f}%', xy=(m, max(monthly_pv[m-1], monthly_load[m-1])/1000),
                    ha='center', fontsize=8, color='gray')

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig7_monthly_energy_balance.pdf'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig7_monthly_energy_balance.png'),
                    dpi=150, bbox_inches='tight')
    return fig


def fig8_decision_topsis(decision_result, save=True):
    """图8: AHP-TOPSIS决策结果 — 方案贴近度对比柱状图

    数据来源: 文件09 §4.3 TOPSIS计算结果
    """
    ensure_output_dir()

    schemes = decision_result.schemes
    scores = decision_result.topsis['scores']
    ranks = decision_result.topsis['rank']

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn(scores / scores.max())

    bars = ax.bar(range(len(schemes)), scores, color=colors,
                  edgecolor='black', linewidth=0.8)

    # 标注贴近度和排名
    for i, (score, rank) in enumerate(zip(scores, ranks)):
        ax.text(i, score + 0.02, f'C={score:.3f}\nRank #{rank}',
                ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels(schemes, fontsize=10)
    ax.set_ylabel('相对贴近度 C_i', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title('AHP-TOPSIS方案综合评价结果 (相对贴近度)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig8_decision_topsis.pdf'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig8_decision_topsis.png'),
                    dpi=150, bbox_inches='tight')
    return fig


def fig9_scenario_comparison(scenario_results, save=True):
    """图9: 多情景NPC-自洽率对比散点图

    Parameters
    ----------
    scenario_results : dict
        {scenario_name: {'npc_wan_yuan': float, 'self_sufficiency': float, ...}}
    """
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(10, 6))

    scenario_colors = {
        'conservative': '#3498db',
        'baseline': '#f39c12',
        'aggressive': '#e74c3c',
    }
    scenario_markers = {'conservative': 's', 'baseline': 'o', 'aggressive': '^'}

    for name, result in scenario_results.items():
        color = scenario_colors.get(name, '#888888')
        marker = scenario_markers.get(name, 'o')
        ax.scatter(result['self_sufficiency'] * 100, result['npc_wan_yuan'],
                   s=200, c=color, marker=marker, edgecolors='black',
                   linewidth=1.5, label=f'{name} 情景', zorder=5)
        ax.annotate(
            f"PV={result.get('pv_capacity', 0):.0f}kWp\n"
            f"ESS={result.get('ess_capacity', 0):.0f}kWh\n"
            f"NPC={result['npc_wan_yuan']:.0f}万",
            xy=(result['self_sufficiency'] * 100, result['npc_wan_yuan']),
            xytext=(10, 15), textcoords='offset points',
            fontsize=8, alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel('新能源自洽率 (%)', fontsize=12)
    ax.set_ylabel('NPC (万元)', fontsize=12)
    ax.set_title('多情景分析: 保守/基准/激进情景对比', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig9_scenario_comparison.pdf'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig9_scenario_comparison.png'),
                    dpi=150, bbox_inches='tight')
    return fig


def fig10_pareto_3d(pareto_solutions, save=True):
    """图10: NSGA-II三维Pareto前沿 (NPC × SSR × Carbon)

    Parameters
    ----------
    pareto_solutions : list[dict]
        NSGA-II输出的Pareto最优解集
    """
    ensure_output_dir()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    npc_wan = [s['npc_wan_yuan'] for s in pareto_solutions]
    ssr = [s['ssr_pct'] for s in pareto_solutions]
    carbon = [s['carbon_reduction_t'] for s in pareto_solutions]
    pv = [s['pv_capacity'] for s in pareto_solutions]

    scatter = ax.scatter(npc_wan, ssr, carbon, c=pv, cmap='RdYlGn_r',
                        s=60, edgecolors='black', linewidth=0.5, alpha=0.85)

    ax.set_xlabel('NPC (万元)', fontsize=11, labelpad=10)
    ax.set_ylabel('自洽率 (%)', fontsize=11, labelpad=10)
    ax.set_zlabel('碳减排 (tCO2/年)', fontsize=11, labelpad=10)
    ax.set_title('NSGA-II 三维Pareto前沿\nNPC vs 自洽率 vs 碳减排',
                 fontsize=14, pad=15)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('PV容量 (kWp)', fontsize=10)

    # 标注最优折中解 (SSR > 25% 中NPC最低)
    feasible = [s for s in pareto_solutions if s['ssr_pct'] > 25]
    if feasible:
        best = min(feasible, key=lambda s: s['npc_wan_yuan'])
        ax.scatter([best['npc_wan_yuan']], [best['ssr_pct']],
                   [best['carbon_reduction_t']],
                   s=200, c='red', marker='*', edgecolors='black',
                   linewidth=1.5, label='推荐折中解', zorder=10)
        ax.legend(fontsize=10)

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig10_pareto_3d.pdf'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig10_pareto_3d.png'),
                    dpi=150, bbox_inches='tight')
    return fig


# ============================================================
# v6 新增图表
# ============================================================

def fig11_topology_radar(topology_names, soi_scores, save=True):
    """v6: 四拓扑多维度对比雷达图

    Parameters
    ----------
    topology_names : list of str
        拓扑名称 ['交流AC', '直流DC', '混合Hybrid', '链式Ring']
    soi_scores : np.ndarray (4, n_dims)
        各拓扑在各维度的归一化得分
    """
    ensure_output_dir()
    n_dims = soi_scores.shape[1]
    dim_labels = ['综合效率', '投资经济性', '技术成熟度', '可靠性',
                   '可再生能源\n利用率', '可扩展性']
    if n_dims < 6:
        dim_labels = dim_labels[:n_dims]

    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    linestyles = ['-', '--', '-.', ':']

    for i, name in enumerate(topology_names):
        values = soi_scores[i].tolist()
        values += values[:1]
        ax.fill(angles, values, alpha=0.08, color=colors[i])
        ax.plot(angles, values, 'o-', linewidth=2, label=name,
                color=colors[i], linestyle=linestyles[i], markersize=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='gray')
    ax.set_title('微网拓扑架构多维度对比', fontsize=14, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig11_topology_radar.pdf'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig11_topology_radar.png'),
                    dpi=150, bbox_inches='tight')
    return fig


def fig12_economic_breakdown(econ_model, npc_val=None, lcoe_val=None,
                              pbp_val=None, irr_val=None, save=True):
    """v6: CAPEX/NPC/LCOE 经济指标瀑布图

    Parameters
    ----------
    econ_model : EconomicModel
        经济模型实例
    npc_val, lcoe_val, pbp_val, irr_val : float or None
        关键经济指标, None则跳过
    """
    ensure_output_dir()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: CAPEX breakdown pie
    capex_items = {
        '光伏系统': econ_model.capex_pv() / 1e4,
        '储能系统': econ_model.capex_ess() / 1e4,
        '充电设施': econ_model.capex_charging() / 1e4,
        '固定投资': _get_fixed_cost() / 1e4,
    }
    colors = ['#FFB74D', '#4FC3F7', '#81C784', '#E0E0E0']
    labels = list(capex_items.keys())
    sizes = list(capex_items.values())
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 10})
    ax1.set_title(f'CAPEX 分项构成\n(总: {sum(sizes):.1f} 万元)', fontsize=12,
                  fontweight='bold')

    # Right: Key indicators bar
    if npc_val is not None and lcoe_val is not None:
        indicators = {
            'NPC\n(万元)': npc_val / 1e4,
            'LCOE\n(×100元/kWh)': lcoe_val * 100,
            '动态回收期\n(年)': pbp_val if pbp_val and pbp_val < 99 else 0,
            'IRR\n(%)': (irr_val * 100) if irr_val else 0,
        }
        bar_colors = ['#FF7043', '#42A5F5', '#66BB6A', '#AB47BC']
        bars = ax2.bar(range(len(indicators)), list(indicators.values()),
                       color=bar_colors, edgecolor='white', linewidth=1.5)
        ax2.set_xticks(range(len(indicators)))
        ax2.set_xticklabels(list(indicators.keys()), fontsize=10)
        ax2.set_title('关键经济指标', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, indicators.values()):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')

    subsidy = econ_model.subsidy_total() / 1e4
    fig.suptitle(f'全生命周期经济评估 [{econ_model.scenario}情景]  '
                 f'补贴: {subsidy:.1f}万元',
                 fontsize=13, fontweight='bold')

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig12_economic_breakdown.pdf'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig12_economic_breakdown.png'),
                    dpi=150, bbox_inches='tight')
    return fig


def fig13_decision_cross_validation(decision_result, mdm, save=True):
    """v6: TOPSIS/VIKOR/GRA 三方法决策一致性对比

    Parameters
    ----------
    decision_result : DecisionResult
        AHP-TOPSIS 结果
    mdm : MultiMethodDecision
        VIKOR/GRA 结果 (已调用 run_all())
    """
    ensure_output_dir()

    schemes = decision_result.schemes
    n = len(schemes)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    # TOPSIS scores (normalize to 0-1)
    topsis_scores = np.array(decision_result.topsis['scores'])
    topsis_norm = topsis_scores / max(topsis_scores.max(), 0.001)

    # VIKOR Q (invert: lower Q = better, convert to 0-1 score)
    vikor_Q = np.array(mdm.vikor_result['Q'])
    q_max = max(vikor_Q.max(), 0.001)
    vikor_score = 1 - vikor_Q / q_max  # invert so higher = better

    # GRA degree
    gra_score = np.array(mdm.gra_result['grey_degree'])

    bars1 = ax.bar(x - width, topsis_norm, width, label='TOPSIS (AHP+熵权+CRITIC)',
                   color='#42A5F5', edgecolor='white')
    bars2 = ax.bar(x, vikor_score, width, label='VIKOR (折中排序)',
                   color='#FF7043', edgecolor='white')
    bars3 = ax.bar(x + width, gra_score, width, label='GRA (灰色关联度)',
                   color='#66BB6A', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(schemes, fontsize=10)
    ax.set_ylabel('归一化得分 / 关联度', fontsize=11)
    ax.set_title('多方法决策一致性验证: TOPSIS vs VIKOR vs GRA', fontsize=14,
                 fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    # 标注最优方案
    best_idx = mdm.get_best_by_consensus()[0]
    ax.annotate('Borda共识最优', xy=(best_idx, 1.05),
                fontsize=10, color='red', fontweight='bold',
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # 一致性标注
    n_agree = sum(1 for i in range(n)
                  if np.argmax([topsis_norm[i], vikor_score[i], gra_score[i]]) ==
                     np.argmax([topsis_norm[i], vikor_score[i], gra_score[i]]))
    ax.text(0.98, 0.95, f'方法一致性: {n_agree}/{n} 相同排序',
            transform=ax.transAxes, fontsize=10, ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig13_decision_cross_validation.pdf'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig13_decision_cross_validation.png'),
                    dpi=150, bbox_inches='tight')
    return fig


if __name__ == '__main__':
    print("可视化模块加载成功")
    print(f"图表输出目录: {OUTPUT_DIR}")
