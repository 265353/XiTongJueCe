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


if __name__ == '__main__':
    print("可视化模块加载成功")
    print(f"图表输出目录: {OUTPUT_DIR}")
