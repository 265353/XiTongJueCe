# 高速公路服务区光储充一体化系统仿真

## 项目概述

基于"高速公路服务区光储充一体化研究数据集"（25份数据文件），构建了从**光伏资源评估 → 蒙特卡洛充电负荷预测 → 建筑负荷ABM → 微网拓扑对比 → PSO/NSGA-II/GA/EGPSO/鲁棒容量优化 → AHP-TOPSIS/VIKOR/GRA决策遴选 → 经济性全生命周期评估**的完整仿真链。

仿真粒度到设备级（组件/逆变器/PCS/充电桩/变压器），生成13张论文级图表和量化分析结果。覆盖6大研究任务全部模块。

## 目录结构

```
simulation/
├── README.md                        # 本文件
├── HANDOVER.md                       # 变更记录 (v6.2→v6.5)
├── HANDOVER_ARCHIVE.md               # 历史归档 (v1→v5)
├── microgrid_architecture.md         # 微网设备级组网架构
├── DATA_REQUIREMENTS.md              # 数据调研清单与进度
│
├── [core — 数据与参数]
│   ├── config.py                     # 全部参数 + 退化模型 + TMY气象
│   └── calendar_utils.py             # 2025真实日历 + Markov天气链
│
├── [task 1+2+3 — 资源评估与负荷预测]
│   ├── pv_generation.py              # 光伏出力8760h合成 (Sandia热模型 + Kasten-Czeplak)
│   ├── pv_resource_analysis.py       # 四类资源区 + 25省辐射量测算
│   ├── mc_charging_load.py           # 蒙特卡洛充电负荷 (GMM到达 + FCFS排队 + 可靠性)
│   └── building_load_abm.py          # Agent-Based建筑负荷 + ASHRAE校准框架
│
├── [task 4 — 微网架构]
│   └── topology_comparison.py        # AC/DC/Hybrid/Ring四拓扑量化对比
│
├── [task 5 — 容量优化]
│   ├── capacity_optimization.py      # PSO + TOU调度 + 8760h验证 + 鲁棒优化
│   ├── nsga2.py                      # NSGA-II多目标 (3目标: NPC/SSR/碳)
│   └── optimization_comparison.py    # GA/EGPSO + 两阶段鲁棒优化框架
│
├── [task 6 — 决策遴选]
│   ├── decision_framework.py         # AHP-熵权-CRITIC-TOPSIS
│   └── advanced_decision_methods.py  # VIKOR折中排序 + GRA + Borda投票
│
├── [shared — 经济与评估]
│   └── economic_model.py             # NPC/LCOE/IRR/PBP/ROI/BCR独立经济模型
│
├── [tests]
│   ├── test_economic_model.py        # 经济模型单元测试 (58 cases)
│   └── test_capacity_optimization.py # 调度回归测试 (15 cases)
│
├── [output — 可视化]
│   ├── visualization.py              # 图表生成 (13张)
│   ├── microgrid_frontend.py         # SCADA风格HTML监控
│   └── interactive_dashboard.py      # Plotly交互式仪表盘
│
├── main.py                           # 主程序 (--mode fast|nsga2|full|v6|robust)
├── results/                          # JSON数值结果
└── figures/                          # 图表输出 (PDF + PNG + HTML)
```

## 快速运行

```bash
cd simulation
pip install numpy scipy pandas matplotlib seaborn pytest

# 快速模式 (仅PSO, 2-5分钟)
python main.py --mode fast

# NSGA-II模式 (PSO + 多目标, 5-15分钟)
python main.py --mode nsga2

# 完整模式 (全部模块含决策框架, 8-20分钟)
python main.py --mode full

# 鲁棒优化模式 (标称+鲁棒PSO对比)
python main.py --mode robust

# 运行测试
python -m pytest test_economic_model.py test_capacity_optimization.py -v
```

## 仿真流程 (8步)

### 1. 蒙特卡洛充电负荷仿真 (`mc_charging_load.py`)

基于概率抽样模拟每辆EV的充电行为，含**FCFS排队模型 + 充电桩可靠性**：

| 参数 | 分布 | 来源 |
|------|------|------|
| 日行驶里程 | LN(3.2, 0.88²) | 文件05 |
| 起始SOC | Gamma(2.8, 0.075) | 文件02/11 |
| 百公里电耗 | N(车型相关, σ²) | 文件02 |
| 电池容量 | N(车型相关, σ²) | 文件11 (7种车型) |
| 充电起始时间 | GMM(k=2, μ1=11, μ2=15) | 文件05 |
| 充电功率 | 离散{60:0.3, 120:0.5, 240:0.2} | 文件03 |

- 车辆到达：Poisson过程 + GMM双峰到达时刻 + 月度车流量调节
- **排队模型**: 26终端(16×120kW + 10×480kW), FCFS + SOC依赖耐心 + 提前离充
- **可靠性**: Poisson故障 (MTBF/MTTR), 120kW可用率95%, 480kW可用率90%
- 每次仿真 N=5,000次, 输出 P5/P25/P50/P75/P95 置信区间
- 覆盖4种日类型 + 5类天气 × 12月

### 2. 光伏出力合成 (`pv_generation.py`)

- 4季×24h逐时归一化系数 + 5类天气Kasten-Czeplak衰减
- **Sandia热模型**: `T_cell = T_amb + (NOCT-20) × G/800`, γ=-0.30%/℃
- TMY逐时温度+GHI (武汉, PVGIS), 夏季高温中午出力下降~9%
- 生成8760h年度出力序列

### 3. PSO容量优化 (`capacity_optimization.py`)

粒子群算法求解最优配置：

```
决策变量: P_pv (kWp), E_ess (kWh), P_ess (kW)
目标函数: min NPC (全生命周期净现值成本)
约束条件:
  - TOU感知最优调度 (365天顺序 + SOC跨天连续)
  - SOC: 10% ≤ SOC ≤ 90%
  - 自洽率 ≥ 70% (软约束)
  - 变压器: 进口≤2,500kVA×0.95, 出口双向约束
  - 场地面积 ≤ 可用面积
  - 电池复合退化 (Arrhenius温度 + DOD应力)
```

### 3a. NSGA-II多目标优化 (`nsga2.py`)

三目标: min NPC, max 自洽率, max 碳减排量。非支配排序 + 拥挤距离 + SBX交叉。

### 3b. 算法对比 (`optimization_comparison.py`)

GA (轮盘赌+SBX) + EGPSO (精英遗传PSO) + 两阶段鲁棒优化 (min-max NPC)。

### 4-6. Pareto前沿 / 敏感性分析 / 多方案对比

非支配排序, 光伏成本±40%×电价±40%热力图, 4方案雷达图。

### 7. AHP-TOPSIS决策遴选 (`decision_framework.py`)

四准则(能源性/经济性/可靠性/环境性), 组合赋权(乘法合成), 多方法交叉验证(VIKOR/GRA/Borda)。

### 8. 全生命周期经济评估 (`economic_model.py`)

NPC / LCOE / IRR / PBP(动态/静态) / ROI / BCR, 含补贴+碳交易+更换+残值。

## 关键结果 (v6.5)

### 最优配置 (中型服务区, 三类资源区)

| 指标 | 数值 | 说明 |
|------|:----:|------|
| 光伏装机 | 1,231 kWp | 场地面积约束(8,000 m²) |
| 储能容量 | ~2,000 kWh | TOU套利驱动 |
| 储能功率 | ~1,000 kW | PSO自由优化 |
| 自洽率(8760h) | ~27% | 全时序验证 |
| NPC (20年) | ~1,821 万元 | 含补贴+碳交易+更换+残值 |
| LCOE | ~1.12 元/kWh | 平准化度电成本 |
| 动态回收期 | ~9 年 | 含ESS投资 |
| 碳减排 | ~440 tCO2/年 | CCER约70元/tCO2 |

### 排队模型效果 (含可靠性)

| 日类型 | 弃充(辆/天) | 排队(辆/天) | 因故障弃充 |
|--------|:--------:|:--------:|:--------:|
| 工作日 | 0 | 8.8 | ~0.5 |
| 周末 | 2.0 | 23.1 | ~1.5 |
| 节假日 | 14.1 | 36.5 | ~3.2 |
| 春运 | 41.4 | 46.1 | ~5.1 |

### 电池退化模型 (v6.5)

| 温度 | 日历衰减 | 80% DOD循环 | 50% DOD循环 | 30% DOD循环 |
|------|:------:|:--------:|:--------:|:--------:|
| 25°C | 2.00%/年 | 8,000次 | 12,212次 | 19,340次 |
| 35°C | 3.02%/年 | — | — | — |
| 45°C | 5.50%/年 | — | — | — |

### 多方法决策交叉验证

| 方法 | 最优方案 | 一致性 |
|------|---------|:--:|
| AHP-TOPSIS | 方案C:光储均衡 | — |
| VIKOR (v=0.5) | 方案C:光储均衡 | ✓ |
| 灰色关联度 (GRA) | 方案C:光储均衡 | ✓ |
| Borda共识投票 | 方案C:光储均衡 (12分) | 3/3一致 |

### 四拓扑架构对比

| 拓扑 | 综合效率 | 总投资 | 技术成熟度 | 推荐场景 |
|------|:--:|:--:|:--:|------|
| AC (交流) | 96.6% | 1,250万 | ★★★★★ | 一般场景, 成本最优 |
| DC (直流) | 93.4% | 1,495万 | ★★★ | 高比例PV+充电 |
| Hybrid (混合) | 96.7% | 1,370万 | ★★ | 大型综合服务区 |
| Ring (链式) | 94.7% | 1,396万 | ★★ | 多服务区带状联动 |

## 图表清单

| 编号 | 内容 | 类型 | 版本 |
|:--:|------|:--:|:--:|
| Fig1 | 充电负荷概率分布 (工作日 P5-P95) | 时间序列 | v1 |
| Fig2 | 四种日类型充电负荷P50对比 | 多曲线对比 | v1 |
| Fig3 | 典型日功率平衡 (含温度效应) | 堆叠面积图 | v5 |
| Fig4 | Pareto前沿 (自洽率 vs NPC) | 散点图 | v1 |
| Fig5 | 敏感性分析热力图 (电价 × 光伏成本) | 热力图 | v1 |
| Fig6 | 四方案雷达图 (六维度综合评价) | 雷达图 | v1 |
| Fig7 | 月度能量平衡 (光伏 vs 用电负荷) | 柱状图 | v1 |
| Fig8 | AHP-TOPSIS决策结果 (方案贴近度) | 柱状图 | v5 |
| Fig9 | 多情景对比 (保守/基准/激进) | 散点图 | v5 |
| Fig10 | NSGA-II三维Pareto前沿 (NPC×SSR×Carbon) | 3D散点 | v5 |
| Fig11 | 四拓扑多维度对比雷达图 | 雷达图 | v6 |
| Fig12 | CAPEX分项+关键经济指标 | 饼图+柱图 | v6 |
| Fig13 | TOPSIS/VIKOR/GRA 三方法一致性 | 分组柱图 | v6 |

## 版本历史

| 版本 | 日期 | 主要变化 |
|------|------|---------|
| v1 | 2026-06 | 初始版本 |
| v2 | 2026-06 | 设备级成本, 7车型, 5天气 |
| v3 | 2026-06 | MC bug修复, 日类型加权, 数据审查修正 |
| v4 | 2026-06 | TOU最优调度, 季节性电价, 排队模型, 变压器约束, 8760h验证 |
| v5 | 2026-06 | NSGA-II多目标, AHP-TOPSIS决策, 碳交易/补贴/温度效应 |
| v6 | 2026-06 | 6新模块: economic/pv_resource/ABM/topology/optimization/decision |
| v6.1 | 2026-06 | 全模块集成, --mode v6, EconomicModel桥接 |
| v6.2 | 2026-06 | 17项理想化修复: GMM到达+SOC连续+有限预见+分项残值 |
| v6.3 | 2026-06 | 12项数据调研: TMY气象+渗透率+Markov链+碳价预测+MTBF |
| v6.4 | 2026-06 | V2G/DR/S曲线/ABM标定 (文件25) |
| v6.5 | 2026-06 | 电池退化温度/DOD + FCFS可靠性 + 两阶段鲁棒优化全链集成 + 单元测试 |

## 测试

```bash
# 经济模型 (58 tests)
python -m pytest test_economic_model.py -v

# 调度回归 (15 tests)
python -m pytest test_capacity_optimization.py -v

# 全部测试
python -m pytest test_*.py -v
```

## 论文引用

1. 交通运输部高速公路充电设施统计数据
2. 国家能源局充电基础设施数据
3. 《2024年中国风能太阳能资源年景公报》
4. 中国电机工程学报: 计及电动汽车需求响应的高速公路服务区光储充鲁棒优化配置 (2025)
5. 综合智慧能源: 基于用能自洽的高速服务区微网光储组合优化配置 (2024)
6. 生态环境部: 2024年全国电网排放因子 0.5703 tCO2/MWh
7. Wang et al. (2016) JPS 327, LFP calendar aging (Ea=31.5 kJ/mol)
8. Sarasketa-Zabala et al. (2014) JES 161, DOD stress exponent
