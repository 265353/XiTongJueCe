# Handover — 高速服务区光储充一体化仿真

> 最后更新: 2026-06-21 (v6.2, 数据驱动仿真真实性增强)

---

## [2026-06-21] v6.2 数据驱动仿真真实性增强

**Branch**: master | **Changed by**: 265353 | **Mode**: full (9 files, +494/-252)

### What Changed

| File | Category | Description |
|------|----------|-------------|
| `mc_charging_load.py` | Feature | GMM双峰到达(N(11,1.5²)45%+N(15,1.8²)55%), 时长上限480kW→1h/120kW→2h, 季节能耗调整 |
| `calendar_utils.py` | Feature | 类型特定天气持续性(P(stay)从月度分布反推), 替代全局persistence=0.65 |
| `capacity_optimization.py` | Refactor | **核心**: 365天顺序仿真+SOC跨天连续+有限预见(horizon=6). evaluate_config重写. 变压器双向约束. params_override支持 |
| `config.py` | Feature | get_rte(soc,c_rate), FEED_IN_PRICE_ESCALATION, 分项残值率(PV10%/ESS5%/Charging5%) |
| `economic_model.py` | Feature | 上网电价年涨幅应用, npc_from_aggregates分项残值计算 |
| `topology_comparison.py` | Feature | 串联组件可靠度模型替代硬编码可靠性评分 |
| `building_load_abm.py` | Feature | ABM峰值自动校准到设计峰值147kW (文件12) |
| `main.py` | Refactor | 敏感度分析: params_override替代全局config突变 |

### Why

前期审计发现15个严重理想化问题。v6.2基于14份研究数据集中的真实数据(文件01/02/03/05/10/11/12/13), 逐项修复充电到达模型、天气持续性、储能跨天调度、变压器约束、经济参数等17项假设。所有修复均引用数据集原文出处。

### Conflict Analysis

| Severity | Type | Detail |
|----------|------|--------|
| NONE | Signature | `simulate_daily_operation`/`evaluate_config`/`optimize_pso` 新增参数均有默认值, 5个调用方兼容 |
| NONE | Import | 所有新增import路径存在, 无循环依赖 |
| NONE | Config | `params_override` 局部传递, 不再修改全局 `config.PV_COST` |
| LOW | Performance | `evaluate_config` 365天顺序仿真(+52%调用), 通过减少贪婪迭代(80→40)和早期终止补偿 |

### Design Decision Impact

- `capacity_optimization.evaluate_config`: 从 typical_days 加权聚合改为 365天顺序仿真 — 与 `verify_8760h` 逻辑统一, 消除目标不一致
- `capacity_optimization._calculate_npc_detailed`: 保留+废弃标注 — v6模式使用 `EconomicModel.npc_from_aggregates`
- `mc_charging_load.simulate_day`: 从每小时Poisson改为天总量Poisson+GMM时刻采样 — 保持 `simulate_monte_carlo` 接口不变
- 全局config突变: **禁止** — 敏感度分析必须用 `params_override` 局部传递

### Caller Impact

`simulate_daily_operation(pv, load, season, tou, initial_soc, foresight_horizon)` — 5 callers:
- `evaluate_config` (新版, 传入 horizon=6)
- `verify_8760h` (新版, 传入 horizon=24 + SOC链)
- `microgrid_frontend.py:63` (旧版, 不传新参数, 默认兼容)
- `main.py:368` (fig3典型日, 不传新参数, 默认兼容)
- `interactive_dashboard.py:72` (仪表盘, 不传新参数, 默认兼容)

---

## 项目概述

基于14份研究数据集构建的完整仿真链：**光伏资源评估 → 蒙特卡洛充电负荷预测 → 建筑负荷ABM → 微网拓扑架构对比 → PSO/NSGA-II/GA/EGPSO容量优化 → AHP-TOPSIS/VIKOR/GRA方案决策遴选 → 经济性全生命周期评估**。

仿真粒度到设备级（组件/逆变器/PCS/充电桩/变压器），生成10张论文级图表和量化分析结果。覆盖6大研究任务全部模块。

---

## 文件结构

```
simulation/
├── HANDOVER.md                      # 本文件
├── README.md                        # 项目说明
├── microgrid_architecture.md        # 微网设备级组网架构 (拓扑/参数/造价)
│
├── [core — 数据与参数]
│   ├── config.py                    # 全部参数 + v6.2: get_rte()/分项残值/上网电价涨幅
│   └── calendar_utils.py           # 2025真实日历 + v6.2: 类型特定天气持续性
│
├── [task 1 — 光伏潜力评估]
│   ├── pv_generation.py            # 光伏出力8760h合成 (Sandia热模型)
│   └── pv_resource_analysis.py     # [v6新增] 四类资源区+25省辐射量+服务区测算
│
├── [task 2+3 — 充电+建筑负荷预测]
│   ├── mc_charging_load.py         # v6.2: GMM双峰到达+季节能耗+时长上限 (文件05/11)
│   └── building_load_abm.py        # v6.2: Agent-Based建筑负荷 + 峰值自动校准
│
├── [task 4 — 微网架构]
│   └── topology_comparison.py      # [v6新增] AC/DC/Hybrid/Ring四拓扑量化对比
│
├── [task 5 — 容量优化]
│   ├── capacity_optimization.py    # v6.2: 365天顺序仿真+SOC连续+有限预见+变压器约束
│   ├── nsga2.py                    # NSGA-II多目标优化 (3目标)
│   └── optimization_comparison.py  # [v6新增] GA+EGPSO算法 + 两阶段鲁棒优化框架
│
├── [task 6 — 决策遴选]
│   ├── decision_framework.py       # AHP-熵权-CRITIC-TOPSIS
│   └── advanced_decision_methods.py# [v6新增] VIKOR折中排序 + 灰色关联度 + Borda投票
│
├── [shared — 经济与评估]
│   └── economic_model.py           # [v6新增] NPC/LCOE/IRR/PBP/ROI/BCR独立经济模型
│
├── [output — 可视化]
│   ├── visualization.py            # 图表生成 (13张, v6新增3张)
│   ├── microgrid_frontend.py       # SCADA风格HTML监控
│   └── interactive_dashboard.py    # Plotly交互式仪表盘 (487行, 已完成)
│
├── main.py                         # 主程序入口 (--mode fast/nsga2/full)
├── results/                        # JSON数值结果
│   ├── mc_summary.json
│   ├── optimization_result.json
│   ├── pareto_results.json
│   └── decision_result.json
└── figures/                        # 图表输出 (PDF + PNG)
    ├── fig1_charging_load_probability.*
    ... (fig1-fig10)
    └── microgrid_frontend.html
```

---

## 仿真流程 (7步)

### Step 1: 蒙特卡洛充电负荷仿真

**文件**: `mc_charging_load.py` | **类**: `MonteCarloChargingSimulator`

| 参数 | 分布 | 来源 |
|------|------|------|
| 日行驶里程 | LN(3.2, 0.88²) | 文件05 |
| 起始SOC | Gamma(2.8, 0.075) | 文件02/11 |
| 百公里电耗 | N(车型相关, σ²) | 文件02 |
| 电池容量 | N(车型相关, σ²) | 文件11 (7种车型) |
| 充电功率 | 离散{60:0.3, 120:0.5, 240:0.2} | 文件03 |

- 车辆到达: Poisson过程 + 小时级车流量
- **排队模型**: 26终端(16×120kW + 2×480kW×5), 超时弃充(工作日45min→春运15min)
- 每次仿真 N=5,000次, 输出 P5/P25/P50/P75/P95 置信区间
- 覆盖4种日类型: 工作日/双休日/节假日/春运

### Step 2: 光伏出力合成 (v5: 含温度效应)

**文件**: `pv_generation.py` | **类**: `PVGenerator`

- 4季×24h逐时归一化系数 × 5类天气Markov链修正
- **[v5]** Sandia热模型温度修正: `T_cell = T_amb + (NOCT-20) × G/800`
- **[v5]** 功率温敏衰减: `P = P_stc × (1 + γ × (T_cell - 25))`, γ=-0.30%/℃
- 夏季高温中午出力下降约9%, 冬季低温略增益约3%
- 生成8760h年度出力序列

### Step 3: PSO容量优化

**文件**: `capacity_optimization.py` | **类**: `MicrogridOptimizer`

```
决策变量: P_pv (kWp), E_ess (kWh), P_ess (kW)
目标函数: min NPC (全生命周期净现值成本)
约束条件:
  - TOU感知最优调度 (24h迭代套利)
  - SOC: 10% ≤ SOC ≤ 90%
  - 自洽率 ≥ 70% (软约束)
  - 变压器容量 ≤ 2,500kVA × 0.95
  - 场地面积 ≤ 8,000 m²
  - 电池逐年衰减 (循环+日历)
```

### Step 3a: [v5] NSGA-II多目标优化

**文件**: `nsga2.py` | **类**: `NSGA2`

- 三目标: min NPC, max 自洽率, max 碳减排量
- 非支配排序 + 拥挤距离 + 锦标赛选择
- SBX模拟二进制交叉 + 多项式变异
- 输出Pareto最优解集

### Step 4: Pareto前沿分析

基于PSO评价的随机采样 + 非支配排序, 探索NPC vs SSR的trade-off关系。

### Step 5: 敏感性分析

光伏成本 ±40% × 电网电价 ±40%, 5×5热力图。

### Step 6: 多方案对比

4方案: 纯电网 / 仅光伏 / 光伏+储能(最优) / 大光伏+大储能。

### Step 6a: [v5] AHP-TOPSIS决策框架

**文件**: `decision_framework.py` | **类**: `DecisionFramework`

- **AHP**: 四准则(能源性/经济性/可靠性/环境性) 1-9标度两两比较, 特征向量法 + 一致性检验(CR<0.1)
- **熵权法**: 基于指标数据离散度客观赋权
- **CRITIC法**: 基于对比强度×冲突性赋权
- **组合赋权**: 乘法合成归一化
- **TOPSIS**: 计算正负理想解欧氏距离 → 相对贴近度排序

### Step 6b: [v5] 多情景分析

| 情景 | 负荷增长率 | EV渗透率 | 电价上涨 | 折现率 |
|------|:--:|:--:|:--:|:--:|
| 保守 | 5% | 10% | 1% | 8% |
| 基准 | 10% | 20% | 3% | 7% |
| 激进 | 15% | 28% | 5% | 6% |

---

## 快速运行

```bash
cd simulation
pip install numpy scipy pandas matplotlib seaborn

# 快速模式 (仅PSO, 2-5分钟)
python main.py --mode fast

# NSGA-II模式 (PSO + 多目标, 5-15分钟)
python main.py --mode nsga2

# 完整模式 (全部模块含决策框架, 8-20分钟)
python main.py --mode full

# 可视化前端 (浏览器打开)
python microgrid_frontend.py && open figures/microgrid_frontend.html
```

---

## 关键结果

### 最优配置 (中型服务区, 三类资源区)

| 指标 | 数值 | 说明 |
|------|:----:|------|
| 光伏装机 | 1,231 kWp | 场地面积约束(8,000 m²) |
| 储能容量 | ~2,000 kWh | TOU套利驱动 |
| 储能功率 | ~1,000 kW | PSO自由优化 |
| 自洽率(典型日) | ~28% | 含ESS时间转移 |
| 自洽率(8760h) | ~27% | 全时序验证 |
| NPC (20年静态) | ~3,200 万元 | 含补贴+碳交易收益 |
| 碳交易年收益 | ~4.8 万元 | CCER 70元/tCO2 |
| 补贴总额 | ~67.6 万元 | 光伏+储能+充电桩 |
| 投资回收期 | ~7.7 年 | 动态回收期 |

### 排队模型效果

| 日类型 | 弃充(辆/天) | 排队(辆/天) |
|--------|:--------:|:--------:|
| 工作日 | 0 | 8.8 |
| 周末 | 2.0 | 23.1 |
| 节假日 | 14.1 | 36.5 |
| 春运 | 41.4 | 46.1 |

### AHP-TOPSIS决策结果

| 方案 | 贴近度 C_i | 排名 |
|------|:------:|:--:|
| 方案B:均衡型 (光储最优) | 0.669 | **1** |
| 方案A:基础型 (纯电网) | 0.618 | 2 |
| 方案C:激进型 (大光储) | 0.568 | 3 |
| 方案D:离网型 | 0.382 | 4 |

### 多方法决策交叉验证 (v6新增)

| 方法 | 最优方案 | 一致性 |
|------|---------|:--:|
| AHP-TOPSIS | 方案C:光储均衡 | — |
| VIKOR (v=0.5) | 方案C:光储均衡 | ✓ |
| 灰色关联度 (GRA) | 方案C:光储均衡 | ✓ |
| Borda共识投票 | 方案C:光储均衡 (12分) | 3/3一致 |

### 四拓扑架构对比 (v6新增)

| 拓扑 | 综合效率 | 总投资 | 技术成熟度 | 推荐场景 |
|------|:--:|:--:|:--:|------|
| AC (交流) | 96.6% | 1,250万 | ★★★★★ | 一般场景, 成本最优 |
| DC (直流) | 93.4% | 1,495万 | ★★★ | 高比例PV+充电 |
| Hybrid (混合) | 96.7% | 1,370万 | ★★ | 大型综合服务区 |
| Ring (链式) | 94.7% | 1,396万 | ★★ | 多服务区带状联动 |

### 经济性指标 (v6独立经济模型)

| 指标 | 数值 | 说明 |
|------|:---:|------|
| NPC (20年) | ~1,821 万元 | 含补贴+碳交易+更换+残值 |
| LCOE | ~1.12 元/kWh | 平准化度电成本 |
| 动态回收期 | ~9 年 | 考虑折现+更换事件 |
| 三情景NPC | 保守1,403万 / 基准1,422万 / 激进1,444万 | — |

## 版本历史摘要

| 版本 | 日期 | 关键变更 | 详档 |
|------|------|---------|------|
| v6.2 | 2026-06-21 | 数据驱动仿真真实性: GMM到达+SOC连续+有限预见+类型特定天气+变压器双向+分项残值 | 见上 |
| v6.1 | 2026-06-21 | 全模块集成: --mode v6 + EconomicModel桥接 + ABM衔接 | 见上 |
| v6 | 2026-06-16 | 6新模块: economic/pv_resource/building_abm/topology/optimization/advanced_decision | 见上 |
| v5 | 2026-06-16 | NSGA-II + AHP-TOPSIS决策 + 碳交易/补贴/温度增强 | [→ARCHIVE](HANDOVER_ARCHIVE.md) |
| v4 | 2026-06-11 | 调度重构(24h套利) + 变压器约束 + 8760h验证 + M/M/n排队 | [→ARCHIVE](HANDOVER_ARCHIVE.md) |
| v3 | 2026-06-11 | 7车型独立分布 + GMM精确值 + 光伏衰减 + 碳因子修正 | [→ARCHIVE](HANDOVER_ARCHIVE.md) |
| v1 | 2026-06-10 | 初始提交: 光储充一体化仿真框架 | — |

---

## 数据来源映射

### 仿真→数据集 追溯 (v6.2扩展)

| 仿真模块 | 数据集文件 | 使用参数 |
|---------|-----------|---------|
| `pv_resource_analysis.py` | 文件04, 文件10 | 四类资源区系数/25省辐射量/服务区面积 |
| `pv_generation.py` | 文件10, 架构文档 | 逐时归一化系数/天气修正/Sandia热模型 |
| `mc_charging_load.py` | 文件02, 文件05, 文件11 | GMM到达模型(v6.2)/车型分布/SOC/季节能耗(v6.2)/时长上限(v6.2) |
| `building_load_abm.py` | 文件06 §4, 文件12 | 人员ABM参数/四季逐时负荷/峰值校准(v6.2: 147kW) |
| `topology_comparison.py` | 文件07 §2, 文件13 | 四拓扑方案/组件可靠度模型(v6.2) |
| `capacity_optimization.py` | 文件08, 文件01, 文件13, 文件03 | PSO/TOU价格/SOC连续(v6.2)/有限预见(v6.2)/变压器双向(v6.2) |
| `nsga2.py` | 文件08 | NSGA-II算法框架 |
| `optimization_comparison.py` | 文件08 §1-2 | GA/EGPSO/鲁棒优化 |
| `economic_model.py` | 文件08 §3, 文件13 | NPC/LCOE/IRR公式 + 上网电价涨幅(v6.2) + 分项残值(v6.2) |
| `decision_framework.py` | 文件09 | AHP/熵权/CRITIC/TOPSIS |
| `advanced_decision_methods.py` | 文件09 §2, §5 | VIKOR/GRA/权重敏感性 |
| `config.py` | 文件01/03/06/10/11/13 | get_rte()(v6.2)/分项残值(v6.2)/FEED_IN_PRICE_ESCALATION(v6.2) |
| `calendar_utils.py` | 文件10 §3 | 类型特定天气持续性(v6.2) |
| 碳交易 | 文件13 + 全国碳市场 | 排放因子0.5703 + CCER 70元/tCO2 |
| 补贴政策 | 文件01/13 | PV/ESS/充电桩地方补贴 |
| 负荷增长 | 文件06 §5 | 综合年增8-15% |

---

## 图表清单

| 编号 | 内容 | 类型 | 版本 |
|:--:|------|:--:|:--:|
| Fig1 | 充电负荷概率分布 (工作日 P5-P95) | 时间序列 | v1 |
| Fig2 | 四种日类型充电负荷P50对比 | 多曲线对比 | v1 |
| Fig3 | 典型日功率平衡 (含温度效应) | 堆叠面积图 | v5增强 |
| Fig4 | Pareto前沿 (自洽率 vs NPC) | 散点图 | v1 |
| Fig5 | 敏感性分析热力图 (电价 × 光伏成本) | 热力图 | v1 |
| Fig6 | 四方案雷达图 (六维度综合评价) | 雷达图 | v1 |
| Fig7 | 月度能量平衡 (光伏 vs 用电负荷) | 柱状图 | v1 |
| Fig8 | AHP-TOPSIS决策结果 (方案贴近度) | 柱状图 | **v5新增** |
| Fig9 | 多情景对比 (保守/基准/激进) | 散点图 | **v5新增** |
| Fig10 | NSGA-II三维Pareto前沿 (NPC×SSR×Carbon) | 3D散点 | **v5新增** |
| Fig11 | 四拓扑多维度对比雷达图 | 雷达图 | **v6新增** |
| Fig12 | CAPEX分项+关键经济指标 | 饼图+柱图 | **v6新增** |
| Fig13 | TOPSIS/VIKOR/GRA 三方法一致性 | 分组柱图 | **v6新增** |

---

## 已知限制 (v6.2更新)

**已解决 (v6.2):**
- [v6.2] 充电到达: 独立Poisson → GMM双峰模型 (文件05)
- [v6.2] 充电时长: 480kW上限3h→1h, 120kW 3h→2h (文件02)
- [v6.2] 天气持续性: 全局persistence=0.65 → 类型特定转移概率 (文件10)
- [v6.2] SOC连续: 每日重置0.5 → 365天跨天连续
- [v6.2] 完美预见: 24h全知 → PSO horizon=6h有限预见
- [v6.2] 变压器约束: 仅进口 → 进出口双向
- [v6.2] 拓扑可靠性: 硬编码 → 组件串联可靠度模型
- [v6.2] ABM峰值: 未校准 → 自动对标设计峰值147kW
- [v6.2] 敏感度分析: 全局config突变 → params_override局部传递
- [v6.2] 残值率: 统一5% → 分项 PV 10%/ESS 5%/Charging 5% (文件13)
- [v6.2] 上网电价: 固定0.35 → FEED_IN_PRICE_ESCALATION=1%/年
- [v6.2] 建筑负荷: 固定季节曲线 → ABM+校准 (v6.1)

**仍待解决 (数据不足):**
- 逐时环境温度 → 需接入PVGIS API获取TMY数据 (文件14提供方法)
- Sandia模型辐照度峰值 → 需外部GHI数据
- 电池退化与温度/DOD关系 → 文献中有但项目数据集缺失
- 充电桩FCFS队列 → 当前算法可接受, 风险/收益比低
- V2G双向充放电 → 未建模
- 需求侧响应收益 → 未包含
- 负荷增长线性假设 → 实际可能呈S曲线
- 两阶段鲁棒优化 → 框架已有但未与仿真全链集成
- ABM参数 → 基于典型值, 未对特定服务区实地标定

---

## 论文引用

运行仿真后打印的关键参数可直接用于课程报告表格。建议引用以下来源:

1. 交通运输部高速公路充电设施统计数据
2. 国家能源局充电基础设施数据
3. 《2024年中国风能太阳能资源年景公报》
4. 中国电机工程学报: 计及电动汽车需求响应的高速公路服务区光储充鲁棒优化配置 (2025)
5. 综合智慧能源: 基于用能自洽的高速服务区微网光储组合优化配置 (2024)
6. 生态环境部: 2024年全国电网排放因子 0.5703 tCO2/MWh
7. 全国碳排放权交易市场 (CEA/CCER) 2024-2025
