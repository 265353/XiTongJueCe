# 仿真待调研数据清单

> 以下12项问题因缺少外部数据而暂时搁置，修复时需先完成数据调研。
> 创建日期: 2026-06-21

---

## 1. 气象数据 (3项，统一来源)

| # | 问题 | 所需数据 | 建议来源 | 优先级 |
|---|------|---------|---------|:---:|
| 1.1 | 逐时气温替代月均温 | TMY逐时干球温度 | [PVGIS API](https://re.jrc.ec.europa.eu/pvg_tools/en/), NASA POWER | 高 |
| 1.2 | 辐照度从PV系数反推→真实GHI | TMY逐时GHI/DNI/DHI | PVGIS (SARAH2), NSRDB | 高 |
| 1.3 | 天气修正系数细化 | 天气类型→实际辐照度衰减统计映射 | 当地光伏电站实测或文献经验公式 | 中 |

**备注**: 文件14已提到PVGIS API获取TMY数据的方法。解决1.1和1.2后，pv_generation.py的温度模型和辐照度计算可从根本上重写。

---

## 2. 交通流量数据 (4项)

| # | 问题 | 所需数据 | 建议来源 | 优先级 |
|---|------|---------|---------|:---:|
| 2.1 | 到达车辆≠充电车辆 | 高速公路服务区充电渗透率（按日类型/时段） | 交通部《高速公路充电设施统计》、充电运营平台公开数据 | 高 |
| 2.2 | 到达率季节性动态调整 | 高速公路月度车流量数据（区分日类型） | 交通部《全国干线公路运行年报》、各省交投集团年报 | 高 |
| 2.3 | 充电负荷与建筑负荷耦合系数 | 每辆充电车辆对应的服务区停留人数/时长分布 | 服务区运营统计（难以获取，建议用典型假设替代） | 高 |
| 2.4 | Markov链升级为二阶 | 连续两天天气类型的联合分布/转移矩阵 | 当地30年气候统计资料、CMA气象数据 | 低 |

---

## 3. 设备/市场数据 (3项)

| # | 问题 | 所需数据 | 建议来源 | 优先级 |
|---|------|---------|---------|:---:|
| 3.1 | 充电桩可用率/故障率 | 直流快充桩实际故障率、MTBF/MTTR | 充电运营商运维数据、NB/T 33001标准 | 中 |
| 3.2 | DC/DC变换器成本 | 大功率DC/DC市场报价/招标数据 | 储能EPC招标公告、BloombergNEF | 中 |
| 3.3 | 组件可靠度数据 | 逆变器/PCS/DC-DC MTBF | 设备厂商datasheet、IEEE 493 Gold Book | 低 |

---

## 4. 政策/市场数据 (2项)

| # | 问题 | 所需数据 | 建议来源 | 优先级 |
|---|------|---------|---------|:---:|
| 4.1 | 碳价走势 | CCER/CEA价格预测 | 全国碳市场年度报告、Refinitiv | 低 |
| 4.2 | 上网电价政策趋势 | 分布式光伏上网电价最新政策 | 国家能源局/发改委文件 | 中 |

---

## 数据调研进度

> 最后更新: 2026-06-21. 粗体=已集成到代码.

| 数据项 | 状态 | 集成版本 | 备注 |
|--------|:--:|:-------:|------|
| **1.1 逐时温度** | ✅ 已集成 | v6.3 | TMY_HOURLY_TEMP (武汉12×24), pv_generation.py use_tmy=True |
| **1.2 逐时GHI** | ✅ 已集成 | v6.3 | TMY_GHI_CLEAR + Kasten-Czeplak衰减, pv_generation.py |
| **1.3 天气修正细化** | ✅ 已集成 | v6.3 | WEATHER_COEFF 5值修正 (Kasten-Czeplak), config.py |
| **2.1 充电渗透率** | ✅ 已集成 | v6.3 | CHARGING_PENETRATION (7.5-14.4%), config.py |
| **2.2 月度车流量** | ✅ 已集成 | v6.3 | MONTHLY_ARRIVAL_MULTIPLIER, mc_charging_load.py |
| **2.3 充-建耦合系数** | ✅ 已集成 | v6.3 | CHARGING_BUILDING_COUPLING, capacity_optimization.py |
| **2.4 二阶Markov矩阵** | ✅ 已集成 | v6.3 | WEATHER_TRANSITION_MATRIX + 二阶近似, calendar_utils.py |
| **3.1 充电桩故障率** | ✅ 已集成 | v6.3 | PILE_AVAILABILITY (120kW 95%/480kW 90%), config.py |
| **3.2 DC/DC成本** | ✅ 已集成 | v6.3 | ESS_COST dc_dc_per_kw=100, pcs_per_kw 250→200, config.py |
| **3.3 组件MTBF** | ✅ 已集成 | v6.3 | MTBF/MTTR数据库 + get_availability(), topology_comparison.py |
| **4.1 碳价预测** | ✅ 已集成 | v6.3 | CCER_PRICE_FORECAST 2025-2040, economic_model.py |
| **4.2 上网电价政策** | ✅ 已集成 | v6.3 | FEED_IN_PRICE_BY_PROVINCE + FEED_IN_SCENARIOS, config.py |
| **5.1 S曲线负荷增长** | ✅ 已集成 | v6.4 | Bass/Logistic参数, get_load_growth_factor(), capacity_opt+economic |
| **5.2 需求响应(DR)** | ✅ 已集成 | v6.4 | DR_ENABLED/EVENTS/COMPENSATION, estimate_dr_annual_revenue() |
| **5.3 V2G双向充放电** | ✅ 已预留 | v6.4 | V2G_ENABLED=False, 参数完整 (需硬件确定后启用) |
| **5.4 ABM实地标定** | ✅ 已集成 | v6.4 | MONTHLY_VISITOR_FACTOR, 随机出勤率, calc_ashrae_metrics() |
