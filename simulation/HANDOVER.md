# 交接文档：高速服务区光储充微网仿真系统

**日期**: 2026-06-24
**版本**: v6.7
**状态**: 可运行，4个高价值改进方向已落地

---

## [2026-06-24] v6.7: 四方向物理真实性提升

**Branch**: master
**Changed by**: 265353
**Mode**: full (10+ files)

### What Changed

| File | Category | Description |
|------|----------|-------------|
| `calendar_utils.py` | feature | PVGIS 8760h实测GHI/温度/风速替代Markov链合成，UTC→本地时间(+8h)纠正 |
| `capacity_optimization.py` | feature | _build_8760h_sequence() 现在返回5元组(pv,load,tou,season,temp)；ABM建筑负荷预计算缓存 |
| `pv_generation.py` | feature | generate_daily_profile() 支持逐时温度+GHI输入，Sandia热模型用实测值代替月均值 |
| `building_load_abm.py` | feature | HVAC热负荷模型(ASHRAE 90.1)、自然采光照明调光、温度+GHI驱动替代固定剖面 |
| `config.py` | refactor | 无核心参数变更，已有函数(v6.5 Arrhenius/DOD退化)已存在 |
| `run_pipeline.py` | refactor | use_abm=True，_build_8760h_sequence()适配5元组返回 |
| `interactive_dashboard.py` | refactor | 适配5元组 |
| `optimization_comparison.py` | refactor | _evaluate_with_scenario()场景包装器适配5元组 |
| `web_server.py` | refactor | 适配5元组 |
| `run_pipeline_continue.py` | refactor | 适配5元组 |

### Why

四项按投入产出比推进：
1. **天气系统**（最高ROI）：PVGIS 8760h逐时GHI+温度+风速替代Markov合成。数据来自 `data/pvgis_tmy_wuhan.json` (8784h真实气象)。UTC→UTC+8纠正修复了武汉本地正午GHI=0的严重错误。
2. **建筑负荷物理模型**：温度驱动空调(ASHRAE 90.1围护结构+新风+COP)、GHI驱动照明(自然采光调光)，替代 `config.py` 4季×24h固定数字。ABM预计算缓存避免PSO每轮(30iter×40pop=1200次)重复调用ABM。
3. **光伏逐时温度**：Sandia热模型用逐时温度替代月均值，寒潮/热浪实时反映到光伏出力。
4. **储能Arrhenius温度模型**：温度依赖RTE、自放电(Arrhenius Ea/R=3800K)、DOD循环应力已接入调度及经济模型。

### Conflict Analysis

| Severity | Finding |
|----------|---------|
| NONE | 所有新增参数均为可选(默认None)，向后兼容旧调用方 |
| NONE | _build_8760h_sequence() 5元组：所有10个调用点已更新 |
| NONE | 签名变更(simulate_daily_operation/get_total_load/generate_daily_profile)均新增可选参数 |

### API Compatibility Notes

- `CalendarContext(seed, ...)` → 新的 `pvgis_location='wuhan'` 参数默认为wuhan，向后兼容
- `_build_8760h_sequence()` → 返回5元组: `(pv, load, tou, seasons, temp)`，旧4元组代码需添加 `_` 占位
- `simulate_daily_operation()` → 新增 `hourly_temp=None`, 旧调用无影响
- `get_total_load()` → 新增 `hourly_temp=None, hourly_ghi=None, month=1`
- `PVGenerator.generate_daily_profile()` → 新增 `hourly_temp=None, hourly_ghi=None`
- `BuildingLoadABM.simulate_hour()` → 新增 `temp=None, ghi=None`

### Design Decision Impact

- 天气数据从"统计合成"变为"2020年真实气象"，`MONTHLY_WEATHER_DAYS` 和 `WEATHER_TRANSITION_MATRIX` 降级为回退方案
- `PV_COEFF` 固定季节系数降级为回退方案（无PVGIS数据时使用）
- `MONTHLY_AMBIENT_TEMP` 月均值降级为回退方案
- 建筑负荷 `BUILDING_LOAD` 固定剖面降级为ABM失效时的回退

### Performance

- ABM首调用 ~16s (365天×24h物理计算)，后续缓存命中 ~6ms
- PSO (40pop×30iter) 完整运行约2-3分钟 (含ABM缓存)

---

**日期**: 2026-06-23  
**版本**: v6.6  
**状态**: 可运行，外部数据已集成

---

## 一、项目概述

高速公路服务区光储充一体化微网仿真系统，支持：
- 蒙特卡洛充电负荷预测
- 8760h 全年光伏+储能+负荷调度仿真
- PSO/NSGA-II 容量优化
- AHP-TOPSIS 决策框架
- 全生命周期经济评估
- SCADA Pro+ 实时监控界面

---

## 二、目录结构

```
系统/
├── simulation/
│   ├── config.py              # 核心参数（~900行，25份研究数据硬编码）
│   ├── data_loader.py         # 外部数据加载层（PVGIS/ACN/NASA）
│   ├── fetch_external_data.py # 批量拉取外部气象数据脚本
│   ├── web_server.py          # FastAPI 后端 v6.6
│   ├── run_pipeline.py        # 一键仿真全流程
│   │
│   ├── mc_charging_load.py    # 蒙特卡洛充电负荷
│   ├── pv_generation.py       # 8760h光伏合成
│   ├── capacity_optimization.py # PSO优化器
│   ├── nsga2.py               # NSGA-II多目标优化
│   ├── decision_framework.py  # AHP-TOPSIS决策
│   ├── economic_model.py      # 全生命周期经济
│   ├── calendar_utils.py      # 日历+天气Markov链
│   │
│   ├── data/                  # 外部数据缓存（已下载）
│   │   ├── pvgis_tmy_*.json   # 10城市TMY（PVGIS ERA5）
│   │   ├── nasa_power_wuhan.json
│   │   ├── acndata_full_distribution.json  # 85,877会话分析
│   │   ├── acndata_session_stats.json
│   │   └── tmy_comparison.json
│   │
│   ├── results/               # 仿真输出JSON
│   │   ├── mc_summary.json
│   │   ├── optimization_result.json
│   │   ├── pareto_results.json
│   │   ├── decision_result.json
│   │   └── daily_8760h.json
│   │
│   └── static/
│       ├── index.html         # 主前端
│       └── scada_pro_plus.html # SCADA Pro+ 界面
│
├── 高速服务区光储充一体化_研究数据集/  # 25份源数据Markdown
└── 仿真支撑数据分类.md        # 数据溯源说明
```

---

## 三、快速启动

### 1. 运行仿真全流程（生成results/*.json）
```bash
cd simulation
python run_pipeline.py
```

### 2. 启动Web服务
```bash
cd simulation
python web_server.py --port 8760
```
访问：
- 主界面: http://127.0.0.1:8760/
- SCADA Pro+: http://127.0.0.1:8760/static/scada_pro_plus.html

### 3. 拉取外部气象数据（可选，已有缓存）
```bash
cd simulation
python fetch_external_data.py --locations all --source all
```

---

## 四、关键API端点 (v6.6新增)

| 端点 | 功能 |
|------|------|
| `GET /api/health` | 健康检查 + 外部数据状态 |
| `GET /api/data/status` | PVGIS/ACN/NASA数据加载情况 |
| `GET /api/data/tmy-comparison` | PVGIS vs config.py 逐时GHI/温度对比 |
| `GET /api/data/tmy-multi-city` | 10城市TMY排名表 |
| `GET /api/data/acndata-distribution` | ACN-Data充电行为分布 |
| `GET /api/data/calibration` | 完整校准报告 + 建议 |

---

## 五、外部数据集成

### 已下载数据 (simulation/data/)

| 文件 | 来源 | 内容 |
|------|------|------|
| `pvgis_tmy_{city}.json` x10 | PVGIS ERA5 | 逐时GHI+温度+风速，12x24月度剖面 |
| `nasa_power_wuhan.json` | NASA POWER | 备选气象数据交叉验证 |
| `acndata_full_distribution.json` | Caltech ACN-Data | 85,877充电会话统计 |
| `acndata_session_stats.json` | ACN-Data | 能量/功率/时长分布 |
| `tmy_comparison.json` | 生成 | 10城市对比汇总 |

### 数据流向
```
simulation/data/ (预下载)
     │
     └── data_loader.py DataManager
              │
              ├── load_all_from_disk()  → 启动时自动加载
              └── get_calibration_api_data() → 供API调用
                       │
                       └── web_server.py → 前端面板
```

---

## 六、关键校准发现

基于PVGIS ERA5 vs config.py对比：

| 参数 | config.py | PVGIS实测 | 偏差 | 建议 |
|------|-----------|-----------|------|------|
| 夏季正午GHI | 1020 W/m2 | 832 W/m2 | +23% | 下调，当前高估光伏峰值 |
| 冬季上午GHI | 180 W/m2 | 366 W/m2 | -51% | 上调，当前低估冬季出力 |
| 年均GHI偏差 | - | - | 31.4% | 需要整体校准TMY_GHI_CLEAR |

ACN-Data适用性：
- ACN是**校园/职场Level 2慢充**数据，不适用高速DC快充场景
- 到达时间分布差异巨大（ACN夜间占19%，高速几乎为0）
- 能量分布形状可参考，但高速场景建议放大2-3倍

---

## 七、前端新功能 (v6.6)

1. **侧边栏数据状态**：显示已加载的PVGIS城市数和ACN会话数
2. **Header徽章**：`[PVGIS:10城]`
3. **分析建议页 > 五、外部数据对标**：
   - PVGIS vs config.py 逐时偏差表
   - 4条校准建议（高/中/低优先级）

---

## 八、核心类/函数说明

### data_loader.py
```python
class DataManager:
    def __init__(self, config_module=None)
    def load_all_from_disk()        # 自动扫描data/目录
    def get_calibration_api_data()  # 生成API响应结构
    def get_tmy_data(location)      # 获取TMY（优先外部，回退config）
    def compare_with_config()       # 生成偏差报告
```

### web_server.py 启动流程
```python
@app.on_event("startup")
def startup():
    # 1. 加载results/*.json缓存
    # 2. 预计算MC+8760h仿真缓存
    # 3. 初始化DataManager，加载外部数据
```

---

## 九、待办事项

- [ ] 根据PVGIS实测校准config.py中的TMY_GHI_CLEAR
- [ ] 寻找高速公路充电站实测数据替代ACN-Data
- [ ] 前端添加交互式城市选择器
- [ ] 支持用户上传自定义气象数据

---

## 十、依赖

```
fastapi
uvicorn
numpy
pandas
scipy
requests
openpyxl (可选，Excel支持)
pvlib (可选，PVGIS备用后端)
```

---

## 十一、联系方式

如有问题，检查：
1. `python --version` >= 3.8
2. `pip install -r requirements.txt`
3. 端口8760未被占用
4. `simulation/data/`目录存在且有JSON文件

---

*文档生成于 2026-06-23*
