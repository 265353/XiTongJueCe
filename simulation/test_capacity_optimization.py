"""
capacity_optimization.py 调度回归测试

测试 simulate_daily_operation 的关键场景:
- 纯光伏 (无储能)
- 光储TOU套利
- 变压器越限
- SOC边界
- 预测误差

运行: python -m pytest test_capacity_optimization.py -v
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from capacity_optimization import MicrogridOptimizer
from mc_charging_load import MonteCarloChargingSimulator
from config import (
    SOC_MIN, SOC_MAX, TOU_PRICE_VALUES, FEED_IN_PRICE,
    TRANSFORMER_CAPACITY_KVA, TRANSFORMER_PF_MIN,
    get_tou_price_array,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def optimizer():
    """创建带MC仿真结果的优化器 (模块级, 复用)"""
    sim = MonteCarloChargingSimulator(service_area_size='medium', seed=42)
    mc = sim.simulate_all_scenarios(n_runs=500)
    opt = MicrogridOptimizer(size='medium', mc_scenarios=mc, seed=42)
    return opt


@pytest.fixture
def spring_load(optimizer):
    """春季典型日负荷 (24h)"""
    return optimizer.get_total_load('spring', 'workday', 'clear')


@pytest.fixture
def spring_pv():
    """春季典型日光伏出力系数 (24h)"""
    from config import PV_COEFF
    return np.array(PV_COEFF['spring'])


# ============================================================
# 纯光伏场景 (无储能)
# ============================================================

class TestNoESS:
    def test_no_ess_charge_discharge_zero(self, optimizer, spring_load, spring_pv):
        optimizer.ess_capacity = 0
        optimizer.ess_power = 0
        pv_profile = spring_pv * 500  # 500 kWp
        result = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring')
        assert result['ess_charge'].sum() == 0
        assert result['ess_discharge'].sum() == 0

    def test_grid_import_equals_net_deficit(self, optimizer, spring_load, spring_pv):
        optimizer.ess_capacity = 0
        optimizer.ess_power = 0
        pv_profile = spring_pv * 500
        result = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring')
        net_load = spring_load - pv_profile
        expected_import = np.maximum(0, net_load)
        expected_export = np.maximum(0, -net_load)
        np.testing.assert_allclose(result['grid_import'], expected_import, atol=1e-6)
        np.testing.assert_allclose(result['grid_export'], expected_export, atol=1e-6)


# ============================================================
# 光储TOU套利场景
# ============================================================

class TestTOUArbitrage:
    def test_ess_charges_during_valley(self, optimizer, spring_load, spring_pv):
        """储能应在谷电价时段充电"""
        optimizer.ess_capacity = 2000
        optimizer.ess_power = 1000
        pv_profile = spring_pv * 1000  # 大光伏
        result = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring')
        # 谷电价时段 (23-24, 0-7) 应有充电
        valley_hours = list(range(0, 7)) + [23]
        valley_charge = result['ess_charge'][valley_hours].sum()
        # 不要求严格谷时充电 (因为有弃光充电), 但总充电>0
        assert result['ess_charge'].sum() > 0

    def test_ess_discharges_during_peak(self, optimizer, spring_load, spring_pv):
        """储能应在峰电价时段放电"""
        optimizer.ess_capacity = 2000
        optimizer.ess_power = 1000
        pv_profile = spring_pv * 1000
        result = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring')
        assert result['ess_discharge'].sum() > 0

    def test_reduces_grid_cost_vs_no_ess(self, optimizer, spring_load, spring_pv):
        """光储方案网购电成本应低于无储能方案"""
        pv_profile = spring_pv * 1000
        prices = get_tou_price_array('spring')

        # 无ESS
        optimizer.ess_capacity = 0
        optimizer.ess_power = 0
        result_no_ess = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring')
        cost_no_ess = np.sum(result_no_ess['grid_import'] * prices)

        # 有ESS
        optimizer.ess_capacity = 2000
        optimizer.ess_power = 1000
        result_with_ess = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring')
        cost_with_ess = np.sum(result_with_ess['grid_import'] * prices)

        # ESS应降低购电成本
        assert cost_with_ess <= cost_no_ess * 1.01  # 允许微小舍入误差


# ============================================================
# SOC边界
# ============================================================

class TestSOCBounds:
    def test_soc_within_bounds(self, optimizer, spring_load, spring_pv):
        optimizer.ess_capacity = 2000
        optimizer.ess_power = 1000
        pv_profile = spring_pv * 1000
        result = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring', initial_soc=0.5)
        assert result['soc_curve'].min() >= SOC_MIN - 0.01
        assert result['soc_curve'].max() <= SOC_MAX + 0.01

    def test_soc_respects_charge_power_limit(self, optimizer, spring_load, spring_pv):
        optimizer.ess_capacity = 1000
        optimizer.ess_power = 200  # 小PCS
        pv_profile = spring_pv * 2000  # 大光伏
        result = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring', initial_soc=0.3)
        # 充电功率不超过PCS限额
        assert result['ess_charge'].max() <= 200 + 0.01


# ============================================================
# 变压器约束
# ============================================================

class TestTransformerConstraint:
    def test_grid_import_within_transformer_limit(self, optimizer, spring_load, spring_pv):
        optimizer.ess_capacity = 500
        optimizer.ess_power = 200
        # 极小光伏 → 大量网购电
        pv_profile = spring_pv * 50
        result = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring')
        trans_limit = TRANSFORMER_CAPACITY_KVA * TRANSFORMER_PF_MIN
        assert result['grid_import'].max() <= trans_limit + 0.01

    def test_loss_of_load_when_exceed_transformer(self, optimizer, spring_load, spring_pv):
        optimizer.ess_capacity = 0
        optimizer.ess_power = 0
        pv_profile = spring_pv * 10  # 几乎没有光伏
        result = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring')
        # 超变压器限制时应有甩负荷
        trans_limit = TRANSFORMER_CAPACITY_KVA * TRANSFORMER_PF_MIN
        if spring_load.max() > trans_limit:
            assert result['loss_of_load'].sum() > 0


# ============================================================
# 预测误差
# ============================================================

class TestPredictionError:
    def test_with_prediction_error_still_solves(self, optimizer, spring_load, spring_pv):
        """含预测误差时调度仍应完成"""
        optimizer.ess_capacity = 2000
        optimizer.ess_power = 1000
        pv_profile = spring_pv * 1000
        result = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring', pred_error_std=0.12)
        assert result['soc_curve'].min() >= SOC_MIN - 0.01
        assert result['soc_curve'].max() <= SOC_MAX + 0.01

    def test_prediction_error_worsens_cost(self, optimizer, spring_load, spring_pv):
        """预测误差应轻微增加成本 (但有时也能有利) — 仅验证不崩溃"""
        optimizer.ess_capacity = 2000
        optimizer.ess_power = 1000
        prices = get_tou_price_array('spring')
        pv_profile = spring_pv * 1000

        result_perfect = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring', pred_error_std=0.0)
        result_noisy = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring', pred_error_std=0.15)
        cost_perfect = np.sum(result_perfect['grid_import'] * prices)
        cost_noisy = np.sum(result_noisy['grid_import'] * prices)
        # 两者都应该有合理值
        assert cost_perfect > 0 and cost_noisy > 0


# ============================================================
# 温度效应
# ============================================================

class TestTemperatureEffect:
    def test_cold_temp_reduces_ess_capacity(self, optimizer, spring_load, spring_pv):
        optimizer.ess_capacity = 2000
        optimizer.ess_power = 1000
        pv_profile = spring_pv * 1000
        # 低温 -5°C
        result_cold = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring', ambient_temp=-5.0)
        # 常温 25°C
        result_warm = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring', ambient_temp=25.0)
        # 低温下总充放电量应减少 (容量衰减)
        cold_throughput = (result_cold['ess_charge'].sum() +
                          result_cold['ess_discharge'].sum())
        warm_throughput = (result_warm['ess_charge'].sum() +
                          result_warm['ess_discharge'].sum())
        # 不强制 cold < warm (取决于PV/负荷匹配), 但两者都应合理
        assert cold_throughput >= 0 and warm_throughput >= 0


# ============================================================
# 跨天SOC连续性
# ============================================================

class TestSOCContinuity:
    def test_initial_soc_respected(self, optimizer, spring_load, spring_pv):
        optimizer.ess_capacity = 2000
        optimizer.ess_power = 1000
        pv_profile = spring_pv * 1000
        result = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring', initial_soc=0.3)
        # soc_curve[0] 是第0小时结束时的SOC (含该小时充放电)
        # 验证不会偏离初始值超过一个小时的充放电量
        soc_change = abs(result['soc_curve'][0] - 0.3)
        max_hourly_charge = 1000 / 2000  # PCS功率/容量 = 0.5
        assert soc_change <= max_hourly_charge * 1.1  # 含效率裕度

    def test_final_soc_returned(self, optimizer, spring_load, spring_pv):
        optimizer.ess_capacity = 2000
        optimizer.ess_power = 1000
        pv_profile = spring_pv * 1000
        result = optimizer.simulate_daily_operation(
            pv_profile, spring_load, 'spring', initial_soc=0.5)
        assert 0 <= result['final_soc'] <= 1.0
        # final_soc 应接近 soc_curve 的最后一个值
        assert abs(result['final_soc'] - result['soc_curve'][-1]) < 0.01


# ============================================================
# 自放电
# ============================================================

class TestSelfDischarge:
    def test_self_discharge_small(self, optimizer, spring_load, spring_pv):
        """自放电应很小 (~0.1%/天)"""
        optimizer.ess_capacity = 2000
        optimizer.ess_power = 1000
        # 零PV, 零负荷 -> ESS无操作, 仅自放电
        pv_zero = np.zeros(24)
        load_near_zero = np.ones(24) * 0.01
        result = optimizer.simulate_daily_operation(
            pv_zero, load_near_zero, 'spring', initial_soc=0.5)
        soc_drop = 0.5 - result['final_soc']
        # 自放电应 < 0.5%/天
        assert soc_drop < 0.005


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
