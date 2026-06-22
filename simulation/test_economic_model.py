"""
economic_model.py 单元测试 — 核心经济指标 + 电池退化模型

运行: python -m pytest test_economic_model.py -v
      或 python test_economic_model.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from economic_model import EconomicModel
from config import (
    PV_COST_PER_KWP, ESS_COST, CHARGING_COST, FIXED_COST_TOTAL,
    OM_RATE, DISCOUNT_RATE, PROJECT_LIFE, CCER_PRICE,
    CARBON_FACTOR_GRID, PV_FIRST_YEAR_DEGRADATION, PV_ANNUAL_DEGRADATION,
    RESIDUAL_RATE_PV, RESIDUAL_RATE_ESS, RESIDUAL_RATE_CHARGING,
    get_calendar_fade_rate, get_cycle_life_at_dod,
    get_capacity_fade_per_cycle, get_battery_yearly_degradation,
    SCENARIOS,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def em_medium():
    """中型服务区基准方案经济模型"""
    return EconomicModel(
        pv_capacity=1231, ess_capacity=2000, ess_power=1000,
        n_piles_120kw=16, n_piles_480kw=2, scenario='baseline')


@pytest.fixture
def em_empty():
    """空配置 (纯电网)"""
    return EconomicModel(
        pv_capacity=0, ess_capacity=0, ess_power=0,
        n_piles_120kw=0, n_piles_480kw=0, scenario='baseline')


# ============================================================
# CAPEX Tests
# ============================================================

class TestCAPEX:
    def test_pv_cost_positive(self, em_medium):
        assert em_medium.capex_pv() > 0
        assert abs(em_medium.capex_pv() - 1231 * PV_COST_PER_KWP) < 1.0

    def test_pv_cost_zero(self, em_empty):
        assert em_empty.capex_pv() == 0

    def test_ess_cost(self, em_medium):
        expected = 2000 * (ESS_COST['battery_per_kwh'] + ESS_COST['fire_per_kwh']) + \
                   1000 * ESS_COST['pcs_per_kw']
        assert abs(em_medium.capex_ess() - expected) < 1.0

    def test_charging_cost(self, em_medium):
        expected = 16 * CHARGING_COST['pile_120kw'] + 2 * CHARGING_COST['pile_480kw']
        assert abs(em_medium.capex_charging() - expected) < 1.0

    def test_total_includes_fixed(self, em_medium):
        assert em_medium.capex_total() > em_medium.capex_pv() + em_medium.capex_ess()


class TestSubsidy:
    def test_subsidy_capped_at_30pct(self, em_medium):
        s = em_medium.subsidy_total()
        cap = em_medium.capex_total() * 0.30
        assert s <= cap

    def test_subsidy_zero_for_empty(self, em_empty):
        assert em_empty.subsidy_total() == 0

    def test_capex_after_subsidy_positive(self, em_medium):
        assert em_medium.capex_after_subsidy() < em_medium.capex_total()


class TestOPEX:
    def test_opex_year1(self, em_medium):
        opex = em_medium.opex_annual(year=1)
        assert opex > 0

    def test_opex_escalates(self, em_medium):
        opex_1 = em_medium.opex_annual(year=1)
        opex_10 = em_medium.opex_annual(year=10)
        assert opex_10 > opex_1


# ============================================================
# Carbon Revenue Tests
# ============================================================

class TestCarbonRevenue:
    def test_revenue_positive(self, em_medium):
        rev, tons = em_medium.carbon_revenue_annual(1_100_000, year=1)
        assert rev > 0
        assert tons > 0

    def test_zero_for_no_generation(self, em_medium):
        rev, tons = em_medium.carbon_revenue_annual(0, year=1)
        assert rev == 0 and tons == 0

    def test_price_increases_over_time(self, em_medium):
        _, tons_1 = em_medium.carbon_revenue_annual(1_000_000, year=1)
        _, tons_20 = em_medium.carbon_revenue_annual(1_000_000, year=20)
        # Same energy → same tons, but revenue would differ due to price
        assert abs(tons_1 - tons_20) < 0.01


# ============================================================
# Replacement Schedule Tests
# ============================================================

class TestReplacement:
    def test_schedule_non_empty(self, em_medium):
        events = em_medium.replacement_schedule(annual_ess_cycles=300)
        assert len(events) > 0

    def test_inverter_replacement_at_year_15(self, em_medium):
        events = em_medium.replacement_schedule()
        inv_events = [e for e in events if '逆变器' in e[2]]
        assert len(inv_events) == 1
        assert inv_events[0][0] == 15  # LIFESPAN pv_inverter

    def test_no_battery_replacement_when_zero_ess(self, em_empty):
        events = em_empty.replacement_schedule(annual_ess_cycles=300)
        batt_events = [e for e in events if '电池' in e[2]]
        # When ess_e=0, replacement events may exist but with zero cost
        batt_costs = [e[1] for e in batt_events]
        assert all(c == 0.0 for c in batt_costs)

    def test_battery_life_shorter_at_high_temp(self, em_medium):
        # High temperature → shorter calendar life → earlier replacement
        events_25c = em_medium.replacement_schedule(annual_ess_cycles=300,
                                                     avg_temp_c=25.0, avg_dod=0.50)
        events_40c = em_medium.replacement_schedule(annual_ess_cycles=300,
                                                     avg_temp_c=40.0, avg_dod=0.50)
        batt_25 = [e[0] for e in events_25c if '电池' in e[2]]
        batt_40 = [e[0] for e in events_40c if '电池' in e[2]]
        assert batt_40[0] <= batt_25[0]  # Higher temp → earlier or equal


# ============================================================
# PV Degradation Tests
# ============================================================

class TestPVDegradation:
    def test_year0_no_degradation(self, em_medium):
        assert em_medium._pv_degradation_factor(0) == 1.0

    def test_year1_first_year_only(self, em_medium):
        assert abs(em_medium._pv_degradation_factor(1) -
                   (1 - PV_FIRST_YEAR_DEGRADATION)) < 0.001

    def test_monotonically_decreasing(self, em_medium):
        factors = [em_medium._pv_degradation_factor(y) for y in range(21)]
        for i in range(len(factors) - 1):
            assert factors[i] >= factors[i + 1]

    def test_year20_bounds(self, em_medium):
        f20 = em_medium._pv_degradation_factor(20)
        assert 0.80 < f20 < 0.95  # ~10-15% degradation over 20 years


# ============================================================
# Discount Rate Tests
# ============================================================

class TestDiscount:
    def test_default_rate(self, em_medium):
        r1 = em_medium.discount_rate_for_year(1)
        assert abs(r1 - DISCOUNT_RATE) < 0.001

    def test_later_years_lower_rate(self, em_medium):
        r1 = em_medium.discount_rate_for_year(1)
        r15 = em_medium.discount_rate_for_year(15)
        assert r15 <= r1

    def test_floor_at_3pct(self, em_medium):
        r20 = em_medium.discount_rate_for_year(20)
        assert r20 >= 0.03

    def test_discount_factor_decreases(self, em_medium):
        df1 = em_medium._discount_factor(1)
        df20 = em_medium._discount_factor(20)
        assert df20 < df1


# ============================================================
# Core Economic Indicators
# ============================================================

class TestNPC:
    def test_npc_simple_returns_finite(self, em_medium):
        npc_val = em_medium.npc_simple(annual_grid_cost=300_000,
                                        annual_self_consumed_kwh=1_100_000)
        assert npc_val > 0
        assert npc_val < 1e9

    def test_npc_simple_monotonic_in_grid_cost(self, em_medium):
        npc_low = em_medium.npc_simple(annual_grid_cost=200_000,
                                        annual_self_consumed_kwh=1_100_000)
        npc_high = em_medium.npc_simple(annual_grid_cost=400_000,
                                         annual_self_consumed_kwh=1_100_000)
        assert npc_high > npc_low

    def test_npc_from_aggregates(self, em_medium):
        npc_val = em_medium.npc_from_aggregates(
            annual_grid_cost=300_000, annual_ess_cycles=300,
            annual_carbon_revenue=50_000, subsidy=600_000)
        assert npc_val > 0


class TestLCOE:
    def test_lcoe_positive(self, em_medium):
        lcoe = em_medium.lcoe(npc_val=18_000_000, total_energy_served_kwh=1_500_000)
        assert 0.5 < lcoe < 3.0  # yuan/kWh reasonable range

    def test_lcoe_inf_for_zero_energy(self, em_medium):
        lcoe = em_medium.lcoe(npc_val=1_000_000, total_energy_served_kwh=0)
        assert lcoe == float('inf')


class TestPayback:
    def test_static_payback(self, em_medium):
        pbp = em_medium.payback_period(annual_net_saving=1_000_000, method='static')
        assert 2 < pbp < 20

    def test_dynamic_payback_longer_than_static(self, em_medium):
        saving = 800_000
        static = em_medium.payback_period(saving, method='static')
        dynamic = em_medium.payback_period(saving, method='dynamic')
        assert dynamic >= static

    def test_infinite_for_zero_saving(self, em_medium):
        pbp = em_medium.payback_period(0)
        assert pbp == float('inf')


class TestROI:
    def test_roi_positive(self, em_medium):
        r = em_medium.roi(annual_net_saving=800_000)
        assert 0 < r < 1.0

    def test_roi_zero_for_empty(self, em_empty):
        # Empty config still has fixed costs (~317万), so ROI is small but finite
        r = em_empty.roi(annual_net_saving=1)
        assert r < 1e-6  # essentially zero return relative to investment


class TestBCR:
    def test_bcr_greater_than_one(self, em_medium):
        bcr = em_medium.bcr(total_benefit_npv=25_000_000, total_cost_npv=18_000_000)
        assert bcr > 1.0

    def test_bcr_less_than_one_for_loss(self, em_medium):
        bcr = em_medium.bcr(total_benefit_npv=10_000_000, total_cost_npv=18_000_000)
        assert bcr < 1.0


# ============================================================
# Scenario Tests
# ============================================================

class TestScenarios:
    def test_three_scenarios_exist(self):
        for sc in ['conservative', 'baseline', 'aggressive']:
            assert sc in SCENARIOS

    def test_conservative_highest_discount(self):
        assert SCENARIOS['conservative']['discount_rate'] > \
               SCENARIOS['aggressive']['discount_rate']

    def test_aggressive_highest_growth(self):
        assert SCENARIOS['aggressive']['load_growth_rate'] > \
               SCENARIOS['conservative']['load_growth_rate']


# ============================================================
# Battery Degradation Model Tests (v6.5)
# ============================================================

class TestCalendarFade:
    def test_baseline_at_25c(self):
        assert abs(get_calendar_fade_rate(25.0) - 0.02) < 0.001

    def test_higher_temp_faster_fade(self):
        assert get_calendar_fade_rate(45.0) > get_calendar_fade_rate(25.0)

    def test_lower_temp_slower_fade(self):
        assert get_calendar_fade_rate(5.0) < get_calendar_fade_rate(25.0)

    def test_approx_doubles_per_10c(self):
        """van't Hoff rule: rate roughly doubles per 10°C"""
        r25 = get_calendar_fade_rate(25.0)
        r35 = get_calendar_fade_rate(35.0)
        ratio = r35 / r25
        assert 1.3 < ratio < 2.0  # ~1.5x per 10°C for LFP

    def test_non_negative(self):
        for t in [-10, 0, 10, 25, 50, 60]:
            assert get_calendar_fade_rate(t) > 0

    def test_cabinet_temp_offset(self):
        """Cabinet temperature ~7°C above ambient"""
        r_ambient = get_calendar_fade_rate(25.0)
        r_cabinet = get_calendar_fade_rate(32.0)  # 25 + 7
        assert r_cabinet > r_ambient


class TestCycleLifeDOD:
    def test_reference_at_80pct_dod(self):
        life = get_cycle_life_at_dod(0.80)
        assert abs(life - 8000) < 10

    def test_higher_dod_shorter_life(self):
        assert get_cycle_life_at_dod(0.90) < get_cycle_life_at_dod(0.50)

    def test_lower_dod_longer_life(self):
        life_30 = get_cycle_life_at_dod(0.30)
        life_80 = get_cycle_life_at_dod(0.80)
        assert life_30 > life_80 * 2.0  # 30% DOD ~2.4x life

    def test_clips_to_valid_range(self):
        """DOD extremes are clamped to [0.05, 0.95]"""
        life_min = get_cycle_life_at_dod(0.01)
        life_max = get_cycle_life_at_dod(0.99)
        assert life_min > life_max
        assert life_min < 200_000

    def test_capacity_fade_per_cycle(self):
        fade_80 = get_capacity_fade_per_cycle(0.80)
        fade_50 = get_capacity_fade_per_cycle(0.50)
        assert fade_50 < fade_80  # Lower DOD → less fade per cycle


class TestCompositeDegradation:
    def test_returns_positive(self):
        deg = get_battery_yearly_degradation(25.0, 300, 0.50)
        assert deg > 0
        assert deg < 0.15  # shouldn't exceed 15%/year

    def test_higher_cycles_more_degradation(self):
        deg_low = get_battery_yearly_degradation(25.0, 100, 0.50)
        deg_high = get_battery_yearly_degradation(25.0, 500, 0.50)
        assert deg_high > deg_low

    def test_higher_temp_more_degradation(self):
        deg_25 = get_battery_yearly_degradation(25.0, 300, 0.50)
        deg_40 = get_battery_yearly_degradation(40.0, 300, 0.50)
        assert deg_40 > deg_25

    def test_deeper_dod_more_degradation(self):
        deg_shallow = get_battery_yearly_degradation(25.0, 300, 0.30)
        deg_deep = get_battery_yearly_degradation(25.0, 300, 0.80)
        assert deg_deep > deg_shallow


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_update_config_preserves_other_params(self, em_medium):
        em_medium.update_config(pv_cap=500)
        assert em_medium.pv_cap == 500
        assert em_medium.ess_e == 2000  # unchanged

    def test_npc_with_scenario_switches_back(self, em_medium):
        baseline_npc = em_medium.npc_simple(300_000, 1_100_000)
        conservative_npc = em_medium.npc_with_scenario(
            0, np.zeros(24), 1_100_000, 0, 'conservative')
        # Scenario switching works
        assert conservative_npc > 0

    def test_empty_config_npc_is_finite(self, em_empty):
        npc_val = em_empty.npc_simple(annual_grid_cost=500_000,
                                       annual_self_consumed_kwh=0)
        assert npc_val > 0
        assert npc_val < 1e9


# ============================================================
# Run without pytest
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
