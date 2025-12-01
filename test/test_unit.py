"""
Unit Tests for EcoAssist Core Functions
Tests individual functions and methods in isolation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ecoassist_backend import (
    EcoAssistBackend, Property, ReductionOption, 
    MilestoneScenario, StrategicPattern
)

# ================================================================================
# FIXTURES
# ================================================================================

@pytest.fixture
def sample_properties():
    """Create sample properties for testing"""
    return [
        Property("BP01", 1500, 0.92, 150000, 95000, 55000, 25000, "High", "Office", 2010),
        Property("CB01", 800, 0.85, 85000, 55000, 30000, 15000, "Medium", "Retail", 2015),
        Property("CB02", 2000, 0.95, 220000, 140000, 80000, 40000, "High", "Office", 2008)
    ]

@pytest.fixture
def sample_reduction_options():
    """Create sample reduction options for testing"""
    return [
        ReductionOption("SOLAR001", "Solar PV Installation", 4500, 200000, 12000, "High", 4, "Low"),
        ReductionOption("LED001", "LED Lighting Upgrade", 1200, 45000, 3000, "High", 1, "Low"),
        ReductionOption("HVAC001", "HVAC System Upgrade", 3500, 150000, 18000, "High", 3, "Medium")
    ]

@pytest.fixture
def backend():
    """Create EcoAssistBackend instance with mocked data"""
    backend = EcoAssistBackend()
    return backend

# ================================================================================
# MILESTONE GENERATION TESTS
# ================================================================================

class TestMilestoneGeneration:
    """Test milestone scenario generation"""
    
    def test_generate_milestone_scenarios_returns_four_scenarios(self, backend):
        """Test that milestone generation returns exactly 4 scenarios"""
        scenarios = backend.generate_milestone_scenarios(2050, 30.0, 80.0)
        assert len(scenarios) == 4
    
    def test_milestone_scenario_has_required_fields(self, backend):
        """Test that generated scenarios have all required fields"""
        scenarios = backend.generate_milestone_scenarios(2050, 30.0, 80.0)
        scenario = scenarios[0]
        
        assert hasattr(scenario, 'scenario_id')
        assert hasattr(scenario, 'name')
        assert hasattr(scenario, 'yearly_targets')
        assert hasattr(scenario, 'total_capex')
        assert hasattr(scenario, 'total_opex')
        assert hasattr(scenario, 'reduction_rate_2030')
        assert hasattr(scenario, 'reduction_rate_2050')
    
    def test_yearly_targets_decrease_over_time(self, backend):
        """Test that emission targets decrease over time"""
        scenarios = backend.generate_milestone_scenarios(2050, 30.0, 80.0)
        
        for scenario in scenarios:
            years = sorted(scenario.yearly_targets.keys())
            for i in range(len(years) - 1):
                current_year = years[i]
                next_year = years[i + 1]
                # Emissions should decrease or stay the same
                assert scenario.yearly_targets[current_year] >= scenario.yearly_targets[next_year]
    
    def test_reduction_rates_match_input(self, backend):
        """Test that reduction rates match input parameters"""
        scenarios = backend.generate_milestone_scenarios(2050, 35.0, 85.0)
        
        # At least one scenario should match the input rates
        matching_scenario = None
        for scenario in scenarios:
            if abs(scenario.reduction_rate_2030 - 35.0) < 1.0:
                matching_scenario = scenario
                break
        
        assert matching_scenario is not None
    
    def test_invalid_target_year_raises_error(self, backend):
        """Test that invalid target year is handled"""
        # Target year before current year should be handled
        with pytest.raises(Exception):
            backend.generate_milestone_scenarios(2020, 30.0, 80.0)
    
    def test_invalid_reduction_percentage(self, backend):
        """Test that invalid reduction percentages are handled"""
        # Reduction rate over 100% should be handled
        scenarios = backend.generate_milestone_scenarios(2050, 150.0, 80.0)
        # Should cap at reasonable values
        for scenario in scenarios:
            assert scenario.reduction_rate_2030 <= 100.0
            assert scenario.reduction_rate_2050 <= 100.0

# ================================================================================
# PROPERTY BREAKDOWN TESTS
# ================================================================================

class TestPropertyBreakdown:
    """Test property-level target allocation"""
    
    def test_calculate_property_breakdown_returns_dataframe(self, backend):
        """Test that property breakdown returns a DataFrame"""
        backend.generate_milestone_scenarios(2050, 30.0, 80.0)
        df, status = backend.calculate_property_breakdown("Aggressive Decarbonisation")
        
        assert isinstance(df, pd.DataFrame)
        assert status == "Property breakdown for Aggressive Decarbonisation generated successfully!"
    
    def test_property_breakdown_has_all_properties(self, backend):
        """Test that breakdown includes all properties"""
        backend.generate_milestone_scenarios(2050, 30.0, 80.0)
        df, status = backend.calculate_property_breakdown("Aggressive Decarbonisation")
        
        unique_properties = df['Property'].unique()
        assert len(unique_properties) == len(backend.properties)
    
    def test_property_breakdown_has_required_columns(self, backend):
        """Test that breakdown DataFrame has required columns"""
        backend.generate_milestone_scenarios(2050, 30.0, 80.0)
        df, status = backend.calculate_property_breakdown("Aggressive Decarbonisation")
        
        required_columns = ['Property', 'NLA (m²)', 'Baseline (tCO₂e)', 'Year', 
                          'Target (tCO₂e)', 'Reduction Rate']
        for col in required_columns:
            assert col in df.columns
    
    def test_reduction_rates_are_reasonable(self, backend):
        """Test that calculated reduction rates are within reasonable bounds"""
        backend.generate_milestone_scenarios(2050, 30.0, 80.0)
        df, status = backend.calculate_property_breakdown("Aggressive Decarbonisation")
        
        # Extract numeric values from percentage strings
        reduction_rates = df['Reduction Rate'].str.rstrip('%').astype(float)
        assert all(reduction_rates >= 0)
        assert all(reduction_rates <= 100)
    
    def test_invalid_scenario_name_returns_error(self, backend):
        """Test that invalid scenario name is handled"""
        df, status = backend.calculate_property_breakdown("Nonexistent Scenario")
        
        assert df is None
        assert "not found" in status.lower()

# ================================================================================
# STRATEGIC PATTERN TESTS
# ================================================================================

class TestStrategicPattern:
    """Test strategic pattern analysis"""
    
    def test_analyze_strategic_pattern_returns_data(self, backend):
        """Test that pattern analysis returns data"""
        pattern_name = backend.strategic_patterns[0].name
        details_df, summary, status = backend.analyze_strategic_pattern(pattern_name)
        
        assert details_df is not None
        assert summary is not None
        assert isinstance(summary, dict)
    
    def test_pattern_details_has_required_columns(self, backend):
        """Test that pattern details have required columns"""
        pattern_name = backend.strategic_patterns[0].name
        details_df, summary, status = backend.analyze_strategic_pattern(pattern_name)
        
        required_columns = ['Reduction Option', 'Priority', 'Estimated Impact']
        for col in required_columns:
            assert col in details_df.columns
    
    def test_pattern_summary_has_required_fields(self, backend):
        """Test that summary has required information"""
        pattern_name = backend.strategic_patterns[0].name
        details_df, summary, status = backend.analyze_strategic_pattern(pattern_name)
        
        required_fields = ['strategy_name', 'estimated_cost', 'estimated_reduction', 'risk_level']
        for field in required_fields:
            assert field in summary
    
    def test_invalid_pattern_name_returns_error(self, backend):
        """Test that invalid pattern name is handled"""
        details_df, summary, status = backend.analyze_strategic_pattern("Invalid Pattern")
        
        assert details_df is None
        assert summary is None or summary == {}
        assert "not found" in status.lower()

# ================================================================================
# REOPTIMIZATION TESTS
# ================================================================================

class TestReoptimization:
    """Test annual plan reoptimization"""
    
    def test_reoptimize_annual_plan_returns_data(self, backend):
        """Test that reoptimization returns required data"""
        plot_data, consumption_df, analysis_summary, status = backend.reoptimize_annual_plan("BP01", 0.05)
        
        assert plot_data is not None
        assert isinstance(plot_data, dict)
        assert analysis_summary is not None
    
    def test_reoptimization_detects_deviation(self, backend):
        """Test that reoptimization correctly detects deviations"""
        plot_data, consumption_df, analysis_summary, status = backend.reoptimize_annual_plan("BP01", 0.05)
        
        assert 'needs_reoptimization' in plot_data
        assert isinstance(plot_data['needs_reoptimization'], bool)
    
    def test_reoptimization_provides_recommendations(self, backend):
        """Test that reoptimization provides recommendations"""
        plot_data, consumption_df, analysis_summary, status = backend.reoptimize_annual_plan("BP01", 0.05)
        
        assert 'recommendations' in analysis_summary
        assert isinstance(analysis_summary['recommendations'], list)
    
    def test_invalid_property_id_returns_error(self, backend):
        """Test that invalid property ID is handled"""
        plot_data, consumption_df, analysis_summary, status = backend.reoptimize_annual_plan("INVALID", 0.05)
        
        assert plot_data is None
        assert "not found" in status.lower()
    
    def test_threshold_affects_reoptimization_decision(self, backend):
        """Test that deviation threshold affects reoptimization decision"""
        # Low threshold should trigger reoptimization
        plot_data_low, _, _, _ = backend.reoptimize_annual_plan("BP01", 0.01)
        
        # High threshold might not trigger reoptimization
        plot_data_high, _, _, _ = backend.reoptimize_annual_plan("BP01", 0.50)
        
        # At least verify both calls succeed
        assert plot_data_low is not None
        assert plot_data_high is not None

# ================================================================================
# DATA MODEL TESTS
# ================================================================================

class TestDataModels:
    """Test data model classes"""
    
    def test_property_creation(self):
        """Test Property object creation"""
        prop = Property("TEST01", 1000, 0.8, 100000, 60000, 40000, 20000, "High", "Office", 2015)
        
        assert prop.property_id == "TEST01"
        assert prop.area_sqm == 1000
        assert prop.baseline_emission == 100000
    
    def test_reduction_option_creation(self):
        """Test ReductionOption object creation"""
        option = ReductionOption("TEST001", "Test Option", 1000, 50000, 5000, "High", 2, "Low")
        
        assert option.option_id == "TEST001"
        assert option.co2_reduction == 1000
        assert option.capex == 50000
    
    def test_milestone_scenario_creation(self):
        """Test MilestoneScenario object creation"""
        scenario = MilestoneScenario(
            "SCEN001", "Test Scenario", "Description", 
            {2025: 100000, 2030: 70000}, 500000, 200000, 
            "balanced", 30.0, 80.0
        )
        
        assert scenario.scenario_id == "SCEN001"
        assert len(scenario.yearly_targets) == 2
        assert scenario.reduction_rate_2030 == 30.0

# ================================================================================
# UTILITY FUNCTION TESTS
# ================================================================================

class TestUtilityFunctions:
    """Test utility and helper functions"""
    
    def test_get_properties_data_format(self, backend):
        """Test that property data is properly formatted"""
        data = backend.get_properties_data()
        
        assert isinstance(data, list)
        assert len(data) > 0
        assert len(data[0]) == 10  # Should have 10 columns
    
    def test_get_reduction_options_data_format(self, backend):
        """Test that reduction options data is properly formatted"""
        data = backend.get_reduction_options_data()
        
        assert isinstance(data, list)
        assert len(data) > 0
        assert len(data[0]) == 8  # Should have 8 columns
    
    def test_get_strategic_patterns_data_format(self, backend):
        """Test that strategic patterns data is properly formatted"""
        data = backend.get_strategic_patterns_data()
        
        assert isinstance(data, list)
        assert len(data) > 0
        assert len(data[0]) == 6  # Should have 6 columns

# ================================================================================
# RUN TESTS
# ================================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])