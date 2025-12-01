"""
Integration Tests for EcoAssist
Tests integration between components and database
"""

import pytest
import pandas as pd
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database_integration import (
    DatabaseConfig, DatabaseManager, PropertyRepository,
    ReductionOptionRepository, MilestoneScenarioRepository,
    EcoAssistBackend, initialize_database_backend
)

# ================================================================================
# FIXTURES
# ================================================================================

@pytest.fixture(scope="module")
def test_db_config():
    """Create test database configuration"""
    config = DatabaseConfig()
    # Use test database
    config.database = "EcoAssistDB_Test"
    return config

@pytest.fixture(scope="module")
def db_manager(test_db_config):
    """Create database manager for testing"""
    return DatabaseManager(test_db_config)

@pytest.fixture(scope="function")
def clean_database(db_manager):
    """Clean database before each test"""
    # This would clean test data - implement based on your needs
    yield
    # Cleanup after test

# ================================================================================
# DATABASE CONNECTION TESTS
# ================================================================================

class TestDatabaseConnection:
    """Test database connectivity"""
    
    def test_database_connection_succeeds(self, db_manager):
        """Test that database connection is successful"""
        try:
            result = db_manager.execute_scalar("SELECT 1")
            assert result == 1
        except Exception as e:
            pytest.fail(f"Database connection failed: {e}")
    
    def test_database_has_required_tables(self, db_manager):
        """Test that all required tables exist"""
        query = """
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_TYPE = 'BASE TABLE' 
        AND TABLE_CATALOG = 'EcoAssistDB'
        """
        
        tables = db_manager.execute_query(query)
        table_names = [t['TABLE_NAME'] for t in tables]
        
        required_tables = [
            'properties', 'reduction_options', 'strategic_patterns',
            'milestone_scenarios', 'historical_consumption', 
            'historical_emissions', 'carbon_credit_prices',
            'renewable_energy_prices', 'renewable_fuel_prices'
        ]
        
        for table in required_tables:
            assert table in table_names, f"Table {table} not found in database"

# ================================================================================
# REPOSITORY INTEGRATION TESTS
# ================================================================================

class TestPropertyRepository:
    """Test PropertyRepository integration"""
    
    def test_get_all_active_properties(self, db_manager):
        """Test retrieving all active properties"""
        repo = PropertyRepository(db_manager)
        properties = repo.get_all_active()
        
        assert isinstance(properties, list)
        assert len(properties) > 0
        assert all(prop.is_active for prop in properties)
    
    def test_get_property_by_id(self, db_manager):
        """Test retrieving property by ID"""
        repo = PropertyRepository(db_manager)
        prop = repo.get_by_id("BP01")
        
        assert prop is not None
        assert prop.property_id == "BP01"
        assert prop.area_sqm > 0
    
    def test_get_nonexistent_property_returns_none(self, db_manager):
        """Test that nonexistent property returns None"""
        repo = PropertyRepository(db_manager)
        prop = repo.get_by_id("NONEXISTENT")
        
        assert prop is None

class TestReductionOptionRepository:
    """Test ReductionOptionRepository integration"""
    
    def test_get_all_reduction_options(self, db_manager):
        """Test retrieving all reduction options"""
        repo = ReductionOptionRepository(db_manager)
        options = repo.get_all()
        
        assert isinstance(options, list)
        assert len(options) > 0
        assert all(hasattr(opt, 'option_id') for opt in options)
    
    def test_reduction_options_have_valid_data(self, db_manager):
        """Test that reduction options have valid data"""
        repo = ReductionOptionRepository(db_manager)
        options = repo.get_all()
        
        for option in options:
            assert option.co2_reduction_potential >= 0
            assert option.capex >= 0
            assert 1 <= option.priority <= 5

class TestMilestoneScenarioRepository:
    """Test MilestoneScenarioRepository integration"""
    
    def test_create_and_retrieve_scenario(self, db_manager):
        """Test creating and retrieving a milestone scenario"""
        from database_integration import MilestoneScenario
        
        repo = MilestoneScenarioRepository(db_manager)
        
        # Create test scenario
        test_scenario = MilestoneScenario(
            scenario_id=f"TEST_{int(datetime.now().timestamp())}",
            name="Test Scenario",
            description="Integration test scenario",
            target_year=2050,
            yearly_targets={2025: 100000, 2030: 70000, 2050: 20000},
            total_capex=1000000,
            total_opex=500000,
            reduction_rate_2030=30.0,
            reduction_rate_2050=80.0,
            strategy_type="test"
        )
        
        # Save to database
        scenario_id = repo.create(test_scenario)
        assert scenario_id is not None
        
        # Retrieve from database
        retrieved = repo.get_by_id(scenario_id)
        assert retrieved is not None
        assert retrieved.scenario_id == scenario_id
        assert retrieved.name == "Test Scenario"

# ================================================================================
# BACKEND INTEGRATION TESTS
# ================================================================================

class TestBackendIntegration:
    """Test EcoAssistBackend integration with database"""
    
    def test_backend_initialization_loads_data(self, test_db_config):
        """Test that backend loads data from database"""
        backend = EcoAssistBackend(test_db_config)
        
        assert len(backend.properties) > 0
        assert len(backend.reduction_options) > 0
        assert len(backend.strategic_patterns) > 0
    
    def test_generate_and_persist_scenarios(self, test_db_config):
        """Test generating scenarios and persisting to database"""
        backend = EcoAssistBackend(test_db_config)
        
        initial_count = len(backend.milestone_scenarios)
        
        # Generate new scenarios
        scenarios = backend.generate_milestone_scenarios(2050, 35.0, 85.0)
        
        assert len(scenarios) == 4
        # Verify scenarios were persisted
        assert len(backend.milestone_scenarios) > initial_count
    
    def test_end_to_end_milestone_workflow(self, test_db_config):
        """Test complete milestone workflow"""
        backend = EcoAssistBackend(test_db_config)
        
        # 1. Generate scenarios
        scenarios = backend.generate_milestone_scenarios(2050, 30.0, 80.0)
        assert len(scenarios) > 0
        
        # 2. Calculate property breakdown
        df, status = backend.calculate_property_breakdown(scenarios[0].name)
        assert df is not None
        assert len(df) > 0
        
        # 3. Verify data consistency
        assert "success" in status.lower()

# ================================================================================
# PRICING DATA INTEGRATION TESTS
# ================================================================================

class TestPricingDataIntegration:
    """Test pricing data integration"""
    
    def test_carbon_credit_prices_available(self, db_manager):
        """Test that carbon credit prices are available"""
        query = "SELECT COUNT(*) as count FROM carbon_credit_prices"
        result = db_manager.execute_scalar(query)
        
        assert result > 0, "No carbon credit price data found"
    
    def test_renewable_energy_prices_available(self, db_manager):
        """Test that renewable energy prices are available"""
        query = "SELECT COUNT(*) as count FROM renewable_energy_prices"
        result = db_manager.execute_scalar(query)
        
        assert result > 0, "No renewable energy price data found"
    
    def test_renewable_fuel_prices_available(self, db_manager):
        """Test that renewable fuel prices are available"""
        query = "SELECT COUNT(*) as count FROM renewable_fuel_prices"
        result = db_manager.execute_scalar(query)
        
        assert result > 0, "No renewable fuel price data found"
    
    def test_pricing_data_has_valid_years(self, db_manager):
        """Test that pricing data has valid year ranges"""
        query = """
        SELECT MIN(year) as min_year, MAX(year) as max_year 
        FROM carbon_credit_prices
        """
        result = db_manager.execute_query(query)[0]
        
        current_year = datetime.now().year
        assert result['min_year'] <= current_year
        assert result['max_year'] >= current_year

# ================================================================================
# DATA CONSISTENCY TESTS
# ================================================================================

class TestDataConsistency:
    """Test data consistency across tables"""
    
    def test_historical_data_matches_properties(self, db_manager):
        """Test that historical data references valid properties"""
        query = """
        SELECT DISTINCT hc.property_id
        FROM historical_consumption hc
        LEFT JOIN properties p ON hc.property_id = p.property_id
        WHERE p.property_id IS NULL
        """
        
        orphaned_records = db_manager.execute_query(query)
        assert len(orphaned_records) == 0, "Found orphaned historical consumption records"
    
    def test_property_targets_reference_valid_scenarios(self, db_manager):
        """Test that property targets reference valid scenarios"""
        query = """
        SELECT COUNT(*) as count
        FROM property_targets pt
        LEFT JOIN milestone_scenarios ms ON pt.scenario_id = ms.scenario_id
        WHERE ms.scenario_id IS NULL
        """
        
        result = db_manager.execute_scalar(query)
        assert result == 0, "Found property targets with invalid scenario references"

# ================================================================================
# RUN TESTS
# ================================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])