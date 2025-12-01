"""
API Endpoint Tests for EcoAssist REST API
Tests all REST API endpoints with various scenarios
"""

import pytest
import requests
import json
from typing import Dict, Any
import time

# ================================================================================
# CONFIGURATION
# ================================================================================

API_BASE_URL = "http://localhost:7860/api/v1"
TEST_TIMEOUT = 30  # seconds

# ================================================================================
# FIXTURES
# ================================================================================

@pytest.fixture(scope="module")
def api_session():
    """Create requests session for API testing"""
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json"
    })
    return session

@pytest.fixture(scope="module")
def auth_token(api_session):
    """Get authentication token (if needed)"""
    # Implement authentication if required
    # For now, return None
    return None

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def make_request(session, method: str, endpoint: str, **kwargs) -> requests.Response:
    """Make API request with error handling"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = session.request(method, url, timeout=TEST_TIMEOUT, **kwargs)
        return response
    except requests.exceptions.Timeout:
        pytest.fail(f"Request to {endpoint} timed out after {TEST_TIMEOUT} seconds")
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Request to {endpoint} failed: {e}")

def assert_response_success(response: requests.Response):
    """Assert that response is successful"""
    assert response.status_code in [200, 201], \
        f"Expected success status code, got {response.status_code}: {response.text}"
    
    data = response.json()
    assert data.get("status") == "success", \
        f"Expected success status, got: {data.get('status')}"

def assert_response_has_data(response: requests.Response):
    """Assert that response contains data"""
    data = response.json()
    assert "data" in data, "Response missing 'data' field"
    assert data["data"] is not None, "Response 'data' field is None"

# ================================================================================
# MODULE 1: MILESTONE SETTING API TESTS
# ================================================================================

class TestMilestoneAPIs:
    """Test Milestone Setting APIs"""
    
    def test_calculate_milestones_success(self, api_session):
        """Test POST /milestones/calculate"""
        payload = {
            "target_year": 2050,
            "reduction_2030": 30.0,
            "reduction_2050": 80.0
        }
        
        response = make_request(api_session, "POST", "/milestones/calculate", json=payload)
        assert_response_success(response)
        assert_response_has_data(response)
        
        data = response.json()["data"]
        assert "scenarios" in data
        assert len(data["scenarios"]) > 0
        assert "calculation_id" in data
    
    def test_calculate_milestones_with_custom_scenario(self, api_session):
        """Test milestone calculation with custom scenario"""
        payload = {
            "target_year": 2050,
            "reduction_2030": 30.0,
            "reduction_2050": 80.0,
            "custom_scenario": {
                "name": "Test Custom Scenario",
                "reduction_2030": 35.0,
                "reduction_2050": 85.0
            }
        }
        
        response = make_request(api_session, "POST", "/milestones/calculate", json=payload)
        assert_response_success(response)
        
        data = response.json()["data"]
        scenarios = data["scenarios"]
        # Should have standard scenarios plus custom
        assert len(scenarios) >= 4
    
    def test_calculate_milestones_invalid_year(self, api_session):
        """Test milestone calculation with invalid year"""
        payload = {
            "target_year": 2020,  # Year in the past
            "reduction_2030": 30.0,
            "reduction_2050": 80.0
        }
        
        response = make_request(api_session, "POST", "/milestones/calculate", json=payload)
        assert response.status_code in [400, 422], "Should return error for invalid year"
    
    def test_calculate_milestones_invalid_reduction_rate(self, api_session):
        """Test milestone calculation with invalid reduction rate"""
        payload = {
            "target_year": 2050,
            "reduction_2030": 150.0,  # Over 100%
            "reduction_2050": 80.0
        }
        
        response = make_request(api_session, "POST", "/milestones/calculate", json=payload)
        # Should either handle gracefully or return error
        assert response.status_code in [200, 400, 422]
    
    def test_get_milestone_visualization(self, api_session):
        """Test GET /milestones/visualization/{scenario_id}"""
        # First create a scenario
        payload = {
            "target_year": 2050,
            "reduction_2030": 30.0,
            "reduction_2050": 80.0
        }
        create_response = make_request(api_session, "POST", "/milestones/calculate", json=payload)
        scenarios = create_response.json()["data"]["scenarios"]
        scenario_id = scenarios[0]["scenario_id"]
        
        # Get visualization
        response = make_request(api_session, "GET", f"/milestones/visualization/{scenario_id}")
        assert_response_success(response)
        
        data = response.json()["data"]
        assert "scenario_data" in data
        assert "chart_config" in data
    
    def test_register_milestone(self, api_session):
        """Test POST /milestones/register"""
        # First create a scenario
        create_payload = {
            "target_year": 2050,
            "reduction_2030": 30.0,
            "reduction_2050": 80.0
        }
        create_response = make_request(api_session, "POST", "/milestones/calculate", json=create_payload)
        scenario_id = create_response.json()["data"]["scenarios"][0]["scenario_id"]
        
        # Register the scenario
        register_payload = {
            "selected_scenario_id": scenario_id,
            "registration_date": "2025-09-30T00:00:00Z",
            "approval_status": "approved",
            "notes": "Test registration"
        }
        
        response = make_request(api_session, "POST", "/milestones/register", json=register_payload)
        assert_response_success(response)
        
        data = response.json()["data"]
        assert data["scenario_id"] == scenario_id
        assert "registration_id" in data

# ================================================================================
# MODULE 2: TARGET DIVISION API TESTS
# ================================================================================

# Continuing test_api.py from where it was cut off...

# ================================================================================
# MODULE 2: TARGET DIVISION API TESTS (continued)
# ================================================================================

class TestTargetDivisionAPIs:
    """Test Target Division APIs"""
    
    def test_calculate_target_division(self, api_session):
        """Test POST /targets/calculate-division"""
        # First create a milestone scenario
        milestone_payload = {
            "target_year": 2050,
            "reduction_2030": 30.0,
            "reduction_2050": 80.0
        }
        milestone_response = make_request(api_session, "POST", "/milestones/calculate", json=milestone_payload)
        milestone_id = milestone_response.json()["data"]["scenarios"][0]["scenario_id"]
        
        # Calculate target division
        payload = {
            "selected_milestone_id": milestone_id,
            "allocation_method": "carbon_intensity_weighted",
            "target_years": [2025, 2026, 2027, 2028, 2029, 2030]
        }
        
        response = make_request(api_session, "POST", "/targets/calculate-division", json=payload)
        assert_response_success(response)
        
        data = response.json()["data"]
        assert "division_id" in data
        assert "property_targets" in data
        assert len(data["property_targets"]) > 0
    
    def test_get_target_division_visualization(self, api_session):
        """Test GET /targets/visualization/{division_id}"""
        # First calculate division
        milestone_payload = {"target_year": 2050, "reduction_2030": 30.0, "reduction_2050": 80.0}
        milestone_response = make_request(api_session, "POST", "/milestones/calculate", json=milestone_payload)
        milestone_id = milestone_response.json()["data"]["scenarios"][0]["scenario_id"]
        
        division_payload = {
            "selected_milestone_id": milestone_id,
            "allocation_method": "carbon_intensity_weighted",
            "target_years": [2025, 2030]
        }
        division_response = make_request(api_session, "POST", "/targets/calculate-division", json=division_payload)
        division_id = division_response.json()["data"]["division_id"]
        
        # Get visualization
        response = make_request(api_session, "GET", f"/targets/visualization/{division_id}")
        assert_response_success(response)
        
        data = response.json()["data"]
        assert "chart_type" in data
        assert "data_series" in data
    
    def test_get_planned_targets(self, api_session):
        """Test GET /targets/planned-targets/{property_id}"""
        property_id = "BP01"
        
        response = make_request(api_session, "GET", f"/targets/planned-targets/{property_id}")
        assert_response_success(response)
        
        data = response.json()["data"]
        assert data["property_id"] == property_id
        assert "yearly_targets" in data
    
    def test_register_target_division(self, api_session):
        """Test POST /targets/register"""
        # First create division
        milestone_payload = {"target_year": 2050, "reduction_2030": 30.0, "reduction_2050": 80.0}
        milestone_response = make_request(api_session, "POST", "/milestones/calculate", json=milestone_payload)
        milestone_id = milestone_response.json()["data"]["scenarios"][0]["scenario_id"]
        
        division_payload = {
            "selected_milestone_id": milestone_id,
            "allocation_method": "carbon_intensity_weighted",
            "target_years": [2025, 2030]
        }
        division_response = make_request(api_session, "POST", "/targets/calculate-division", json=division_payload)
        division_id = division_response.json()["data"]["division_id"]
        
        # Register
        register_payload = {
            "selected_scenario_id": division_id,
            "registration_date": "2025-09-30T00:00:00Z",
            "approval_status": "approved",
            "notes": "Test registration"
        }
        
        response = make_request(api_session, "POST", "/targets/register", json=register_payload)
        assert_response_success(response)

# ================================================================================
# MODULE 3: LONG-TERM PLANNING API TESTS
# ================================================================================

class TestLongTermPlanningAPIs:
    """Test Long-term Planning APIs"""
    
    def test_calculate_long_term_plan(self, api_session):
        """Test POST /planning/long-term/calculate"""
        payload = {
            "property_id": "BP01",
            "target_data": {
                "baseline_emission": 150000,
                "target_reduction_2030": 45.0,
                "target_reduction_2050": 80.0
            },
            "strategic_pattern": "Active Installation of RE",
            "planning_horizon": "2025-2050"
        }
        
        response = make_request(api_session, "POST", "/planning/long-term/calculate", json=payload)
        assert_response_success(response)
        
        data = response.json()["data"]
        assert "plan_id" in data
        assert data["property_id"] == "BP01"
        assert "yearly_emission_targets" in data
        assert "fuel_type_breakdown" in data
    
    def test_get_long_term_plan_visualization(self, api_session):
        """Test GET /planning/long-term/visualization/{plan_id}"""
        # First create plan
        create_payload = {
            "property_id": "BP01",
            "target_data": {"baseline_emission": 150000, "target_reduction_2030": 45.0, "target_reduction_2050": 80.0},
            "strategic_pattern": "Active Installation of RE",
            "planning_horizon": "2025-2050"
        }
        create_response = make_request(api_session, "POST", "/planning/long-term/calculate", json=create_payload)
        plan_id = create_response.json()["data"]["plan_id"]
        
        # Get visualization
        response = make_request(api_session, "GET", f"/planning/long-term/visualization/{plan_id}")
        assert_response_success(response)
        
        data = response.json()["data"]
        assert "visualization_config" in data
        assert "kpi_dashboard" in data
    
    def test_register_long_term_plan(self, api_session):
        """Test POST /planning/long-term/register"""
        # First create plan
        create_payload = {
            "property_id": "BP01",
            "target_data": {"baseline_emission": 150000, "target_reduction_2030": 45.0, "target_reduction_2050": 80.0},
            "strategic_pattern": "Active Installation of RE",
            "planning_horizon": "2025-2050"
        }
        create_response = make_request(api_session, "POST", "/planning/long-term/calculate", json=create_payload)
        plan_id = create_response.json()["data"]["plan_id"]
        
        # Register
        register_payload = {
            "selected_scenario_id": plan_id,
            "registration_date": "2025-09-30T00:00:00Z",
            "approval_status": "approved",
            "notes": "Test plan registration"
        }
        
        response = make_request(api_session, "POST", "/planning/long-term/register", json=register_payload)
        assert_response_success(response)

# ================================================================================
# MODULE 4: REOPTIMIZATION API TESTS
# ================================================================================

class TestReoptimizationAPIs:
    """Test Reoptimization APIs"""
    
    def test_calculate_annual_reoptimization(self, api_session):
        """Test POST /planning/annual/reoptimize"""
        payload = {
            "property_id": "BP01",
            "deviation_threshold": 5.0,
            "actual_performance_data": {
                "ytd_emissions": [12500, 11800, 13200, 12000],
                "ytd_costs": [18000, 17500, 19200, 18500],
                "period": "Q1-Q4 2026"
            },
            "analysis_period": "YTD-2026"
        }
        
        response = make_request(api_session, "POST", "/planning/annual/reoptimize", json=payload)
        assert_response_success(response)
        
        data = response.json()["data"]
        assert "reoptimization_id" in data
        assert data["property_id"] == "BP01"
        assert "reoptimization_required" in data
        assert "recommendations" in data
    
    def test_get_annual_plan_visualization(self, api_session):
        """Test GET /planning/annual/visualization/{property_id}"""
        property_id = "BP01"
        
        response = make_request(api_session, "GET", f"/planning/annual/visualization/{property_id}")
        assert_response_success(response)
        
        data = response.json()["data"]
        assert data["property_id"] == property_id
        assert "chart_configurations" in data
    
    def test_register_reoptimized_plan(self, api_session):
        """Test POST /planning/annual/register"""
        # First create reoptimization
        create_payload = {
            "property_id": "BP01",
            "deviation_threshold": 5.0,
            "actual_performance_data": {
                "ytd_emissions": [12500, 11800, 13200, 12000],
                "ytd_costs": [18000, 17500, 19200, 18500],
                "period": "Q1-Q4 2026"
            },
            "analysis_period": "YTD-2026"
        }
        create_response = make_request(api_session, "POST", "/planning/annual/reoptimize", json=create_payload)
        reopt_id = create_response.json()["data"]["reoptimization_id"]
        
        # Register
        register_payload = {
            "selected_scenario_id": reopt_id,
            "registration_date": "2025-09-30T00:00:00Z",
            "approval_status": "approved",
            "notes": "Test reoptimization registration"
        }
        
        response = make_request(api_session, "POST", "/planning/annual/register", json=register_payload)
        assert_response_success(response)

# ================================================================================
# PRICING DATA API TESTS
# ================================================================================

class TestPricingAPIs:
    """Test Pricing Data APIs"""
    
    def test_get_carbon_credit_prices(self, api_session):
        """Test GET /api/carbon-credit-prices/{year}"""
        year = 2024
        
        response = make_request(api_session, "GET", f"/carbon-credit-prices/{year}")
        assert_response_success(response)
        
        data = response.json()["data"]
        assert data["year"] == year
        assert "price_data" in data
        assert len(data["price_data"]) > 0
    
    def test_get_renewable_energy_prices(self, api_session):
        """Test GET /api/renewable-energy-prices/{year}"""
        year = 2024
        
        response = make_request(api_session, "GET", f"/renewable-energy-prices/{year}")
        assert_response_success(response)
        
        data = response.json()["data"]
        assert data["year"] == year
        assert "price_data" in data
    
    def test_get_renewable_fuel_prices(self, api_session):
        """Test GET /api/renewable-fuel-prices/{year}"""
        year = 2024
        
        response = make_request(api_session, "GET", f"/renewable-fuel-prices/{year}")
        assert_response_success(response)
        
        data = response.json()["data"]
        assert data["year"] == year
        assert "price_data" in data

# ================================================================================
# PERFORMANCE AND LOAD TESTS
# ================================================================================

class TestAPIPerformance:
    """Test API performance characteristics"""
    
    def test_milestone_calculation_performance(self, api_session):
        """Test that milestone calculation completes in reasonable time"""
        payload = {
            "target_year": 2050,
            "reduction_2030": 30.0,
            "reduction_2050": 80.0
        }
        
        start_time = time.time()
        response = make_request(api_session, "POST", "/milestones/calculate", json=payload)
        elapsed_time = time.time() - start_time
        
        assert_response_success(response)
        assert elapsed_time < 5.0, f"Milestone calculation took {elapsed_time:.2f}s (threshold: 5s)"
    
    def test_concurrent_requests(self, api_session):
        """Test API handling of concurrent requests"""
        import concurrent.futures
        
        def make_test_request():
            payload = {"target_year": 2050, "reduction_2030": 30.0, "reduction_2050": 80.0}
            return make_request(api_session, "POST", "/milestones/calculate", json=payload)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_test_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in results:
            assert response.status_code == 200

# ================================================================================
# ERROR HANDLING TESTS
# ================================================================================

class TestAPIErrorHandling:
    """Test API error handling"""
    
    def test_missing_required_fields(self, api_session):
        """Test API response to missing required fields"""
        payload = {
            "target_year": 2050
            # Missing reduction_2030 and reduction_2050
        }
        
        response = make_request(api_session, "POST", "/milestones/calculate", json=payload)
        assert response.status_code in [400, 422], "Should return error for missing fields"
    
    def test_invalid_json(self, api_session):
        """Test API response to invalid JSON"""
        response = requests.post(
            f"{API_BASE_URL}/milestones/calculate",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code in [400, 422], "Should return error for invalid JSON"
    
    def test_nonexistent_endpoint(self, api_session):
        """Test API response to nonexistent endpoint"""
        response = requests.get(
            f"{API_BASE_URL}/nonexistent/endpoint",
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 404, "Should return 404 for nonexistent endpoint"
    
    def test_method_not_allowed(self, api_session):
        """Test API response to wrong HTTP method"""
        # Try GET on POST-only endpoint
        response = requests.get(
            f"{API_BASE_URL}/milestones/calculate",
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 405, "Should return 405 for method not allowed"

# ================================================================================
# DATA VALIDATION TESTS
# ================================================================================

class TestAPIDataValidation:
    """Test API data validation"""
    
    def test_validate_year_range(self, api_session):
        """Test year validation"""
        # Test future year beyond reasonable range
        payload = {
            "target_year": 2100,
            "reduction_2030": 30.0,
            "reduction_2050": 80.0
        }
        
        response = make_request(api_session, "POST", "/milestones/calculate", json=payload)
        # Should either accept with warning or reject
        assert response.status_code in [200, 400, 422]
    
    def test_validate_percentage_range(self, api_session):
        """Test percentage validation"""
        payload = {
            "target_year": 2050,
            "reduction_2030": -10.0,  # Negative percentage
            "reduction_2050": 80.0
        }
        
        response = make_request(api_session, "POST", "/milestones/calculate", json=payload)
        assert response.status_code in [400, 422], "Should reject negative percentages"
    
    def test_validate_property_id(self, api_session):
        """Test property ID validation"""
        payload = {
            "property_id": "INVALID_ID",
            "target_data": {"baseline_emission": 150000, "target_reduction_2030": 45.0, "target_reduction_2050": 80.0},
            "strategic_pattern": "Active Installation of RE",
            "planning_horizon": "2025-2050"
        }
        
        response = make_request(api_session, "POST", "/planning/long-term/calculate", json=payload)
        # Should return error for invalid property ID
        assert response.status_code in [400, 404, 422]

# ================================================================================
# RESPONSE FORMAT TESTS
# ================================================================================

class TestAPIResponseFormat:
    """Test API response format consistency"""
    
    def test_success_response_format(self, api_session):
        """Test that success responses follow standard format"""
        payload = {
            "target_year": 2050,
            "reduction_2030": 30.0,
            "reduction_2050": 80.0
        }
        
        response = make_request(api_session, "POST", "/milestones/calculate", json=payload)
        data = response.json()
        
        # Check standard fields
        assert "status" in data
        assert "message" in data
        assert "data" in data
        
        # Check data has execution metadata
        assert "execution_time_ms" in data or "timestamp" in data.get("data", {})
    
    def test_error_response_format(self, api_session):
        """Test that error responses follow standard format"""
        payload = {
            "target_year": 2050
            # Missing required fields
        }
        
        response = make_request(api_session, "POST", "/milestones/calculate", json=payload)
        
        if response.status_code >= 400:
            data = response.json()
            assert "status" in data
            assert "message" in data or "error" in data

# ================================================================================
# RUN TESTS
# ================================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])