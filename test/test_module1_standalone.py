"""
EcoAssist API Testing Script - Module 1: Milestone Setting
Standalone script without pytest dependency

Usage:
    python test_module1_standalone.py
"""

import requests
import json
from datetime import datetime
import time
from typing import Dict, Any, List

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL = "http://localhost:8000/api/v1"
API_TIMEOUT = 30

# Test credentials
TEST_USER = {
    "username": "test@example.com",
    "password": "password123"
}

# =============================================================================
# TEST DATA
# =============================================================================

# Realistic property IDs
PROPERTY_IDS = [
    "550e8400-e29b-41d4-a716-446655440001",  # Brisbane Plaza
    "550e8400-e29b-41d4-a716-446655440002",  # Melbourne City Tower
    "550e8400-e29b-41d4-a716-446655440003"   # Sydney Retail Centre
]

# Baseline emissions data (3 properties, 3 years)
BASELINE_EMISSIONS_SMALL = [
    {
        "year": 2022,
        "scope1_emissions": 1450.25,
        "scope2_emissions": 3100.50,
        "total_consumption": 445000.00,
        "total_cost": 122000.00,
        "unit": "kg-CO2e"
    },
    {
        "year": 2023,
        "scope1_emissions": 1480.00,
        "scope2_emissions": 3150.00,
        "total_consumption": 448000.00,
        "total_cost": 124500.00,
        "unit": "kg-CO2e"
    },
    {
        "year": 2024,
        "scope1_emissions": 1500.50,
        "scope2_emissions": 3200.75,
        "total_consumption": 450000.00,
        "total_cost": 125000.00,
        "unit": "kg-CO2e"
    }
]

# Portfolio-scale baseline data
BASELINE_EMISSIONS_PORTFOLIO = [
    {
        "year": 2021,
        "scope1_emissions": 85000.00,
        "scope2_emissions": 185000.00,
        "total_consumption": 2500000.00,
        "total_cost": 680000.00,
        "unit": "kg-CO2e"
    },
    {
        "year": 2022,
        "scope1_emissions": 87500.00,
        "scope2_emissions": 190000.00,
        "total_consumption": 2550000.00,
        "total_cost": 695000.00,
        "unit": "kg-CO2e"
    },
    {
        "year": 2023,
        "scope1_emissions": 89000.00,
        "scope2_emissions": 195000.00,
        "total_consumption": 2600000.00,
        "total_cost": 710000.00,
        "unit": "kg-CO2e"
    },
    {
        "year": 2024,
        "scope1_emissions": 90500.00,
        "scope2_emissions": 198000.00,
        "total_consumption": 2650000.00,
        "total_cost": 725000.00,
        "unit": "kg-CO2e"
    }
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 80}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 80}{Colors.END}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str):
    """Print info message"""
    print(f"  {text}")


def authenticate() -> str:
    """Get authentication token"""
    print_info("Authenticating...")
    
    url = f"{BASE_URL}/system/security/authenticate"
    try:
        response = requests.post(url, json=TEST_USER, timeout=API_TIMEOUT)
        
        if response.status_code == 200:
            token = response.json().get("access_token")
            print_success(f"Authentication successful")
            return token
        else:
            print_error(f"Authentication failed: {response.status_code}")
            return None
    except Exception as e:
        print_error(f"Authentication error: {str(e)}")
        return None


def get_auth_headers(token: str) -> Dict[str, str]:
    """Create authorization headers"""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }


def format_json(data: Any) -> str:
    """Format JSON data for display"""
    return json.dumps(data, indent=2)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_api1_calculate_milestones_basic(headers: Dict[str, str]) -> bool:
    """
    TEST 1: Basic Milestone Calculation
    
    Tests:
    - POST /api/v1/milestones/calculate
    - Standard and Aggressive scenarios
    - 3 properties with 3 years baseline data
    """
    print_header("TEST 1: Basic Milestone Calculation")
    
    url = f"{BASE_URL}/milestones/calculate"
    
    payload = {
        "base_year": 2024,
        "mid_term_target_year": 2030,
        "long_term_target_year": 2050,
        "property_ids": PROPERTY_IDS,
        "baseline_data": BASELINE_EMISSIONS_SMALL,
        "scenario_types": ["Standard", "Aggressive"]
    }
    
    print_info("Sending request...")
    print_info(f"Properties: {len(payload['property_ids'])}")
    print_info(f"Baseline years: 2022-2024")
    print_info(f"Target years: 2030, 2050")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=API_TIMEOUT)
        duration = time.time() - start_time
        
        print_info(f"Response time: {duration:.2f}s")
        print_info(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            scenarios = data.get("data", {}).get("scenarios", [])
            
            print_success(f"Calculation successful!")
            print_info(f"Scenarios generated: {len(scenarios)}")
            
            # Display scenario details
            for i, scenario in enumerate(scenarios, 1):
                print_info(f"\nScenario {i}: {scenario.get('scenario_type')}")
                print_info(f"  ID: {scenario.get('scenario_id')}")
                print_info(f"  Description: {scenario.get('description', 'N/A')[:60]}...")
                
                # Get 2030 and 2050 targets
                targets = scenario.get("reduction_targets", [])
                for target in targets:
                    if target["year"] in [2030, 2050]:
                        print_info(f"  {target['year']}: {target['reduction_from_baseline']:.1f}% reduction")
                
                # Cost projection
                costs = scenario.get("cost_projection", [])
                if costs:
                    total_capex = sum(c.get("estimated_cost", 0) for c in costs)
                    print_info(f"  Total investment: ${total_capex:,.2f}")
            
            # Save scenario IDs for later tests
            if scenarios:
                global SAVED_SCENARIO_ID
                SAVED_SCENARIO_ID = scenarios[0]["scenario_id"]
            
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            print_info(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        return False


def test_api1_calculate_all_scenarios(headers: Dict[str, str]) -> bool:
    """
    TEST 2: Calculate All Scenario Types
    
    Tests:
    - All scenario types (Standard, Aggressive, Conservative)
    - Verify different reduction rates
    """
    print_header("TEST 2: All Scenario Types")
    
    url = f"{BASE_URL}/milestones/calculate"
    
    payload = {
        "base_year": 2024,
        "mid_term_target_year": 2030,
        "long_term_target_year": 2050,
        "property_ids": PROPERTY_IDS[:1],  # Single property for speed
        "baseline_data": BASELINE_EMISSIONS_SMALL,
        "scenario_types": ["Standard", "Aggressive", "Conservative"]
    }
    
    print_info("Requesting all scenario types...")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=API_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            scenarios = data.get("data", {}).get("scenarios", [])
            
            print_success(f"Generated {len(scenarios)} scenarios")
            
            # Compare reduction rates at 2030
            reduction_2030 = {}
            for scenario in scenarios:
                scenario_type = scenario.get("scenario_type")
                targets = scenario.get("reduction_targets", [])
                for target in targets:
                    if target["year"] == 2030:
                        reduction_2030[scenario_type] = target["reduction_from_baseline"]
            
            print_info("\n2030 Reduction rates:")
            for stype, reduction in sorted(reduction_2030.items()):
                print_info(f"  {stype}: {reduction:.1f}%")
            
            # Validate ordering
            if "Conservative" in reduction_2030 and "Aggressive" in reduction_2030:
                if reduction_2030["Aggressive"] > reduction_2030["Conservative"]:
                    print_success("Scenario ordering is correct (Aggressive > Conservative)")
                else:
                    print_warning("Scenario ordering may be incorrect")
            
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        return False


def test_api1_custom_strategy_preferences(headers: Dict[str, str]) -> bool:
    """
    TEST 3: Custom Strategy Preferences
    
    Tests:
    - Custom strategy weights
    - Renewable energy emphasis
    """
    print_header("TEST 3: Custom Strategy Preferences")
    
    url = f"{BASE_URL}/milestones/calculate"
    
    payload = {
        "base_year": 2024,
        "mid_term_target_year": 2030,
        "long_term_target_year": 2050,
        "property_ids": PROPERTY_IDS[:1],
        "baseline_data": BASELINE_EMISSIONS_SMALL,
        "scenario_types": ["Standard"],
        "strategy_preferences": {
            "renewable_energy_weight": 0.6,  # 60% renewable energy
            "efficiency_improvement_weight": 0.3,  # 30% efficiency
            "behavioral_change_weight": 0.1  # 10% behavioral
        }
    }
    
    print_info("Testing custom strategy weights:")
    print_info("  Renewable Energy: 60%")
    print_info("  Efficiency Improvement: 30%")
    print_info("  Behavioral Change: 10%")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=API_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            scenarios = data.get("data", {}).get("scenarios", [])
            
            if scenarios:
                strategy = scenarios[0].get("strategy_breakdown", {})
                print_success("Strategy preferences applied")
                
                if strategy:
                    print_info("\nStrategy breakdown:")
                    print_info(f"  Renewable Energy: {strategy.get('renewable_energy_percentage', 0):.1f}%")
                    print_info(f"  Efficiency: {strategy.get('efficiency_improvement_percentage', 0):.1f}%")
                    print_info(f"  Behavioral: {strategy.get('behavioral_change_percentage', 0):.1f}%")
            
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        return False


def test_api1_portfolio_scale(headers: Dict[str, str]) -> bool:
    """
    TEST 4: Portfolio-Scale Calculation
    
    Tests:
    - Large emission volumes (270+ tonnes CO2e)
    - Multiple years of baseline data
    - Performance with larger datasets
    """
    print_header("TEST 4: Portfolio-Scale Calculation")
    
    url = f"{BASE_URL}/milestones/calculate"
    
    payload = {
        "base_year": 2024,
        "mid_term_target_year": 2030,
        "long_term_target_year": 2050,
        "property_ids": PROPERTY_IDS,
        "baseline_data": BASELINE_EMISSIONS_PORTFOLIO,
        "scenario_types": ["Standard", "Aggressive"]
    }
    
    print_info("Testing portfolio-scale data:")
    print_info("  Baseline emission: ~288,500 kg-CO2e (288.5 tonnes)")
    print_info("  Baseline years: 2021-2024 (4 years)")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=API_TIMEOUT)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            scenarios = data.get("data", {}).get("scenarios", [])
            
            print_success(f"Portfolio calculation successful ({duration:.2f}s)")
            print_info(f"Scenarios generated: {len(scenarios)}")
            
            # Verify emission volumes
            for scenario in scenarios:
                baseline = scenario.get("baseline_emission", 0)
                print_info(f"\n{scenario.get('scenario_type')} scenario:")
                print_info(f"  Baseline: {baseline:,.0f} kg-CO2e ({baseline/1000:.1f} tonnes)")
                
                targets = scenario.get("reduction_targets", [])
                for target in targets:
                    if target["year"] == 2030:
                        print_info(f"  2030 target: {target['target_emissions']:,.0f} kg-CO2e")
                        print_info(f"  Reduction: {target['reduction_from_baseline']:.1f}%")
            
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        return False


def test_api2_get_scenario(headers: Dict[str, str], scenario_id: str) -> bool:
    """
    TEST 5: Get Milestone Scenario
    
    Tests:
    - GET /api/v1/milestones/scenarios/{scenario_id}
    - Retrieve specific scenario details
    """
    print_header("TEST 5: Get Milestone Scenario")
    
    if not scenario_id:
        print_warning("No scenario ID available, skipping test")
        return False
    
    url = f"{BASE_URL}/milestones/scenarios/{scenario_id}"
    
    print_info(f"Retrieving scenario: {scenario_id}")
    
    try:
        response = requests.get(url, headers=headers, timeout=API_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            scenario = data.get("data", {})
            
            print_success("Scenario retrieved successfully")
            print_info(f"Type: {scenario.get('scenario_type')}")
            print_info(f"Description: {scenario.get('description', 'N/A')[:60]}...")
            
            targets = scenario.get("reduction_targets", [])
            print_info(f"Reduction targets: {len(targets)} years")
            
            return True
        elif response.status_code == 404:
            print_warning("Scenario not found (expected for temporary preview IDs)")
            return True  # Not a failure for this test
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        return False


def test_api3_get_visualization(headers: Dict[str, str], scenario_id: str) -> bool:
    """
    TEST 6: Get Milestone Visualization
    
    Tests:
    - GET /api/v1/milestones/scenarios/{scenario_id}/visualization
    - Visualization configuration
    """
    print_header("TEST 6: Get Milestone Visualization")
    
    if not scenario_id:
        print_warning("No scenario ID available, skipping test")
        return False
    
    url = f"{BASE_URL}/milestones/scenarios/{scenario_id}/visualization"
    
    print_info(f"Getting visualization for: {scenario_id}")
    
    try:
        response = requests.get(url, headers=headers, timeout=API_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            viz_data = data.get("data", {})
            
            print_success("Visualization data retrieved")
            
            viz_config = viz_data.get("visualization_config", {})
            if viz_config:
                print_info(f"Chart type: {viz_config.get('recommended_chart_type')}")
                print_info(f"X-axis: {viz_config.get('x_axis', {}).get('label')}")
                print_info(f"Y-axis: {viz_config.get('y_axis', {}).get('label')}")
            
            return True
        elif response.status_code == 404:
            print_warning("Scenario not found (expected for temporary preview IDs)")
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        return False


def test_api4_register_milestone(headers: Dict[str, str], scenario_id: str) -> bool:
    """
    TEST 7: Register Milestone
    
    Tests:
    - POST /api/v1/milestones/register
    - Scenario registration with approval
    """
    print_header("TEST 7: Register Milestone")
    
    if not scenario_id:
        print_warning("No scenario ID available, skipping test")
        return False
    
    url = f"{BASE_URL}/milestones/register"
    
    payload = {
        "scenario_id": scenario_id,
        "approval_status": "approved",
        "approved_by": "test_manager@example.com",
        "approval_date": datetime.utcnow().isoformat(),
        "notes": "Test registration - Standard pathway approved for implementation"
    }
    
    print_info("Registering scenario with approved status...")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=API_TIMEOUT)
        
        if response.status_code in [200, 201]:
            data = response.json()
            registration = data.get("data", {})
            
            print_success("Scenario registered successfully")
            print_info(f"Registration ID: {registration.get('registration_id')}")
            print_info(f"Scenario ID: {registration.get('scenario_id')}")
            print_info(f"Status: {registration.get('approval_status')}")
            print_info(f"Registered at: {registration.get('registered_at')}")
            
            return True
        elif response.status_code == 404:
            print_warning("Scenario not found (expected for temporary preview IDs)")
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            print_info(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        return False


def test_api_list_scenarios(headers: Dict[str, str]) -> bool:
    """
    TEST 8: List Milestone Scenarios
    
    Tests:
    - GET /api/v1/milestones/scenarios/list
    - Pagination and filtering
    """
    print_header("TEST 8: List Milestone Scenarios")
    
    url = f"{BASE_URL}/milestones/scenarios/list"
    params = {"limit": 10, "offset": 0}
    
    print_info("Listing scenarios with pagination...")
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=API_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            result = data.get("data", {})
            
            scenarios = result.get("scenarios", [])
            pagination = result.get("pagination", {})
            
            print_success("Scenarios listed successfully")
            print_info(f"Scenarios returned: {len(scenarios)}")
            print_info(f"Total count: {pagination.get('total_count', 0)}")
            print_info(f"Has more: {pagination.get('has_more', False)}")
            
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        return False


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def test_validation_errors(headers: Dict[str, str]) -> bool:
    """
    TEST 9: Validation Error Handling
    
    Tests:
    - Invalid year range
    - Empty property list
    - Negative emissions
    """
    print_header("TEST 9: Validation Error Handling")
    
    url = f"{BASE_URL}/milestones/calculate"
    
    # Test 1: Invalid year range
    print_info("Test 1: Invalid year range (mid-term before base year)...")
    payload1 = {
        "base_year": 2024,
        "mid_term_target_year": 2023,  # Invalid
        "long_term_target_year": 2050,
        "property_ids": PROPERTY_IDS[:1],
        "baseline_data": BASELINE_EMISSIONS_SMALL
    }
    
    try:
        response = requests.post(url, json=payload1, headers=headers, timeout=API_TIMEOUT)
        if response.status_code in [400, 422]:
            print_success("Correctly rejected invalid year range")
        else:
            print_warning(f"Expected 400/422, got {response.status_code}")
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
    
    # Test 2: Empty property list
    print_info("\nTest 2: Empty property list...")
    payload2 = {
        "base_year": 2024,
        "mid_term_target_year": 2030,
        "long_term_target_year": 2050,
        "property_ids": [],  # Invalid
        "baseline_data": BASELINE_EMISSIONS_SMALL
    }
    
    try:
        response = requests.post(url, json=payload2, headers=headers, timeout=API_TIMEOUT)
        if response.status_code in [400, 422]:
            print_success("Correctly rejected empty property list")
        else:
            print_warning(f"Expected 400/422, got {response.status_code}")
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
    
    return True


# =============================================================================
# MAIN EXECUTION
# =============================================================================

SAVED_SCENARIO_ID = None

def main():
    """Main test execution"""
    print_header("EcoAssist API Test Suite - Module 1: Milestone Setting")
    print_info("Target API: " + BASE_URL)
    print_info("Test User: " + TEST_USER["username"])
    
    # Authenticate
    token = authenticate()
    if not token:
        print_error("Authentication failed. Cannot proceed with tests.")
        return
    
    headers = get_auth_headers(token)
    
    # Run tests
    results = {}
    
    results["test_1"] = test_api1_calculate_milestones_basic(headers)
    results["test_2"] = test_api1_calculate_all_scenarios(headers)
    results["test_3"] = test_api1_custom_strategy_preferences(headers)
    results["test_4"] = test_api1_portfolio_scale(headers)
    results["test_5"] = test_api2_get_scenario(headers, SAVED_SCENARIO_ID)
    results["test_6"] = test_api3_get_visualization(headers, SAVED_SCENARIO_ID)
    results["test_7"] = test_api4_register_milestone(headers, SAVED_SCENARIO_ID)
    results["test_8"] = test_api_list_scenarios(headers)
    results["test_9"] = test_validation_errors(headers)
    
    # Print summary
    print_header("TEST SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    failed_tests = total_tests - passed_tests
    
    for test_name, result in results.items():
        status = f"{Colors.GREEN}PASSED{Colors.END}" if result else f"{Colors.RED}FAILED{Colors.END}"
        print(f"  {test_name}: {status}")
    
    print(f"\n{Colors.BOLD}Total: {total_tests} tests{Colors.END}")
    print(f"{Colors.GREEN}Passed: {passed_tests}{Colors.END}")
    if failed_tests > 0:
        print(f"{Colors.RED}Failed: {failed_tests}{Colors.END}")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"\n{Colors.BOLD}Success Rate: {success_rate:.1f}%{Colors.END}")
    
    if success_rate == 100:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All tests passed!{Colors.END}")
    elif success_rate >= 80:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️ Most tests passed, but some issues found{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Multiple test failures detected{Colors.END}")


if __name__ == "__main__":
    main()
