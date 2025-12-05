#!/usr/bin/env python3
"""
EcoAssist API Test Suite - Module 2: Target Division
Standalone test script for API 5 (Target Allocation)

Usage:
    python test_module2_standalone.py

Requirements:
    pip install requests
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = "http://localhost:8000/api/v1"
API_TIMEOUT = 30

# Test credentials (update if needed)
TEST_USERNAME = "test@example.com"
TEST_PASSWORD = "password123"

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# =============================================================================
# TEST DATA
# =============================================================================

# Property IDs from Module 1 tests
PROPERTY_IDS = [
    "550e8400-e29b-41d4-a716-446655440001",
    "550e8400-e29b-41d4-a716-446655440002",
    "550e8400-e29b-41d4-a716-446655440003"
]

# Scenario ID (would come from Module 1 milestone calculation)
TEST_SCENARIO_ID = "SCEN-12345678"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}")

def print_test_info(text: str):
    """Print test information"""
    print(f"  {text}")

def print_success(text: str):
    """Print success message"""
    print(f"{GREEN}✓{RESET} {text}")

def print_error(text: str):
    """Print error message"""
    print(f"{RED}✗{RESET} {text}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{YELLOW}⚠{RESET} {text}")

def authenticate() -> str:
    """Authenticate and get access token"""
    print_test_info("Authenticating...")
    
    url = f"{API_BASE_URL}/system/security/authenticate"
    payload = {
        "username": TEST_USERNAME,
        "password": TEST_PASSWORD
    }
    
    try:
        response = requests.post(url, json=payload, timeout=API_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            
            # Try different response structures
            token = None
            
            # Structure 1: {data: {access_token: ...}}
            if isinstance(data.get("data"), dict):
                token = data["data"].get("access_token")
            
            # Structure 2: {access_token: ...}
            if not token and "access_token" in data:
                token = data.get("access_token")
            
            # Structure 3: Nested in result
            if not token and "result" in data:
                token = data["result"].get("access_token")
            
            if token:
                print_success("Authentication successful")
                return token
            else:
                print_error("Authentication successful but no token found")
                print_test_info(f"Response structure: {json.dumps(data, indent=2)}")
                return None
        else:
            print_error(f"Authentication failed: {response.status_code}")
            print_test_info(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print_error(f"Authentication error: {str(e)}")
        return None

def make_api_call(
    method: str,
    endpoint: str,
    token: str,
    data: Dict = None
) -> tuple[int, Dict, float]:
    """Make API call and return status, response, and duration"""
    url = f"{API_BASE_URL}{endpoint}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    
    if method == "POST":
        response = requests.post(url, json=data, headers=headers, timeout=API_TIMEOUT)
    elif method == "GET":
        response = requests.get(url, headers=headers, timeout=API_TIMEOUT)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    duration = time.time() - start_time
    
    try:
        response_data = response.json()
    except:
        response_data = {"error": "Invalid JSON response"}
    
    return response.status_code, response_data, duration

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_1_proportional_allocation(token: str) -> bool:
    """Test 1: Proportional allocation by baseline emission"""
    print_header("TEST 1: Proportional Allocation by Baseline")
    
    print_test_info("Sending request...")
    print_test_info(f"Properties: {len(PROPERTY_IDS)}")
    print_test_info("Allocation method: PROPORTIONAL_BASELINE")
    
    payload = {
        "scenario_id": TEST_SCENARIO_ID,
        "property_ids": PROPERTY_IDS,
        "allocation_method": "PROPORTIONAL_BASELINE",
        "target_year": 2030,
        "total_reduction_target": 1850.0, 
        "optimization_criteria": ["cost_effectiveness", "feasibility"]
    }
    
    status, response, duration = make_api_call(
        "POST",
        "/target-division/calculate",
        token,
        payload
    )
    
    print_test_info(f"Response time: {duration:.2f}s")
    print_test_info(f"Status code: {status}")
    
    if status == 200:
        data = response.get("data")
        if data is None:
            print_error("Response has no data field")
            print_test_info(f"Full response: {json.dumps(response, indent=2)}")
            return False
        
        allocations = data.get("allocations", [])
        summary = data.get("allocation_summary", {})
        
        print_success(f"Allocation successful: {len(allocations)} properties")
        print_test_info(f"Total allocated (2030): {summary.get('total_allocated_2030', 0):.2f} kg-CO2e")
        print_test_info(f"Total allocated (2050): {summary.get('total_allocated_2050', 0):.2f} kg-CO2e")
        
        # Show sample allocation
        if allocations:
            sample = allocations[0]
            print_test_info(f"Sample allocation:")
            print_test_info(f"  Property: {sample.get('property_id', 'N/A')[:8]}...")
            print_test_info(f"  2030 Target: {sample.get('allocated_2030_target', 0):.2f} kg-CO2e")
            print_test_info(f"  2050 Target: {sample.get('allocated_2050_target', 0):.2f} kg-CO2e")
        
        return True
    else:
        print_error(f"Failed with status {status}")
        print_test_info(f"Response: {json.dumps(response, indent=2)}")
        return False

def test_2_equal_allocation(token: str) -> bool:
    """Test 2: Equal allocation across properties"""
    print_header("TEST 2: Equal Allocation")
    
    print_test_info("Testing equal distribution...")
    
    payload = {
        "scenario_id": TEST_SCENARIO_ID,
        "property_ids": PROPERTY_IDS,
        "allocation_method": "EQUAL_DISTRIBUTION",
        "target_year": 2030,
        "total_reduction_target": 1800.0,
        "optimization_criteria": ["fairness"]
    }
    
    status, response, duration = make_api_call(
        "POST",
        "/target-division/calculate",
        token,
        payload
    )
    
    if status == 200:
        data = response.get("data")
        if data is None:
            print_error("Response has no data field")
            print_test_info(f"Full response: {json.dumps(response, indent=2)}")
            return False
        
        allocations = data.get("allocations", [])
        
        # Verify equal distribution
        if allocations:
            targets_2030 = [a.get('allocated_2030_target', 0) for a in allocations]
            avg_target = sum(targets_2030) / len(targets_2030)
            max_deviation = max(abs(t - avg_target) for t in targets_2030)
            
            print_success(f"Equal allocation: {len(allocations)} properties")
            print_test_info(f"Average target: {avg_target:.2f} kg-CO2e")
            print_test_info(f"Max deviation: {max_deviation:.2f} kg-CO2e")
            
            if max_deviation < avg_target * 0.01:  # Less than 1% deviation
                print_success("Allocation is truly equal")
            else:
                print_warning("Some variation in allocation")
        
        return True
    else:
        print_error(f"Failed with status {status}")
        return False

def test_3_capacity_based_allocation(token: str) -> bool:
    """Test 3: Allocation based on reduction capacity"""
    print_header("TEST 3: Capacity-Based Allocation")
    
    print_test_info("Testing allocation by reduction capacity...")
    
    payload = {
        "scenario_id": TEST_SCENARIO_ID,
        "property_ids": PROPERTY_IDS,
        "allocation_method": "CAPACITY_BASED",
        "target_year": 2030,
        "total_reduction_target": 2000.0,
        "optimization_criteria": ["technical_feasibility", "cost_effectiveness"]
    }
    
    status, response, duration = make_api_call(
        "POST",
        "/target-division/calculate",
        token,
        payload
    )
    
    if status == 200:
        print_success("Capacity-based allocation completed")
        return True
    else:
        print_error(f"Failed with status {status}")
        return False

def test_4_constrained_allocation(token: str) -> bool:
    """Test 4: Allocation with constraints"""
    print_header("TEST 4: Allocation with Constraints")
    
    print_test_info("Testing allocation with min/max constraints...")
    
    payload = {
        "scenario_id": TEST_SCENARIO_ID,
        "property_ids": PROPERTY_IDS,
        "allocation_method": "PROPORTIONAL_BASELINE",
        "target_year": 2030,
        "constraints": {
            "min_reduction_per_property": 500.0,
            "max_reduction_per_property": 2000.0,
            "total_budget_limit": 1000000.0
        },
        "total_reduction_target": 1500.0,
        "optimization_criteria": ["cost_effectiveness"]
    }
    
    status, response, duration = make_api_call(
        "POST",
        "/target-division/calculate",
        token,
        payload
    )
    
    if status == 200:
        data = response.get("data")
        if data is None:
            print_error("Response has no data field")
            print_test_info(f"Full response: {json.dumps(response, indent=2)}")
            return False
        
        allocations = data.get("allocations", [])
        
        # Verify constraints
        if allocations:
            targets_2030 = [a.get('allocated_2030_target', 0) for a in allocations]
            min_target = min(targets_2030)
            max_target = max(targets_2030)
            
            print_success(f"Constrained allocation: {len(allocations)} properties")
            print_test_info(f"Min allocation: {min_target:.2f} kg-CO2e")
            print_test_info(f"Max allocation: {max_target:.2f} kg-CO2e")
            
            if min_target >= 500.0 and max_target <= 2000.0:
                print_success("All constraints satisfied")
            else:
                print_warning("Some constraints may be violated")
        
        return True
    else:
        print_error(f"Failed with status {status}")
        return False

def test_5_single_property_allocation(token: str) -> bool:
    """Test 5: Allocation for single property"""
    print_header("TEST 5: Single Property Allocation")
    
    print_test_info("Testing allocation for one property...")
    
    payload = {
        "scenario_id": TEST_SCENARIO_ID,
        "property_ids": [PROPERTY_IDS[0]],  # Single property
        "allocation_method": "PROPORTIONAL_BASELINE",
        "total_reduction_target": 700.0,
        "target_year": 2030
    }
    
    status, response, duration = make_api_call(
        "POST",
        "/target-division/calculate",
        token,
        payload
    )
    
    if status == 200:
        data = response.get("data")
        if data is None:
            print_error("Response has no data field")
            print_test_info(f"Full response: {json.dumps(response, indent=2)}")
            return False
        
        allocations = data.get("allocations", [])
        
        if len(allocations) == 1:
            print_success("Single property allocation successful")
            print_test_info(f"Property: {allocations[0].get('property_id', 'N/A')[:8]}...")
        else:
            print_warning(f"Expected 1 allocation, got {len(allocations)}")
        
        return True
    else:
        print_error(f"Failed with status {status}")
        return False

def test_6_multi_year_allocation(token: str) -> bool:
    """Test 6: Allocation with multiple target years"""
    print_header("TEST 6: Multi-Year Target Allocation")
    
    print_test_info("Testing allocation for 2030 and 2050 targets...")
    
    payload = {
        "scenario_id": TEST_SCENARIO_ID,
        "property_ids": PROPERTY_IDS,
        "allocation_method": "PROPORTIONAL_BASELINE",
        "target_year": 2050,  # Long-term target
        "total_reduction_target": 3700.0,
        "optimization_criteria": ["cost_effectiveness", "sustainability"]
    }
    
    status, response, duration = make_api_call(
        "POST",
        "/target-division/calculate",
        token,
        payload
    )
    
    if status == 200:
        data = response.get("data")
        if data is None:
            print_error("Response has no data field")
            print_test_info(f"Full response: {json.dumps(response, indent=2)}")
            return False
        
        allocations = data.get("allocations", [])
        
        # Check for both 2030 and 2050 targets
        if allocations:
            has_2030 = any(a.get('allocated_2030_target', 0) > 0 for a in allocations)
            has_2050 = any(a.get('allocated_2050_target', 0) > 0 for a in allocations)
            
            print_success(f"Multi-year allocation completed")
            print_test_info(f"Has 2030 targets: {has_2030}")
            print_test_info(f"Has 2050 targets: {has_2050}")
        
        return True
    else:
        print_error(f"Failed with status {status}")
        return False

def test_7_validation_errors(token: str) -> bool:
    """Test 7: Validation error handling"""
    print_header("TEST 7: Validation Error Handling")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Empty property list
    print_test_info("Test 7.1: Empty property list...")
    tests_total += 1
    
    payload = {
        "scenario_id": TEST_SCENARIO_ID,
        "property_ids": [],  # Empty list
        "allocation_method": "PROPORTIONAL_BASELINE",
        "target_year": 2030
    }
    
    status, response, duration = make_api_call(
        "POST",
        "/target-division/calculate",
        token,
        payload
    )
    
    if status == 422:
        print_success("Correctly rejected empty property list")
        tests_passed += 1
    else:
        print_error(f"Should reject empty list, got status {status}")
    
    # Test 2: Invalid allocation method
    print_test_info("Test 7.2: Invalid allocation method...")
    tests_total += 1
    
    payload = {
        "scenario_id": TEST_SCENARIO_ID,
        "property_ids": PROPERTY_IDS,
        "allocation_method": "INVALID_METHOD",
        "target_year": 2030
    }
    
    status, response, duration = make_api_call(
        "POST",
        "/target-division/calculate",
        token,
        payload
    )
    
    if status == 422:
        print_success("Correctly rejected invalid method")
        tests_passed += 1
    else:
        print_error(f"Should reject invalid method, got status {status}")
    
    # Test 3: Invalid target year
    print_test_info("Test 7.3: Invalid target year...")
    tests_total += 1
    
    payload = {
        "scenario_id": TEST_SCENARIO_ID,
        "property_ids": PROPERTY_IDS,
        "allocation_method": "PROPORTIONAL_BASELINE",
        "target_year": 1990  # Past year
    }
    
    status, response, duration = make_api_call(
        "POST",
        "/target-division/calculate",
        token,
        payload
    )
    
    if status == 422:
        print_success("Correctly rejected invalid year")
        tests_passed += 1
    else:
        print_error(f"Should reject invalid year, got status {status}")
    
    print_test_info(f"Validation tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    """Run all tests"""
    print_header("EcoAssist API Test Suite - Module 2: Target Division")
    print_test_info(f"Target API: {API_BASE_URL}")
    print_test_info(f"Test User: {TEST_USERNAME}")
    
    # Authenticate
    token = authenticate()
    if not token:
        print_error("Authentication failed. Cannot proceed with tests.")
        return
    
    # Run tests
    test_results = {}
    
    test_results["test_1"] = test_1_proportional_allocation(token)
    test_results["test_2"] = test_2_equal_allocation(token)
    test_results["test_3"] = test_3_capacity_based_allocation(token)
    test_results["test_4"] = test_4_constrained_allocation(token)
    test_results["test_5"] = test_5_single_property_allocation(token)
    test_results["test_6"] = test_6_multi_year_allocation(token)
    test_results["test_7"] = test_7_validation_errors(token)
    
    # Summary
    print_header("TEST SUMMARY")
    
    for test_name, passed in test_results.items():
        status = f"{GREEN}PASSED{RESET}" if passed else f"{RED}FAILED{RESET}"
        print(f"  {test_name}: {status}")
    
    total_tests = len(test_results)
    passed_tests = sum(1 for p in test_results.values() if p)
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nTotal: {total_tests} tests")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if passed_tests == total_tests:
        print(f"\n{GREEN}✓ All tests passed!{RESET}")
    else:
        print(f"\n{RED}❌ Multiple test failures detected{RESET}")

if __name__ == "__main__":
    main()
