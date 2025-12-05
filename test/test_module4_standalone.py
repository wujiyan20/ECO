#!/usr/bin/env python3
"""
EcoAssist API Test Suite - Module 4: Annual Reoptimization
Standalone test script for Module 4 APIs

Usage:
    python test_module4_standalone.py

Requirements:
    pip install requests
"""

import requests
import json
from datetime import datetime
from typing import Dict, Any

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL = "http://localhost:8000/api/v1"
TEST_USER = "test@example.com"
TEST_PASSWORD = "password123"

# Test data - using valid UUIDs
PLAN_ID = "550e8400-e29b-41d4-a716-446655440030"
PROPERTY_ID = "550e8400-e29b-41d4-a716-446655440001"
TARGET_YEAR = 2026

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}")

def print_success(text: str):
    """Print success message"""
    print(f"{GREEN}✓{RESET} {text}")

def print_error(text: str):
    """Print error message"""
    print(f"{RED}✗{RESET} {text}")

def print_info(text: str):
    """Print info message"""
    print(f"  {text}")

def authenticate() -> str:
    """Authenticate and get token"""
    print_info("Authenticating...")
    response = requests.post(
        f"{BASE_URL}/system/security/authenticate",
        json={"username": TEST_USER, "password": TEST_PASSWORD},
        timeout=30
    )
    
    if response.status_code in [200, 201]:
        data = response.json()
        
        # Try multiple possible token locations
        token = None
        if isinstance(data, dict):
            # Try: data.data.access_token
            if "data" in data and isinstance(data["data"], dict):
                token = data["data"].get("access_token") or data["data"].get("token")
            # Try: data.access_token or data.token
            if not token:
                token = data.get("access_token") or data.get("token")
        
        if token:
            print_success("Authentication successful")
            return token
        else:
            # Debug: show what we got
            print_error("Token not found in response")
            print_info(f"Response structure: {json.dumps(data, indent=2)}")
            return None
    
    print_error(f"Authentication failed: {response.status_code}")
    print_info(f"Response: {response.text}")
    return None

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_calculate_reoptimization(token: str) -> Dict[str, Any]:
    """Test API 11: Calculate Reoptimization"""
    print_header("TEST 1: Calculate Annual Reoptimization")
    print_info(f"Plan ID: {PLAN_ID}")
    print_info(f"Target Year: {TARGET_YEAR}")
    
    payload = {
        "plan_id": PLAN_ID,
        "target_year": TARGET_YEAR,
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "frequency": "quarterly",
        "budget_adjustment": {
            "annual_budget_limit": 60000.00,
            "currency": "USD",
            "reason": "Performance ahead of target, increasing budget"
        }
    }
    
    print_info("Requesting reoptimization calculation...")
    print_info("Note: This will fetch actual data from EEL server (mock in test mode)")
    response = requests.post(
        f"{BASE_URL}/planning/annual/reoptimize",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=30
    )
    
    print_info(f"Status code: {response.status_code}")
    
    if response.status_code in [200, 201]:
        data = response.json()
        print_success("Reoptimization calculated successfully")
        
        if "data" in data:
            reopt_data = data["data"]
            patterns = reopt_data.get("reoptimization_patterns", [])
            print_info(f"Reoptimization patterns generated: {len(patterns)}")
            
            for i, pattern in enumerate(patterns, 1):
                print_info(f"  Pattern {i}: {pattern.get('pattern_name', 'Unknown')}")
                print_info(f"    Rationale: {pattern.get('rationale', 'N/A')[:80]}...")
                
                adjusted_plans = pattern.get("adjusted_annual_plan", [])
                if adjusted_plans:
                    for plan in adjusted_plans:
                        actions = plan.get("recommended_actions", [])
                        print_info(f"    Year {plan.get('year')}: {len(actions)} recommended actions")
                        
                        changes = plan.get("changes_from_original")
                        if changes:
                            print_info(f"      Added: {changes.get('actions_added', 0)} actions")
                            print_info(f"      Budget change: ${changes.get('budget_change', 0):,.2f}")
                
                if pattern.get("performance_comparison"):
                    perf = pattern["performance_comparison"]
                    print_info(f"    Performance status: {perf.get('status', 'N/A')}")
        
        return data
    else:
        print_error(f"Request failed with status {response.status_code}")
        print_info(f"Response: {response.text}")
        return None


def test_get_reoptimization_visualization(token: str, property_id: str, year: int):
    """Test API 12: Get Reoptimization Visualization"""
    print_header("TEST 2: Get Reoptimization Visualization")
    print_info(f"Property ID: {property_id}")
    print_info(f"Year: {year}")
    
    response = requests.get(
        f"{BASE_URL}/planning/annual/visualization/{property_id}/{year}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    
    print_info(f"Status code: {response.status_code}")
    
    if response.status_code in [200, 201]:
        data = response.json()
        print_success("Reoptimization visualization retrieved successfully")
        
        if "data" in data:
            viz_data = data["data"]
            
            if viz_data.get("performance_trends"):
                trends = viz_data["performance_trends"]
                monthly_data = trends.get("monthly_data", [])
                print_info(f"Monthly performance data points: {len(monthly_data)}")
                print_info(f"Overall trend: {trends.get('overall_trend', 'N/A')}")
            
            if viz_data.get("comparison_charts"):
                charts = viz_data["comparison_charts"]
                for chart_name, comparison in charts.items():
                    print_info(f"Comparison: {chart_name}")
                    print_info(f"  Original reduction: {comparison.get('original_total_reduction', 0):.2f}")
                    print_info(f"  Adjusted reduction: {comparison.get('adjusted_total_reduction', 0):.2f}")
                    print_info(f"  Improvement: {comparison.get('difference', 0):.2f}")
            
            if viz_data.get("key_metrics"):
                metrics = viz_data["key_metrics"]
                print_info("Key Metrics:")
                print_info(f"  Target achievement: {metrics.get('target_achievement_rate', 0):.1f}%")
                print_info(f"  Cost efficiency improvement: {metrics.get('cost_efficiency_improvement', 0):.1f}%")
                print_info(f"  Implementation progress: {metrics.get('implementation_progress', 0):.1f}%")
        
        return data
    else:
        print_error(f"Request failed with status {response.status_code}")
        print_info(f"Response: {response.text}")
        return None


def test_register_reoptimization(token: str, pattern_id: str, plan_id: str):
    """Test API 13: Register Reoptimization"""
    print_header("TEST 3: Register Annual Reoptimization")
    print_info(f"Pattern ID: {pattern_id}")
    print_info(f"Plan ID: {plan_id}")
    
    payload = {
        "reoptimization_pattern_id": pattern_id,
        "plan_id": plan_id,
        "approval_info": {
            "approved_by": "550e8400-e29b-41d4-a716-446655440099",
            "approval_date": datetime.now().strftime("%Y-%m-%d"),
            "comments": "Approved adjustments based on Q4 2025 performance - Test registration"
        }
    }
    
    print_info("Registering reoptimization...")
    response = requests.post(
        f"{BASE_URL}/planning/annual/register",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=30
    )
    
    print_info(f"Status code: {response.status_code}")
    
    if response.status_code in [200, 201]:
        data = response.json()
        print_success("Reoptimization registered successfully")
        
        if "data" in data:
            reg_data = data["data"]
            print_info(f"Reoptimization ID: {reg_data.get('reoptimization_id', 'N/A')}")
            print_info(f"Updated Plan ID: {reg_data.get('updated_plan_id', 'N/A')}")
            print_info(f"Status: {reg_data.get('status', 'N/A')}")
            
            changes = reg_data.get("changes_applied")
            if changes:
                print_info("Changes applied:")
                print_info(f"  Actions added: {changes.get('actions_added', 0)}")
                print_info(f"  Actions removed: {changes.get('actions_removed', 0)}")
                print_info(f"  Budget adjustment: ${changes.get('budget_adjustment', 0):,.2f}")
        
        return data
    else:
        print_error(f"Request failed with status {response.status_code}")
        print_info(f"Response: {response.text}")
        return None


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

def main():
    """Run all tests"""
    print_header("EcoAssist API Test Suite - Module 4: Annual Reoptimization")
    print_info(f"Target API: {BASE_URL}")
    print_info(f"Test User: {TEST_USER}")
    
    # Authenticate
    token = authenticate()
    if not token:
        print_error("Authentication failed. Cannot proceed with tests.")
        return
    
    print()
    
    # Test 1: Calculate reoptimization
    result1 = test_calculate_reoptimization(token)
    test1_passed = result1 is not None
    
    # Extract pattern_id for subsequent tests
    pattern_id = None
    if result1 and "data" in result1:
        patterns = result1["data"].get("reoptimization_patterns", [])
        if patterns and len(patterns) > 0:
            pattern_id = patterns[0].get("pattern_id")
    
    print()
    
    # Test 2: Get visualization
    result2 = test_get_reoptimization_visualization(token, PROPERTY_ID, TARGET_YEAR)
    test2_passed = result2 is not None
    
    print()
    
    # Test 3: Register reoptimization
    test_pattern_id = pattern_id if pattern_id else "550e8400-e29b-41d4-a716-446655440040"
    result3 = test_register_reoptimization(token, test_pattern_id, PLAN_ID)
    test3_passed = result3 is not None
    
    # Summary
    print_header("TEST SUMMARY")
    print_info(f"test_1 (Calculate): {'PASSED' if test1_passed else 'FAILED'}")
    print_info(f"test_2 (Visualization): {'PASSED' if test2_passed else 'FAILED'}")
    print_info(f"test_3 (Register): {'PASSED' if test3_passed else 'FAILED'}")
    
    total_tests = 3
    passed_tests = sum([test1_passed, test2_passed, test3_passed])
    
    print()
    print_info(f"Total: {total_tests} tests")
    print_info(f"Passed: {passed_tests}")
    print_info(f"Failed: {total_tests - passed_tests}")
    print_info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print_success("\n✓ All tests passed!")
    else:
        print_error(f"\n✗ {total_tests - passed_tests} test(s) failed")


if __name__ == "__main__":
    main()
