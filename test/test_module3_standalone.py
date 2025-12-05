#!/usr/bin/env python3
"""
EcoAssist API Test Suite - Module 3: Long-term Planning
Standalone test script for Module 3 APIs

Usage:
    python test_module3_standalone.py

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
SCENARIO_ID = "550e8400-e29b-41d4-a716-446655440000"
ALLOCATION_ID = "550e8400-e29b-41d4-a716-446655440010"
PLAN_ID = "550e8400-e29b-41d4-a716-446655440030"

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
    
    if response.status_code == 200:
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

def test_calculate_long_term_plan(token: str) -> Dict[str, Any]:
    """Test API 8: Calculate Long-term Plan"""
    print_header("TEST 1: Calculate Long-term Plan")
    print_info(f"Scenario ID: {SCENARIO_ID}")
    print_info(f"Allocation ID: {ALLOCATION_ID}")
    
    payload = {
        "scenario_id": SCENARIO_ID,
        "allocation_id": ALLOCATION_ID,
        "planning_horizon": {
            "start_year": 2025,
            "end_year": 2050,
            "evaluation_intervals": "annual"
        },
        "budget_constraints": {
            "total_budget": 500000.00,
            "annual_budget_limit": 50000.00,
            "currency": "USD",
            "cost_escalation_rate": 3.0
        },
        "strategy_preferences": {
            "renewable_energy_priority": 0.4,
            "energy_efficiency_priority": 0.35,
            "behavioral_change_priority": 0.25
        }
    }
    
    print_info("Requesting long-term planning calculation...")
    response = requests.post(
        f"{BASE_URL}/planning/long-term/calculate",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=30
    )
    
    print_info(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print_success("Long-term plan calculated successfully")
        
        if "data" in data:
            planning_data = data["data"]
            patterns = planning_data.get("planning_patterns", [])
            print_info(f"Planning patterns generated: {len(patterns)}")
            
            for i, pattern in enumerate(patterns, 1):
                print_info(f"  Pattern {i}: {pattern.get('pattern_name', 'Unknown')}")
                annual_plans = pattern.get("annual_plan", [])
                print_info(f"    Annual plans: {len(annual_plans)}")
                
                if pattern.get("financial_summary"):
                    fs = pattern["financial_summary"]
                    print_info(f"    Total investment: ${fs.get('total_investment', 0):,.2f}")
                    print_info(f"    ROI years: {fs.get('overall_roi_years', 0):.1f}")
        
        return data
    else:
        print_error(f"Request failed with status {response.status_code}")
        print_info(f"Response: {response.text}")
        return None


def test_get_visualization(token: str, plan_id: str):
    """Test API 9: Get Visualization"""
    print_header("TEST 2: Get Long-term Plan Visualization")
    print_info(f"Plan ID: {plan_id}")
    
    response = requests.get(
        f"{BASE_URL}/planning/long-term/visualization/{plan_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    
    print_info(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print_success("Visualization data retrieved successfully")
        
        if "data" in data:
            viz_data = data["data"]
            timeline = viz_data.get("timeline_data", [])
            print_info(f"Timeline data points: {len(timeline)}")
            
            if viz_data.get("action_distribution"):
                dist = viz_data["action_distribution"]
                by_type = dist.get("by_type", {})
                print_info(f"Action types: {len(by_type)}")
                for action_type, count in by_type.items():
                    print_info(f"  {action_type}: {count}")
            
            if viz_data.get("cost_benefit_analysis"):
                cba = viz_data["cost_benefit_analysis"]
                print_info(f"Breakeven year: {cba.get('breakeven_year', 'N/A')}")
                print_info(f"Total ROI: {cba.get('total_roi_percentage', 0):.1f}%")
        
        return data
    else:
        print_error(f"Request failed with status {response.status_code}")
        print_info(f"Response: {response.text}")
        return None


def test_register_plan(token: str, pattern_id: str):
    """Test API 10: Register Plan"""
    print_header("TEST 3: Register Long-term Plan")
    print_info(f"Pattern ID: {pattern_id}")
    
    payload = {
        "pattern_id": pattern_id,
        "plan_name": "Brisbane 2025-2050 Implementation Plan",
        "approved_by": "550e8400-e29b-41d4-a716-446655440099",
        "approval_date": datetime.now().strftime("%Y-%m-%d"),
        "comments": "Approved for implementation - Test registration"
    }
    
    print_info("Registering plan...")
    print_info(f"Payload: {json.dumps(payload, indent=2)}")  # DEBUG: Show payload
    
    response = requests.post(
        f"{BASE_URL}/planning/long-term/register",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=30
    )
    
    print_info(f"Status code: {response.status_code}")
    
    if response.status_code in [200, 201]:
        data = response.json()
        print_success("Plan registered successfully")
        
        if "data" in data:
            reg_data = data["data"]
            print_info(f"Plan ID: {reg_data.get('plan_id', 'N/A')}")
            print_info(f"Status: {reg_data.get('status', 'N/A')}")
            
            next_steps = reg_data.get("next_steps", [])
            if next_steps:
                print_info("Next steps:")
                for step in next_steps:
                    print_info(f"  - {step}")
        
        return data
    else:
        print_error(f"Request failed with status {response.status_code}")
        print_info(f"Response: {response.text}")
        
        # Try to parse error details
        try:
            error_data = response.json()
            if "detail" in error_data:
                print_info(f"Error details: {json.dumps(error_data['detail'], indent=2)}")
        except:
            pass
        
        return None


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

def main():
    """Run all tests"""
    print_header("EcoAssist API Test Suite - Module 3: Long-term Planning")
    print_info(f"Target API: {BASE_URL}")
    print_info(f"Test User: {TEST_USER}")
    
    # Authenticate
    token = authenticate()
    if not token:
        print_error("Authentication failed. Cannot proceed with tests.")
        return
    
    print()
    
    # Test 1: Calculate long-term plan
    result1 = test_calculate_long_term_plan(token)
    test1_passed = result1 is not None
    
    # Extract pattern_id for subsequent tests
    pattern_id = None
    if result1 and "data" in result1:
        patterns = result1["data"].get("planning_patterns", [])
        if patterns and len(patterns) > 0:
            pattern_id = patterns[0].get("pattern_id")
    
    print()
    
    # Test 2: Get visualization
    test_plan_id = pattern_id if pattern_id else PLAN_ID
    result2 = test_get_visualization(token, test_plan_id)
    test2_passed = result2 is not None
    
    print()
    
    # Test 3: Register plan
    test_pattern_id = pattern_id if pattern_id else "550e8400-e29b-41d4-a716-446655440020"
    result3 = test_register_plan(token, test_pattern_id)
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
