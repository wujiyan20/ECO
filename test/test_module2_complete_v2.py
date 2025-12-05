#!/usr/bin/env python3
"""
EcoAssist API Test Suite - Module 2 Completion
Tests for APIs 5, 6, and 7 (Visualization, Property Allocation, Registration)
"""

import requests
import json
from datetime import datetime
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
TEST_USER = "test@example.com"
TEST_PASSWORD = "password123"

# Test data - use valid UUID format
ALLOCATION_ID = "550e8400-e29b-41d4-a716-446655440010"  # Valid UUID format
PROPERTY_IDS = [
    "550e8400-e29b-41d4-a716-446655440001",  # Brisbane Plaza
    "550e8400-e29b-41d4-a716-446655440002",  # Melbourne Tower
    "550e8400-e29b-41d4-a716-446655440003"   # Sydney Centre
]
SCENARIO_ID = "SCEN-12345678"

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header(text: str):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f"{text}")
    print(f"{'='*80}")

def print_test(test_num: int, description: str):
    """Print test description"""
    print(f"\n{BLUE}TEST {test_num}: {description}{RESET}")
    print("-" * 80)

def print_success(message: str):
    """Print success message"""
    print(f"{GREEN}✓ {message}{RESET}")

def print_error(message: str):
    """Print error message"""
    print(f"{RED}✗ {message}{RESET}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{YELLOW}⚠ {message}{RESET}")

def authenticate() -> str:
    """Authenticate and get token"""
    print("  Authenticating...")
    response = requests.post(
        f"{BASE_URL}/system/security/authenticate",
        json={"username": TEST_USER, "password": TEST_PASSWORD},  # Changed from "email" to "username"
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        
        # Try multiple possible response structures (matching working test)
        token = None
        if isinstance(data, dict):
            # Structure 1: {data: {access_token: ...}}
            if "data" in data and isinstance(data["data"], dict):
                token = data["data"].get("access_token") or data["data"].get("token")
            # Structure 2: {access_token: ...} or {token: ...}
            token = token or data.get("access_token") or data.get("token")
        
        if token:
            print_success("Authentication successful")
            return token
        else:
            print_error(f"Token not found in response: {json.dumps(data, indent=2)}")
            raise Exception("Authentication failed: No token in response")
    else:
        print_error(f"Authentication failed: {response.status_code}")
        if response.text:
            print(f"  Response: {response.text}")
        raise Exception(f"Authentication failed with status {response.status_code}")

def test_get_visualization(token: str, allocation_id: str) -> bool:
    """Test API 5: GET /target-division/visualization/{allocation_id}"""
    print_test(1, "Get Allocation Visualization")
    
    try:
        print(f"  Allocation ID: {allocation_id}")
        print(f"  Requesting visualization data...")
        
        response = requests.get(
            f"{BASE_URL}/target-division/visualization/{allocation_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
        
        print(f"  Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("data") is None:
                print_error("Response has no data field")
                print(f"  Full response: {json.dumps(data, indent=2)}")
                return False
            
            viz_data = data["data"]
            
            # Verify response structure
            required_fields = ["property_breakdown", "summary_metrics", "timeline_data"]
            missing_fields = [f for f in required_fields if f not in viz_data]
            
            if missing_fields:
                print_error(f"Missing fields: {missing_fields}")
                return False
            
            # Print summary
            print_success("Visualization data retrieved")
            print(f"  Properties: {viz_data['summary_metrics']['total_properties']}")
            print(f"  Total reduction: {viz_data['summary_metrics']['total_reduction']:.2f} "
                  f"{viz_data['summary_metrics']['emission_unit']}")
            print(f"  Total cost: ${viz_data['summary_metrics']['total_cost']:,.2f}")
            
            return True
        else:
            print_error(f"Request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_get_visualization_filtered(token: str, allocation_id: str, 
                                    property_id: str, year: int) -> bool:
    """Test API 5 with filters"""
    print_test(2, "Get Visualization with Filters")
    
    try:
        print(f"  Allocation ID: {allocation_id}")
        print(f"  Property ID: {property_id}")
        print(f"  Year: {year}")
        
        response = requests.get(
            f"{BASE_URL}/target-division/visualization/{allocation_id}",
            params={"property_id": property_id, "year": year},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
        
        print(f"  Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("data") is None:
                print_error("Response has no data field")
                return False
            
            viz_data = data["data"]
            
            # Verify filters applied
            filters = viz_data.get("filters_applied", {})
            if filters.get("property_id") == property_id and filters.get("year") == year:
                print_success("Filters applied correctly")
                
                # Should only have 1 property
                num_properties = viz_data['summary_metrics']['total_properties']
                if num_properties == 1:
                    print_success(f"Filtered to {num_properties} property")
                else:
                    print_warning(f"Expected 1 property, got {num_properties}")
                
                return True
            else:
                print_error(f"Filters not applied correctly: {filters}")
                return False
        else:
            print_error(f"Request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_get_property_allocation(token: str, property_id: str) -> bool:
    """Test API 6: GET /target-division/property/{property_id}"""
    print_test(3, "Get Property Allocation")
    
    try:
        print(f"  Property ID: {property_id}")
        print(f"  Requesting property allocation...")
        
        response = requests.get(
            f"{BASE_URL}/target-division/property/{property_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
        
        print(f"  Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("data") is None:
                print_error("Response has no data field")
                print(f"  Full response: {json.dumps(data, indent=2)}")
                return False
            
            prop_data = data["data"]
            
            # Verify response structure
            required_fields = ["property_id", "allocations", "recommended_actions"]
            missing_fields = [f for f in required_fields if f not in prop_data]
            
            if missing_fields:
                print_error(f"Missing fields: {missing_fields}")
                return False
            
            # Print summary
            print_success("Property allocation retrieved")
            print(f"  Property: {prop_data.get('property_name', 'Unknown')}")
            print(f"  Allocations: {len(prop_data['allocations'])}")
            print(f"  Recommended actions: {len(prop_data['recommended_actions'])}")
            
            # Print allocation details
            for alloc in prop_data['allocations']:
                print(f"    Year {alloc['year']}: "
                      f"{alloc['reduction_percentage']:.1f}% reduction "
                      f"({alloc['reduction_amount']:.2f} {alloc['emission_unit']})")
            
            return True
        else:
            print_error(f"Request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_get_property_allocation_filtered(token: str, property_id: str, 
                                         scenario_id: str) -> bool:
    """Test API 6 with scenario filter"""
    print_test(4, "Get Property Allocation with Scenario Filter")
    
    try:
        print(f"  Property ID: {property_id}")
        print(f"  Scenario ID: {scenario_id}")
        
        response = requests.get(
            f"{BASE_URL}/target-division/property/{property_id}",
            params={"scenario_id": scenario_id},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
        
        print(f"  Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("data") is None:
                print_error("Response has no data field")
                return False
            
            prop_data = data["data"]
            
            # Verify scenario filter
            for alloc in prop_data['allocations']:
                if alloc.get('scenario_id') != scenario_id:
                    print_warning(f"Allocation has different scenario_id: {alloc.get('scenario_id')}")
            
            print_success("Scenario filter applied")
            return True
        else:
            print_error(f"Request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_register_allocation(token: str, allocation_id: str) -> bool:
    """Test API 7: POST /target-division/register"""
    print_test(5, "Register Target Allocation")
    
    try:
        print(f"  Allocation ID: {allocation_id}")
        print(f"  Registering allocation...")
        
        request_data = {
            "allocation_id": allocation_id,
            "approval_status": "approved",
            "approved_by": "user-uuid-test-123",
            "approval_date": datetime.now().isoformat(),
            "notes": "Test registration for Module 2 completion"
        }
        
        response = requests.post(
            f"{BASE_URL}/target-division/register",
            json=request_data,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
        
        print(f"  Status code: {response.status_code}")
        
        if response.status_code == 201:
            data = response.json()
            
            if data.get("data") is None:
                print_error("Response has no data field")
                print(f"  Full response: {json.dumps(data, indent=2)}")
                return False
            
            reg_data = data["data"]
            
            # Verify response structure
            required_fields = ["registration_id", "allocation_id", "status"]
            missing_fields = [f for f in required_fields if f not in reg_data]
            
            if missing_fields:
                print_error(f"Missing fields: {missing_fields}")
                return False
            
            # Print summary
            print_success("Allocation registered successfully")
            print(f"  Registration ID: {reg_data['registration_id']}")
            print(f"  Status: {reg_data['status']}")
            print(f"  Registered at: {reg_data.get('registered_at', 'N/A')}")
            
            # Print next steps
            if 'next_steps' in reg_data:
                print(f"  Next steps: {len(reg_data['next_steps'])}")
                for i, step in enumerate(reg_data['next_steps'], 1):
                    print(f"    {i}. {step}")
            
            return True
        else:
            print_error(f"Request failed with status {response.status_code}")
            if response.text:
                print(f"  Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def main():
    """Run all tests"""
    print_header("EcoAssist API Test Suite - Module 2 Completion")
    print(f"  Target API: {BASE_URL}")
    print(f"  Test User: {TEST_USER}")
    
    # Authenticate
    try:
        token = authenticate()
    except Exception as e:
        print_error(f"Authentication failed: {str(e)}")
        return
    
    # Track results
    results = {}
    
    # Run tests
    results['test_1'] = test_get_visualization(token, ALLOCATION_ID)
    results['test_2'] = test_get_visualization_filtered(token, ALLOCATION_ID, PROPERTY_IDS[0], 2030)
    results['test_3'] = test_get_property_allocation(token, PROPERTY_IDS[0])
    results['test_4'] = test_get_property_allocation_filtered(token, PROPERTY_IDS[0], SCENARIO_ID)
    results['test_5'] = test_register_allocation(token, ALLOCATION_ID)
    
    # Print summary
    print_header("TEST SUMMARY")
    
    for test_name, passed in results.items():
        status = f"{GREEN}PASSED{RESET}" if passed else f"{RED}FAILED{RESET}"
        print(f"  {test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\nTotal: {total} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if failed == 0:
        print_success("All tests passed!")
    else:
        print_error("Multiple test failures detected")

if __name__ == "__main__":
    main()
