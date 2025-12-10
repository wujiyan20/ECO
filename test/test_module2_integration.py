"""
Test Module 2: Target Division/Allocation APIs
Tests allocation calculation and database integration
"""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000/api/v1"

def authenticate():
    """Get authentication token"""
    auth_request = {
        "username": "test@example.com",
        "password": "password123"
    }
    
    response = requests.post(
        f"{BASE_URL}/system/security/authenticate",
        json=auth_request,
        timeout=10
    )
    
    if response.status_code == 200:
        auth_data = response.json()
        token = auth_data.get('access_token') or auth_data.get('data', {}).get('access_token')
        if token:
            return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    return {"Content-Type": "application/json"}

print("=" * 80)
print("MODULE 2 TEST: TARGET ALLOCATION WITH DATABASE INTEGRATION")
print("=" * 80)

# Authenticate
print("\n🔐 Authenticating...")
headers = authenticate()
if "Authorization" in headers:
    print("✅ Authenticated successfully")
else:
    print("⚠️  Running without authentication")

# =============================================================================
# TEST 1: CALCULATE TARGET ALLOCATION
# =============================================================================

print("\n" + "=" * 80)
print("TEST 1: Calculate Target Allocation API")
print("=" * 80)

# First, get a scenario_id from database (or use a test one)
test_scenario_id = "550e8400-e29b-41d4-a716-446655440000"  # Use test UUID

allocation_request = {
    "scenario_id": test_scenario_id,
    "property_ids": [
        "550e8400-e29b-41d4-a716-446655440001",
        "550e8400-e29b-41d4-a716-446655440002"
    ],
    "total_reduction_target": 1000.0,
    "target_years": [2030, 2050],
    "allocation_method": "PROPORTIONAL_BASELINE",
    "constraints": {
        "min_reduction_per_property": 50.0,
        "max_reduction_per_property": 800.0,
        "total_budget_limit": 500000.0
    }
}

try:
    response = requests.post(
        f"{BASE_URL}/target-division/calculate",
        json=allocation_request,
        headers=headers,
        timeout=30
    )
    
    print(f"\nResponse Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        result = data.get('data', data)
        
        print("✅ Allocation API Success!")
        print(f"\n   Response Summary:")
        
        # Check for allocations
        allocations = result.get('allocations', [])
        allocation_summary = result.get('allocation_summary', {})
        
        print(f"   - Total Properties: {allocation_summary.get('total_properties', len(allocations))}")
        print(f"   - Allocation Method: {allocation_summary.get('allocation_method', 'N/A')}")
        print(f"   - Total Allocated (2030): {allocation_summary.get('total_allocated_2030', 0):.2f}")
        print(f"   - Total Allocated (2050): {allocation_summary.get('total_allocated_2050', 0):.2f}")
        print(f"   - Total Allocated: {allocation_summary.get('total_allocated', 0):.2f}")
        
        if allocations:
            print(f"\n   Property Allocations:")
            for i, alloc in enumerate(allocations[:3], 1):  # Show first 3
                property_id = alloc.get('property_id', 'N/A')
                alloc_2030 = alloc.get('allocated_2030_target', 0)
                alloc_2050 = alloc.get('allocated_2050_target', 0)
                print(f"      {i}. Property {property_id[:8]}... - 2030: {alloc_2030:.2f}, 2050: {alloc_2050:.2f}")
        
        print(f"\n✅ DATABASE INTEGRATION STATUS:")
        print(f"   The allocation was calculated successfully!")
        print(f"   Check SQL Server to see if allocation was saved.")
        print(f"\n   Run this query in SSMS:")
        print(f"   SELECT TOP 5 * FROM target_allocations ORDER BY created_at DESC;")
        print(f"   SELECT TOP 10 * FROM property_targets ORDER BY created_at DESC;")
        
    elif response.status_code == 401:
        print(f"❌ Authentication Required!")
        print(f"   Update credentials in the test script")
    else:
        print(f"❌ API Call Failed!")
        print(f"   Response: {response.text[:500]}")
        
except requests.exceptions.ConnectionError:
    print("❌ ERROR: Cannot connect to server!")
    print("   Make sure uvicorn is running on port 8000")
except Exception as e:
    print(f"❌ ERROR: {e}")

# =============================================================================
# TEST 2: VERIFY DATABASE RECORDS
# =============================================================================

print("\n" + "=" * 80)
print("TEST 2: Verify Database Records")
print("=" * 80)

try:
    import pyodbc
    
    conn_str = (
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=jiyan;'
        'DATABASE=EcoAssist;'
        'Trusted_Connection=yes;'
        'TrustServerCertificate=yes;'
    )
    
    print("\n📊 Connecting to database...")
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    # Check allocations
    print("\n--- Target Allocations ---")
    cursor.execute("""
        SELECT TOP 3
            allocation_id,
            allocation_name,
            allocation_strategy,
            total_properties,
            total_target_reduction,
            status,
            created_at
        FROM target_allocations
        ORDER BY created_at DESC
    """)
    
    allocations = cursor.fetchall()
    if allocations:
        print(f"✅ Found {len(allocations)} recent allocations:")
        for alloc in allocations:
            print(f"   - {alloc.allocation_name}")
            print(f"     ID: {alloc.allocation_id}")
            print(f"     Strategy: {alloc.allocation_strategy}")
            print(f"     Properties: {alloc.total_properties}")
            print(f"     Total Reduction: {alloc.total_target_reduction:.2f}")
            print(f"     Status: {alloc.status}")
            print(f"     Created: {alloc.created_at}")
            print()
    else:
        print("⚠️  No allocations found in database")
        print("   The database save might not be working yet")
    
    # Check property targets
    print("--- Property Targets ---")
    cursor.execute("""
        SELECT COUNT(*) as count
        FROM property_targets
    """)
    
    count = cursor.fetchone()[0]
    print(f"Total property targets in database: {count}")
    
    if count > 0:
        cursor.execute("""
            SELECT TOP 5
                target_id,
                property_id,
                target_year,
                baseline_emission,
                allocated_reduction,
                reduction_percentage
            FROM property_targets
            ORDER BY created_at DESC
        """)
        
        targets = cursor.fetchall()
        print(f"\n✅ Recent property targets:")
        for target in targets:
            print(f"   - Property {str(target.property_id)[:8]}... ({target.target_year})")
            print(f"     Baseline: {target.baseline_emission:.2f}")
            print(f"     Reduction: {target.allocated_reduction:.2f}")
            print(f"     Percentage: {target.reduction_percentage:.2f}%")
            print()
    
    cursor.close()
    conn.close()
    
except ImportError:
    print("⚠️  pyodbc not installed")
    print("   Install with: pip install pyodbc")
    print("   Then run this test again to verify database")
except Exception as e:
    print(f"❌ Database check error: {e}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

print("\n📊 To verify Module 2 database integration, run in SSMS:")
print("")
print("   -- Check allocations")
print("   SELECT ")
print("       allocation_id,")
print("       allocation_name,")
print("       allocation_strategy,")
print("       total_properties,")
print("       total_target_reduction,")
print("       status,")
print("       created_at")
print("   FROM target_allocations")
print("   ORDER BY created_at DESC;")
print("")
print("   -- Check property targets")
print("   SELECT ")
print("       pt.property_id,")
print("       pt.target_year,")
print("       pt.baseline_emission,")
print("       pt.allocated_reduction,")
print("       pt.reduction_percentage,")
print("       ta.allocation_name")
print("   FROM property_targets pt")
print("   JOIN target_allocations ta ON pt.allocation_id = ta.allocation_id")
print("   ORDER BY pt.created_at DESC;")
print("")
print("   -- Count records")
print("   SELECT ")
print("       (SELECT COUNT(*) FROM target_allocations) as allocations,")
print("       (SELECT COUNT(*) FROM property_targets) as property_targets;")

print("\n🎯 Expected Results:")
print("   ✅ New allocation record in target_allocations")
print("   ✅ Property target records in property_targets (2 per property)")
print("   ✅ Status = 'calculated'")
print("   ✅ Timestamps from today")

print("\n✅ Tests Completed!")
print("=" * 80)
