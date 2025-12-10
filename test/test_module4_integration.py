"""
Test Module 4: Annual Reoptimization APIs
Tests reoptimization calculation and database integration
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
print("MODULE 4 TEST: ANNUAL REOPTIMIZATION WITH DATABASE INTEGRATION")
print("=" * 80)

# Authenticate
print("\n🔐 Authenticating...")
headers = authenticate()
if "Authorization" in headers:
    print("✅ Authenticated successfully")
else:
    print("⚠️  Running without authentication")

# =============================================================================
# TEST 1: CALCULATE ANNUAL REOPTIMIZATION
# =============================================================================

print("\n" + "=" * 80)
print("TEST 1: Calculate Annual Reoptimization API")
print("=" * 80)

# Use test plan ID (from Module 3)
test_plan_id = "550e8400-e29b-41d4-a716-446655440030"

reoptimization_request = {
    "plan_id": test_plan_id,
    "target_year": 2026,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "frequency": "quarterly",
    "budget_adjustment": {
        "annual_budget_limit": 250000.0,
        "currency": "USD",
        "reason": "Market conditions changed"
    }
}

try:
    response = requests.post(
        f"{BASE_URL}/planning/annual/reoptimize",
        json=reoptimization_request,
        headers=headers,
        timeout=30
    )
    
    print(f"\nResponse Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        result = data.get('data', data)
        
        print("✅ Reoptimization API Success!")
        print(f"\n   Response Summary:")
        
        # Check for reoptimization patterns
        patterns = result.get('reoptimization_patterns', [])
        metadata = result.get('calculation_metadata', {})
        
        print(f"   - Patterns Generated: {len(patterns)}")
        print(f"   - Plan ID: {metadata.get('plan_id', 'N/A')}")
        print(f"   - Target Year: {metadata.get('target_year', 'N/A')}")
        print(f"   - Analysis Period: {metadata.get('analysis_period', 'N/A')}")
        
        # Check actual performance data
        actual_perf = metadata.get('actual_performance', {})
        if actual_perf:
            print(f"\n   Actual Performance:")
            print(f"   - Planned Emission: {actual_perf.get('planned_emission', 0):,.2f} tCO2e")
            print(f"   - Actual Emission: {actual_perf.get('actual_emission', 0):,.2f} tCO2e")
            print(f"   - Variance: {actual_perf.get('variance_percentage', 0):,.2f}%")
        
        if patterns:
            print(f"\n   Reoptimization Patterns:")
            for i, pattern in enumerate(patterns, 1):
                pattern_name = pattern.get('pattern_name', 'N/A')
                pattern_type = pattern.get('pattern_type', 'N/A')
                
                # Get changes summary
                changes = pattern.get('changes_from_original', {})
                budget_change = changes.get('budget_change', 0)
                timeline_change = changes.get('timeline_change', 0)
                
                print(f"      {i}. {pattern_name}")
                print(f"         Type: {pattern_type}")
                print(f"         Budget Change: ${budget_change:,.2f}")
                print(f"         Timeline Change: {timeline_change} months")
        
        print(f"\n✅ API CALL SUCCESSFUL")
        print(f"   Check database to verify records were saved")
        
    elif response.status_code == 401:
        print(f"❌ Authentication Required!")
    else:
        print(f"❌ API Call Failed!")
        print(f"   Response: {response.text[:500]}")
        
except requests.exceptions.ConnectionError:
    print("❌ ERROR: Cannot connect to server!")
    print("   Make sure uvicorn is running on port 8000")
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

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
    
    # Check reoptimizations
    print("\n--- Reoptimizations ---")
    cursor.execute("""
        SELECT TOP 5
            r.reoptimization_id,
            r.plan_id,
            r.target_year,
            r.analysis_start_date,
            r.analysis_end_date,
            r.frequency,
            r.planned_emission,
            r.actual_emission,
            r.variance,
            r.variance_percentage,
            r.performance_status,
            r.performance_trend,
            r.pattern_type,
            r.status,
            r.created_at,
            ltp.plan_name
        FROM reoptimizations r
        LEFT JOIN long_term_plans ltp ON r.plan_id = ltp.plan_id
        ORDER BY r.created_at DESC
    """)
    
    reopts = cursor.fetchall()
    if reopts:
        print(f"\n✅ Found {len(reopts)} recent reoptimizations:")
        for reopt in reopts:
            print(f"\n   Reoptimization for: {reopt.plan_name or 'Unknown Plan'}")
            print(f"   - ID: {reopt.reoptimization_id}")
            print(f"   - Target Year: {reopt.target_year}")
            print(f"   - Analysis Period: {reopt.analysis_start_date} to {reopt.analysis_end_date}")
            print(f"   - Frequency: {reopt.frequency}")
            print(f"   - Planned Emission: {reopt.planned_emission:,.2f} tCO2e")
            print(f"   - Actual Emission: {reopt.actual_emission:,.2f} tCO2e")
            print(f"   - Variance: {reopt.variance:,.2f} tCO2e ({reopt.variance_percentage:.2f}%)")
            print(f"   - Performance: {reopt.performance_status} ({reopt.performance_trend})")
            print(f"   - Pattern Type: {reopt.pattern_type}")
            print(f"   - Status: {reopt.status}")
            print(f"   - Created: {reopt.created_at}")
    else:
        print("⚠️  No reoptimizations found in database")
        print("   The database save might not be working yet")
    
    # Check today's records
    print("\n--- Today's Records ---")
    cursor.execute("""
        SELECT COUNT(*) as count
        FROM reoptimizations
        WHERE CAST(created_at AS DATE) = CAST(GETDATE() AS DATE)
    """)
    
    today_count = cursor.fetchone()[0]
    
    if today_count > 0:
        print(f"✅ Found {today_count} reoptimization(s) created TODAY!")
        print("   Database integration is WORKING!")
        
        # Show today's details
        cursor.execute("""
            SELECT 
                r.target_year,
                r.variance_percentage,
                r.performance_status,
                r.pattern_type,
                ltp.plan_name
            FROM reoptimizations r
            LEFT JOIN long_term_plans ltp ON r.plan_id = ltp.plan_id
            WHERE CAST(r.created_at AS DATE) = CAST(GETDATE() AS DATE)
        """)
        
        today_reopts = cursor.fetchall()
        for tr in today_reopts:
            print(f"\n   - Plan: {tr.plan_name or 'Unknown'}")
            print(f"     Year: {tr.target_year}")
            print(f"     Variance: {tr.variance_percentage:.2f}%")
            print(f"     Status: {tr.performance_status}")
            print(f"     Pattern: {tr.pattern_type}")
        
    else:
        print(f"⚠️  No reoptimizations created today")
        print("   The reoptimization service might not be initialized")
        print("   Check: Did you add initialize_reoptimization_service() to main_api_application.py?")
    
    # Summary statistics
    print("\n--- Summary Statistics ---")
    cursor.execute("""
        SELECT 
            COUNT(*) as total_reopts,
            COUNT(DISTINCT plan_id) as unique_plans,
            AVG(variance_percentage) as avg_variance,
            COUNT(CASE WHEN performance_status = 'behind_target' THEN 1 END) as behind_count,
            COUNT(CASE WHEN performance_status = 'on_track' THEN 1 END) as on_track_count,
            COUNT(CASE WHEN performance_status = 'ahead_of_target' THEN 1 END) as ahead_count
        FROM reoptimizations
    """)
    
    stats = cursor.fetchone()
    print(f"   Total Reoptimizations: {stats.total_reopts}")
    print(f"   Unique Plans: {stats.unique_plans}")
    print(f"   Average Variance: {stats.avg_variance:.2f}%" if stats.avg_variance else "   Average Variance: N/A")
    print(f"   Behind Target: {stats.behind_count}")
    print(f"   On Track: {stats.on_track_count}")
    print(f"   Ahead of Target: {stats.ahead_count}")
    
    cursor.close()
    conn.close()
    
except ImportError:
    print("⚠️  pyodbc not installed")
    print("   Install with: pip install pyodbc")
except Exception as e:
    print(f"❌ Database check error: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

print("\n📊 Manual Verification in SSMS:")
print("")
print("   -- Check today's reoptimizations")
print("   SELECT ")
print("       r.reoptimization_id,")
print("       r.target_year,")
print("       r.variance_percentage,")
print("       r.performance_status,")
print("       r.pattern_type,")
print("       r.status,")
print("       r.created_at,")
print("       ltp.plan_name")
print("   FROM reoptimizations r")
print("   LEFT JOIN long_term_plans ltp ON r.plan_id = ltp.plan_id")
print("   WHERE CAST(r.created_at AS DATE) = CAST(GETDATE() AS DATE)")
print("   ORDER BY r.created_at DESC;")

print("\n🎯 Expected Results:")
print("   ✅ New reoptimization record in reoptimizations table")
print("   ✅ Status = 'calculated'")
print("   ✅ Performance metrics populated")
print("   ✅ Timestamps from today")

print("\n✅ Tests Completed!")
print("=" * 80)
