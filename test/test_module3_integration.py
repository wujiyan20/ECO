"""
Test Module 3: Long-term Planning APIs
Tests planning calculation and database integration
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
print("MODULE 3 TEST: LONG-TERM PLANNING WITH DATABASE INTEGRATION")
print("=" * 80)

# Authenticate
print("\n🔐 Authenticating...")
headers = authenticate()
if "Authorization" in headers:
    print("✅ Authenticated successfully")
else:
    print("⚠️  Running without authentication")

# =============================================================================
# TEST 1: CALCULATE LONG-TERM PLAN
# =============================================================================

print("\n" + "=" * 80)
print("TEST 1: Calculate Long-term Plan API")
print("=" * 80)

# Use test IDs
test_scenario_id = "550e8400-e29b-41d4-a716-446655440000"
test_allocation_id = "550e8400-e29b-41d4-a716-446655440010"

planning_request = {
    "scenario_id": test_scenario_id,
    "allocation_id": test_allocation_id,
    "planning_horizon": {
        "start_year": 2025,
        "end_year": 2050,
        "evaluation_intervals": "annual"
    },
    "budget_constraints": {
        "total_budget": 5000000.0,
        "annual_budget_limit": 200000.0,
        "currency": "USD",
        "cost_escalation_rate": 3.0
    },
    "strategy_preferences": {
        "renewable_energy_priority": 0.4,
        "energy_efficiency_priority": 0.4,
        "behavioral_change_priority": 0.2
    }
}

try:
    response = requests.post(
        f"{BASE_URL}/planning/long-term/calculate",
        json=planning_request,
        headers=headers,
        timeout=30
    )
    
    print(f"\nResponse Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        result = data.get('data', data)
        
        print("✅ Planning API Success!")
        print(f"\n   Response Summary:")
        
        # Check for planning patterns
        patterns = result.get('planning_patterns', [])
        metadata = result.get('calculation_metadata', {})
        
        print(f"   - Patterns Generated: {len(patterns)}")
        print(f"   - Scenario ID: {metadata.get('scenario_id', 'N/A')}")
        print(f"   - Planning Horizon: {metadata.get('planning_horizon', {}).get('start_year', 'N/A')} - {metadata.get('planning_horizon', {}).get('end_year', 'N/A')}")
        
        if patterns:
            print(f"\n   Planning Patterns:")
            for i, pattern in enumerate(patterns, 1):
                pattern_name = pattern.get('pattern_name', 'N/A')
                pattern_type = pattern.get('pattern_type', 'N/A')
                financial = pattern.get('financial_summary', {})
                total_investment = financial.get('total_investment', 0)
                total_reduction = financial.get('total_reduction_achieved', 0)
                payback_period = financial.get('payback_period', 0)
                
                print(f"      {i}. {pattern_name} ({pattern_type})")
                print(f"         Investment: ${total_investment:,.2f}")
                print(f"         Reduction: {total_reduction:,.2f} tCO2e")
                print(f"         Payback: {payback_period:.1f} years")
        
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
    
    # Check plans
    print("\n--- Long-term Plans ---")
    cursor.execute("""
        SELECT TOP 5
            plan_id,
            plan_name,
            pattern_type,
            start_year,
            end_year,
            total_investment,
            total_reduction_target,
            overall_roi_years,
            total_actions,
            status,
            created_at
        FROM long_term_plans
        ORDER BY created_at DESC
    """)
    
    plans = cursor.fetchall()
    if plans:
        print(f"\n✅ Found {len(plans)} recent plans:")
        for plan in plans:
            print(f"   - {plan.plan_name}")
            print(f"     ID: {plan.plan_id}")
            print(f"     Pattern: {plan.pattern_type}")
            print(f"     Years: {plan.start_year} - {plan.end_year}")
            print(f"     Investment: ${plan.total_investment:,.2f}")
            print(f"     Reduction: {plan.total_reduction_target:,.2f} tCO2e")
            print(f"     ROI: {plan.overall_roi_years:.1f} years" if plan.overall_roi_years else "     ROI: N/A")
            print(f"     Annual Plans: {plan.total_actions}")
            print(f"     Status: {plan.status}")
            print(f"     Created: {plan.created_at}")
            print()
    else:
        print("⚠️  No plans found in database")
        print("   The database save might not be working yet")
    
    # Check annual plans
    print("--- Annual Action Plans ---")
    cursor.execute("""
        SELECT COUNT(*) as count
        FROM annual_action_plans
    """)
    
    count = cursor.fetchone()[0]
    print(f"Total annual plans in database: {count}")
    
    if count > 0:
        cursor.execute("""
            SELECT TOP 5
                aap.year,
                aap.total_investment,
                aap.target_reduction,
                aap.action_count,
                aap.status,
                ltp.plan_name
            FROM annual_action_plans aap
            JOIN long_term_plans ltp ON aap.plan_id = ltp.plan_id
            ORDER BY aap.created_at DESC
        """)
        
        annual_plans = cursor.fetchall()
        print(f"\n✅ Recent annual plans:")
        for ap in annual_plans:
            print(f"   - Year {ap.year} ({ap.plan_name})")
            print(f"     Investment: ${ap.total_investment:,.2f}")
            print(f"     Reduction: {ap.target_reduction:,.2f} tCO2e")
            print(f"     Actions: {ap.action_count}")
            print(f"     Status: {ap.status}")
            print()
    
    # Check today's records
    print("\n--- Today's Records ---")
    cursor.execute("""
        SELECT COUNT(*) as count
        FROM long_term_plans
        WHERE CAST(created_at AS DATE) = CAST(GETDATE() AS DATE)
    """)
    
    today_count = cursor.fetchone()[0]
    
    if today_count > 0:
        print(f"✅ Found {today_count} plan(s) created TODAY!")
        print("   Database integration is WORKING!")
        
        # Count annual plans for today's plans
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM annual_action_plans aap
            JOIN long_term_plans ltp ON aap.plan_id = ltp.plan_id
            WHERE CAST(ltp.created_at AS DATE) = CAST(GETDATE() AS DATE)
        """)
        
        annual_count = cursor.fetchone()[0]
        print(f"✅ Found {annual_count} annual plan(s) created TODAY!")
        
    else:
        print(f"⚠️  No plans created today")
        print("   The planning service might not be initialized")
        print("   Check: Did you add initialize_planning_service() to main_api_application.py?")
    
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
print("   -- Check today's plans")
print("   SELECT ")
print("       plan_id,")
print("       plan_name,")
print("       pattern_type,")
print("       total_investment,")
print("       total_reduction_target,")
print("       status,")
print("       created_at")
print("   FROM long_term_plans")
print("   WHERE CAST(created_at AS DATE) = CAST(GETDATE() AS DATE)")
print("   ORDER BY created_at DESC;")
print("")
print("   -- Check annual plans")
print("   SELECT TOP 10")
print("       aap.year,")
print("       aap.total_investment,")
print("       aap.target_reduction,")
print("       ltp.plan_name")
print("   FROM annual_action_plans aap")
print("   JOIN long_term_plans ltp ON aap.plan_id = ltp.plan_id")
print("   ORDER BY aap.created_at DESC;")

print("\n🎯 Expected Results:")
print("   ✅ New plan record in long_term_plans")
print("   ✅ Annual plan records in annual_action_plans (25 years)")
print("   ✅ Status = 'calculated'")
print("   ✅ Timestamps from today")

print("\n✅ Tests Completed!")
print("=" * 80)
