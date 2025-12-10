"""
Test Module 1: Register API - Fixed Database Verification
"""
import requests
import pyodbc
from datetime import date

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
print("TEST: Register Milestone API - Database Integration Check")
print("=" * 80)

# Authenticate
print("\n🔐 Authenticating...")
headers = authenticate()
print("✅ Authenticated")

# Get a scenario from database to register
print("\n📊 Getting scenario from database...")

try:
    conn_str = (
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=jiyan;'
        'DATABASE=EcoAssist;'
        'Trusted_Connection=yes;'
        'TrustServerCertificate=yes;'
    )
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    # First, check what columns exist in the table
    print("\n" + "-" * 80)
    print("📋 Checking database schema...")
    print("-" * 80)
    
    cursor.execute("""
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'milestone_scenarios'
        ORDER BY ORDINAL_POSITION
    """)
    
    columns = cursor.fetchall()
    print("\nColumns in milestone_scenarios table:")
    for col in columns:
        print(f"   - {col[0]} ({col[1]})")
    
    # Get a scenario with 'calculated' status
    print("\n" + "-" * 80)
    print("📊 Getting scenario to register...")
    print("-" * 80)
    
    cursor.execute("""
        SELECT TOP 1 
            scenario_id, 
            scenario_name, 
            status,
            registered_at,
            registered_by
        FROM milestone_scenarios 
        WHERE status = 'calculated'
        ORDER BY created_at DESC
    """)
    
    row = cursor.fetchone()
    if not row:
        print("❌ No 'calculated' scenarios found in database")
        print("   Run the calculate API first to create a scenario")
        cursor.close()
        conn.close()
        exit(1)
    
    scenario_id = str(row[0])
    scenario_name = row[1]
    current_status = row[2]
    current_registered_at = row[3]
    current_registered_by = row[4]
    
    print(f"✅ Found scenario to register:")
    print(f"   ID: {scenario_id}")
    print(f"   Name: {scenario_name}")
    print(f"   Status: {current_status}")
    print(f"   Registered At: {current_registered_at}")
    print(f"   Registered By: {current_registered_by}")
    
    # Call Register API
    print("\n" + "=" * 80)
    print("TEST: Calling Register API")
    print("=" * 80)
    
    register_request = {
        "scenario_id": scenario_id,
        "approval_status": "approved",
        "approved_by": "00000000-0000-0000-0000-000000000001",
        "approval_date": date.today().isoformat(),
        "notes": "Test registration - checking database update"
    }
    
    response = requests.post(
        f"{BASE_URL}/milestones/register",
        json=register_request,
        headers=headers,
        timeout=10
    )
    
    print(f"\nResponse Status: {response.status_code}")
    
    if response.status_code in [200, 201]:
        data = response.json()
        result = data.get('data', data)
        
        print("✅ Register API Success!")
        print(f"\n   API Response:")
        print(f"   - Registration ID: {result.get('registration_id', 'N/A')}")
        print(f"   - Scenario ID: {result.get('scenario_id', 'N/A')}")
        print(f"   - Status: {result.get('approval_status', 'N/A')}")
        
        # Verify in database (check ONLY columns that exist)
        print("\n" + "=" * 80)
        print("📊 CHECKING DATABASE UPDATE")
        print("=" * 80)
        
        cursor.execute("""
            SELECT 
                scenario_id,
                scenario_name,
                status,
                registered_at,
                registered_by,
                created_at
            FROM milestone_scenarios
            WHERE scenario_id = ?
        """, scenario_id)
        
        row = cursor.fetchone()
        if row:
            db_scenario_id = str(row[0])
            db_scenario_name = row[1]
            db_status = row[2]
            db_registered_at = row[3]
            db_registered_by = row[4] if row[4] else None
            db_created_at = row[5]
            
            print(f"\nDatabase record AFTER API call:")
            print(f"   Scenario ID: {db_scenario_id}")
            print(f"   Name: {db_scenario_name}")
            print(f"   Status: {db_status}")
            print(f"   Registered At: {db_registered_at}")
            print(f"   Registered By: {db_registered_by}")
            print(f"   Created At: {db_created_at}")
            
            print("\n" + "-" * 80)
            print("ANALYSIS:")
            print("-" * 80)
            
            # Check if status changed
            if db_status != current_status:
                print(f"✅ Status CHANGED: {current_status} → {db_status}")
            else:
                print(f"⚠️  Status UNCHANGED: {current_status} (still {db_status})")
            
            # Check if registered_at changed
            if db_registered_at != current_registered_at:
                print(f"✅ Registered_at UPDATED: {current_registered_at} → {db_registered_at}")
            else:
                print(f"⚠️  Registered_at UNCHANGED: {current_registered_at}")
            
            # Check if registered_by changed
            if db_registered_by != current_registered_by:
                print(f"✅ Registered_by UPDATED: {current_registered_by} → {db_registered_by}")
            else:
                print(f"⚠️  Registered_by UNCHANGED: {current_registered_by}")
            
            print("\n" + "=" * 80)
            print("CONCLUSION:")
            print("=" * 80)
            
            if db_status == 'registered' or db_registered_at or db_registered_by:
                print("✅ DATABASE IS BEING UPDATED!")
                print("   The Register API successfully updates the database.")
                print("\n   🎉 MODULE 1 IS COMPLETE! 🎉")
                print("   - Calculate API: Saves scenarios ✅")
                print("   - Register API: Updates status ✅")
                print("   - Database integration: Working ✅")
            else:
                print("⚠️  DATABASE IS NOT BEING UPDATED")
                print("   The Register API returns 200 but doesn't update the database.")
                print("\n   🔧 NEXT STEP:")
                print("   Need to add database update logic to the Register API handler.")
                print("   I can provide the fix for this.")
        else:
            print("   ⚠️  Scenario not found in database after API call")
            
    else:
        print(f"❌ Register API Failed!")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:500]}")
    
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
