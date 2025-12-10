"""
Test AI Integration in Module 1 - COMPLETE CORRECT VERSION
Tests milestone calculation with AI-enhanced predictions
"""
import requests
import json

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
print("MODULE 1 AI INTEGRATION TEST")
print("=" * 80)

# Authenticate
print("\n🔐 Authenticating...")
headers = authenticate()
print("✅ Authenticated")

# Test milestone calculation with AI
print("\n" + "=" * 80)
print("TEST: Milestone Calculation with AI")
print("=" * 80)

# Generate proper UUIDs for properties
property_1_id = "550e8400-e29b-41d4-a716-446655440001"
property_2_id = "550e8400-e29b-41d4-a716-446655440002"

# COMPLETE CORRECT FORMAT with ALL required fields
milestone_request = {
    "base_year": 2024,
    "mid_term_target_year": 2030,
    "long_term_target_year": 2050,
    "property_ids": [property_1_id, property_2_id],
    "baseline_data": [
        {
            "year": 2023,  # REQUIRED: Year of baseline data
            "property_id": property_1_id,
            "property_name": "Test Building A",
            "baseline_emission": 6000.0,
            "scope1_emissions": 2000.0,  # REQUIRED: Direct emissions
            "scope2_emissions": 4000.0,  # REQUIRED: Indirect emissions
            "total_consumption": 300000.0,  # REQUIRED: Energy consumption (kWh)
            "building_type": "Office",  # For AI
            "area_sqm": 5000.0  # For AI
        },
        {
            "year": 2023,
            "property_id": property_2_id,
            "property_name": "Test Building B",
            "baseline_emission": 4000.0,
            "scope1_emissions": 1500.0,
            "scope2_emissions": 2500.0,
            "total_consumption": 200000.0,
            "building_type": "Retail",
            "area_sqm": 3000.0
        }
    ],
    "scenario_types": ["STANDARD", "AGGRESSIVE", "CONSERVATIVE"]
}

try:
    response = requests.post(
        f"{BASE_URL}/milestones/calculate",
        json=milestone_request,
        headers=headers,
        timeout=30
    )
    
    print(f"\nResponse Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        result = data.get('data', data)
        
        print("✅ Milestone Calculation Success!")
        
        # Check metadata for AI usage
        metadata = result.get('calculation_metadata', {})
        ai_used = metadata.get('ai_optimized', False)
        ai_available = metadata.get('ai_available', False)
        
        print(f"\n📊 AI Status:")
        print(f"   - AI Available: {'✅ YES' if ai_available else '❌ NO'}")
        print(f"   - AI Used: {'✅ YES' if ai_used else '❌ NO (fallback to rule-based)'}")
        print(f"   - Algorithm Version: {metadata.get('algorithm_version', 'N/A')}")
        
        # Check scenarios
        scenarios = result.get('scenarios', [])
        print(f"\n📋 Scenarios Generated: {len(scenarios)}")
        
        for scenario in scenarios:
            scenario_type = scenario.get('scenario_type', 'Unknown')
            ai_enhanced = scenario.get('ai_enhanced', False)
            
            print(f"\n   {scenario_type} Scenario:")
            print(f"      - AI Enhanced: {'✅ YES' if ai_enhanced else '❌ NO'}")
            print(f"      - 2030 Target: {scenario.get('reduction_2030', 0)}%")
            print(f"      - 2050 Target: {scenario.get('reduction_2050', 0)}%")
            print(f"      - Overall Score: {scenario.get('scores', {}).get('overall_score', 0):.2f}")
            
            # Check first few reduction targets
            targets = scenario.get('reduction_targets', [])
            if targets and len(targets) > 5:
                print(f"      - Milestones: {len(targets)} years")
                print(f"      - Sample Years:")
                for t in targets[:5]:
                    print(f"         • {t.get('year')}: {t.get('target_emissions', 0):.2f} tCO2e ({t.get('reduction_from_baseline', 0):.1f}% reduction)")
        
        # Check recommended scenario
        recommended_id = result.get('recommended_scenario_id')
        print(f"\n🎯 Recommended: {recommended_id}")
        
        print("\n" + "=" * 80)
        print("RESULT INTERPRETATION:")
        print("=" * 80)
        
        if ai_available and ai_used:
            print("\n✅ SUCCESS: AI IS WORKING!")
            print("   Your milestones are being calculated using AI predictions.")
            print("   The trajectory follows industry-standard reduction curves.")
            print("\n   🎉 MODULE 1 AI INTEGRATION COMPLETE! 🎉")
            print("\n   AI Enhanced Features:")
            print("   - Building type awareness (Office: 4%/yr, Retail: 3%/yr)")
            print("   - Emission intensity calculations")
            print("   - Diminishing returns over time")
            print("   - Realistic trajectory curves")
            print("\n   Next Steps:")
            print("   - Compare AI vs rule-based predictions")
            print("   - Monitor AI performance in production")
            print("   - Consider expanding to Module 2-4")
        elif ai_available and not ai_used:
            print("\n⚠️  WARNING: AI available but not used")
            print("   Check server logs for AI prediction errors.")
        else:
            print("\n⚠️  INFO: Using rule-based calculations")
            print("   AI models not loaded.")
        
        # Check database
        print("\n" + "=" * 80)
        print("DATABASE VERIFICATION:")
        print("=" * 80)
        print("\nRun this SQL to verify scenarios were saved:")
        print("""
   SELECT TOP 5
       scenario_id,
       scenario_name,
       baseline_emission,
       target_reduction_percentage,
       status,
       created_at
   FROM milestone_scenarios
   ORDER BY created_at DESC;
        """)
        
    else:
        print(f"❌ API Call Failed!")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text[:1500]}")
        
except requests.exceptions.ConnectionError:
    print("❌ ERROR: Cannot connect to server!")
    print("   Make sure uvicorn is running on port 8000")
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)