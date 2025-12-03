# database/examples/usage_examples.py - Usage Examples
"""
Comprehensive usage examples for the EcoAssist database package
"""

from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_quick_setup():
    """Example 1: Quick setup and basic usage"""
    print("\n" + "=" * 60)
    print("Example 1: Quick Setup")
    print("=" * 60)
    
    from database import quick_setup, create_repositories
    
    # Quick setup from environment
    manager = quick_setup()
    
    # Create repositories
    repos = create_repositories(manager)
    
    # Use repositories
    properties = repos['property'].get_all_active()
    print(f"Found {len(properties)} active properties")
    
    # Get statistics
    stats = manager.get_statistics()
    print(f"Database statistics: {stats}")
    
    # Cleanup
    manager.close()


def example_high_level_interface():
    """Example 2: Using high-level EcoAssistDatabase interface"""
    print("\n" + "=" * 60)
    print("Example 2: High-Level Interface")
    print("=" * 60)
    
    from database import EcoAssistDatabase
    
    # Use context manager (auto cleanup)
    with EcoAssistDatabase() as db:
        # Test connection
        if not db.test_connection():
            print("Connection failed!")
            return
        
        print("✓ Connected to database")
        
        # Get properties
        properties = db.property.get_all_active()
        print(f"✓ Found {len(properties)} properties")
        
        # Get dashboard metrics
        metrics = db.dashboard.get_portfolio_metrics()
        print(f"✓ Portfolio metrics: {metrics.get('total_properties')} properties")
        print(f"  Total emission: {metrics.get('total_emission', 0):,.0f} kg-CO2e")
        
        # Get statistics
        stats = db.get_statistics()
        print(f"✓ Database queries executed: {stats['query_count']}")


def example_property_operations():
    """Example 3: Property repository operations"""
    print("\n" + "=" * 60)
    print("Example 3: Property Operations")
    print("=" * 60)
    
    from database import EcoAssistDatabase, Property
    
    with EcoAssistDatabase() as db:
        # Get all active properties
        properties = db.property.get_all_active()
        print(f"Active properties: {len(properties)}")
        
        if properties:
            # Display first property
            prop = properties[0]
            print(f"\nSample property:")
            print(f"  ID: {prop.property_id}")
            print(f"  Name: {prop.name}")
            print(f"  Type: {prop.building_type}")
            print(f"  Emission: {prop.baseline_emission:,.0f} kg-CO2e")
        
        # Get by building type
        offices = db.property.get_by_building_type("Office")
        print(f"\nOffice buildings: {len(offices)}")
        
        # Get high emitters
        top_emitters = db.property.get_high_emitters(limit=5)
        print(f"\nTop 5 emitters:")
        for i, prop in enumerate(top_emitters, 1):
            print(f"  {i}. {prop.name}: {prop.baseline_emission:,.0f} kg-CO2e")
        
        # Get portfolio summary
        summary = db.property.get_portfolio_summary()
        print(f"\nPortfolio summary:")
        print(f"  Total properties: {summary.get('total_properties', 0)}")
        print(f"  Total emission: {summary.get('total_emission', 0):,.0f} kg-CO2e")
        print(f"  Total area: {summary.get('total_area', 0):,.0f} m²")
        
        # Search properties
        search_results = db.property.search_by_name("Melbourne")
        print(f"\nSearch for 'Melbourne': {len(search_results)} results")


def example_reduction_options():
    """Example 4: Reduction option operations"""
    print("\n" + "=" * 60)
    print("Example 4: Reduction Options")
    print("=" * 60)
    
    from database import EcoAssistDatabase
    
    with EcoAssistDatabase() as db:
        # Get all options
        options = db.reduction_option.get_all_active()
        print(f"Total reduction options: {len(options)}")
        
        # Get by category
        energy_efficiency = db.reduction_option.get_by_category("Energy Efficiency")
        print(f"Energy Efficiency options: {len(energy_efficiency)}")
        
        # Get quick wins
        quick_wins = db.reduction_option.get_quick_wins(max_payback=3.0)
        print(f"\nQuick wins (payback <= 3 years): {len(quick_wins)}")
        for i, opt in enumerate(quick_wins[:5], 1):
            print(f"  {i}. {opt.name}")
            print(f"     Payback: {opt.payback_period_years:.1f} years")
            print(f"     Reduction: {opt.co2_reduction_potential:,.0f} kg-CO2e")
        
        # Get high-impact options
        high_impact = db.reduction_option.get_high_impact(min_reduction=100.0, limit=5)
        print(f"\nHigh-impact options (>100 kg-CO2e): {len(high_impact)}")
        
        # Get options for specific property
        properties = db.property.get_all_active()
        if properties:
            property = properties[0]
            suitable = db.reduction_option.get_suitable_for_property(property)
            print(f"\nOptions suitable for {property.name}: {len(suitable)}")
            
            # Get recommendations
            recommendations = db.reduction_option.get_recommendation_matrix(property, max_options=5)
            print(f"\nTop 5 recommendations for {property.name}:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['name']} (Score: {rec['recommendation_score']:.0f})")


def example_milestone_scenarios():
    """Example 5: Milestone scenario operations"""
    print("\n" + "=" * 60)
    print("Example 5: Milestone Scenarios")
    print("=" * 60)
    
    from database import EcoAssistDatabase
    
    with EcoAssistDatabase() as db:
        # Get all scenarios
        scenarios = db.milestone.get_all_scenarios(limit=10)
        print(f"Total scenarios: {len(scenarios)}")
        
        if scenarios:
            # Display first scenario
            scenario = scenarios[0]
            print(f"\nSample scenario:")
            print(f"  ID: {scenario.scenario_id}")
            print(f"  Name: {scenario.name}")
            print(f"  Target year: {scenario.target_year}")
            print(f"  Reduction: {scenario.reduction_percentage:.1f}%")
            print(f"  Strategy: {scenario.strategy_type}")
            print(f"  SBT aligned: {scenario.sbt_aligned}")
        
        # Get SBT-aligned scenarios
        sbt_scenarios = db.milestone.get_sbt_aligned()
        print(f"\nSBT-aligned scenarios: {len(sbt_scenarios)}")
        
        # Get by strategy type
        aggressive = db.milestone.get_by_strategy_type("aggressive")
        balanced = db.milestone.get_by_strategy_type("balanced")
        print(f"Aggressive scenarios: {len(aggressive)}")
        print(f"Balanced scenarios: {len(balanced)}")
        
        # Compare scenarios
        if len(scenarios) >= 2:
            comparison = db.milestone.get_scenario_comparison(
                [s.scenario_id for s in scenarios[:3]]
            )
            print(f"\nScenario comparison:")
            for comp in comparison:
                print(f"  {comp['name']}: {comp['reduction_percentage']:.1f}% reduction")
                print(f"    CAPEX: ${comp['total_capex']:,.0f}")


def example_dashboard_metrics():
    """Example 6: Dashboard metrics and analytics"""
    print("\n" + "=" * 60)
    print("Example 6: Dashboard Metrics")
    print("=" * 60)
    
    from database import EcoAssistDatabase
    
    with EcoAssistDatabase() as db:
        # Get complete dashboard data
        dashboard = db.dashboard.get_complete_dashboard_data()
        
        print("Portfolio Metrics:")
        metrics = dashboard['portfolio_metrics']
        print(f"  Total properties: {metrics.get('total_properties', 0)}")
        print(f"  Total emission: {metrics.get('total_emission', 0):,.0f} kg-CO2e")
        print(f"  Average intensity: {metrics.get('avg_carbon_intensity', 0):.2f} kg-CO2e/m²")
        
        print("\nEmission KPIs:")
        kpis = dashboard['emission_kpis']
        print(f"  Scope 1: {kpis.get('scope1_percentage', 0):.1f}%")
        print(f"  Scope 2: {kpis.get('scope2_percentage', 0):.1f}%")
        print(f"  Scope 3: {kpis.get('scope3_percentage', 0):.1f}%")
        
        print("\nBuilding Type Distribution:")
        for item in dashboard['building_type_distribution']:
            print(f"  {item['building_type']}: {item['count']} properties")
        
        print("\nTop Emitters:")
        for i, prop in enumerate(dashboard['top_emitters'][:5], 1):
            print(f"  {i}. {prop['name']}: {prop['baseline_emission']:,.0f} kg-CO2e")
        
        print("\nReduction Opportunities:")
        opportunities = dashboard['reduction_opportunities']
        print(f"  Critical: {opportunities.get('critical_count', 0)} properties")
        print(f"  High: {opportunities.get('high_count', 0)} properties")
        print(f"  High priority emissions: {opportunities.get('high_priority_percentage', 0):.1f}%")


def example_historical_data():
    """Example 7: Historical data operations"""
    print("\n" + "=" * 60)
    print("Example 7: Historical Data")
    print("=" * 60)
    
    from database import EcoAssistDatabase
    
    with EcoAssistDatabase() as db:
        # Get properties
        properties = db.property.get_all_active()
        
        if properties:
            property_id = properties[0].property_id
            
            # Get consumption data
            consumption = db.historical.consumption.get_by_property(property_id, year=2024)
            print(f"Consumption records for 2024: {len(consumption)}")
            
            # Get monthly trend
            trend = db.historical.consumption.get_monthly_trend(property_id, 2020, 2024)
            print(f"\nMonthly consumption trend: {len(trend)} data points")
            
            # Year-over-year comparison
            comparison = db.historical.consumption.get_year_over_year_comparison(
                property_id, 2023, 2024
            )
            if comparison:
                print(f"\nYear-over-year comparison:")
                for fuel_type, data in comparison.items():
                    print(f"  {fuel_type}:")
                    print(f"    2023: {data.get('year_2023', 0):,.0f}")
                    print(f"    2024: {data.get('year_2024', 0):,.0f}")
                    print(f"    Change: {data.get('change_percentage', 0):,.1f}%")
            
            # Get emissions
            emissions = db.historical.emission.get_by_property(property_id, year=2024)
            print(f"\nEmission records for 2024: {len(emissions)}")
            
            # Get annual emissions
            annual = db.historical.emission.get_annual_emissions(property_id, 2024)
            if annual:
                print(f"\nAnnual emissions 2024:")
                print(f"  Total: {annual.get('total_emission', 0):,.0f} kg-CO2e")
                print(f"  Scope 1: {annual.get('total_scope1', 0):,.0f} kg-CO2e")
                print(f"  Scope 2: {annual.get('total_scope2', 0):,.0f} kg-CO2e")


def example_pricing_data():
    """Example 8: Pricing data operations"""
    print("\n" + "=" * 60)
    print("Example 8: Pricing Data")
    print("=" * 60)
    
    from database import EcoAssistDatabase
    
    with EcoAssistDatabase() as db:
        current_year = datetime.now().year
        
        # Get carbon credit prices
        carbon_prices = db.carbon_credit.carbon_prices.get_by_year(current_year)
        print(f"Carbon credit prices for {current_year}: {len(carbon_prices)}")
        
        if carbon_prices:
            for price in carbon_prices[:3]:
                print(f"  {price.credit_type}: ${price.price_per_tonne:.2f}/tonne")
        
        # Calculate offset cost
        offset_cost = db.carbon_credit.calculate_offset_cost(
            tonnes_co2=1000,
            year=current_year,
            credit_type="ACCU"
        )
        if offset_cost:
            print(f"\nOffset cost for 1000 tonnes:")
            print(f"  Type: {offset_cost['credit_type']}")
            print(f"  Price: ${offset_cost['price_per_tonne']:.2f}/tonne")
            print(f"  Total: ${offset_cost['total_cost']:,.2f}")
        
        # Get renewable energy prices
        re_prices = db.carbon_credit.renewable_energy_prices.get_by_year(current_year)
        print(f"\nRenewable energy prices: {len(re_prices)}")
        
        # Get price trend
        trend = db.carbon_credit.carbon_prices.get_price_trend("ACCU", 2020, current_year)
        print(f"\nACCU price trend: {len(trend)} years")


def example_pagination():
    """Example 9: Pagination"""
    print("\n" + "=" * 60)
    print("Example 9: Pagination")
    print("=" * 60)
    
    from database import EcoAssistDatabase
    
    with EcoAssistDatabase() as db:
        # Get first page
        page = db.property.get_page(page=1, page_size=10)
        
        print(f"Page {page['page']} of {page['total_pages']}")
        print(f"Total items: {page['total_items']}")
        print(f"Page size: {page['page_size']}")
        print(f"Has next: {page['has_next']}")
        print(f"Has previous: {page['has_previous']}")
        
        print(f"\nProperties on this page:")
        for i, prop in enumerate(page['items'], 1):
            print(f"  {i}. {prop.name}")


def example_error_handling():
    """Example 10: Error handling"""
    print("\n" + "=" * 60)
    print("Example 10: Error Handling")
    print("=" * 60)
    
    from database import EcoAssistDatabase
    
    try:
        with EcoAssistDatabase() as db:
            # Try to get non-existent property
            property = db.property.get_by_id("NONEXISTENT")
            
            if property is None:
                print("Property not found (expected)")
            
            # Try invalid query
            try:
                results = db.property.execute_custom_query(
                    "SELECT * FROM nonexistent_table"
                )
            except Exception as e:
                print(f"Query error (expected): {type(e).__name__}")
            
            print("✓ Error handling working correctly")
    
    except Exception as e:
        print(f"Database error: {e}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("EcoAssist Database Package - Usage Examples")
    print("=" * 60)
    
    try:
        example_quick_setup()
        example_high_level_interface()
        example_property_operations()
        example_reduction_options()
        example_milestone_scenarios()
        example_dashboard_metrics()
        example_historical_data()
        example_pricing_data()
        example_pagination()
        example_error_handling()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        logger.exception("Example error")


if __name__ == "__main__":
    main()
