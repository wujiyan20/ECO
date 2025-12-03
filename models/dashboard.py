# database/repositories/dashboard.py - Dashboard Repository
"""
Repository for dashboard metrics, KPIs, and aggregated data
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from ..models import DashboardMetrics, PortfolioSummary
from .base import ReadOnlyRepository

logger = logging.getLogger(__name__)


class DashboardRepository:
    """Repository for dashboard data and metrics"""
    
    def __init__(self, db_manager):
        self.db = db_manager
    
    def get_portfolio_metrics(self, portfolio_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive portfolio metrics
        
        Args:
            portfolio_id: Optional portfolio filter
        
        Returns:
            Dictionary with portfolio metrics
        """
        where_clause = "is_active = 1"
        params = None
        
        if portfolio_id:
            where_clause += " AND portfolio_id = ?"
            params = (portfolio_id,)
        
        query = f"""
        SELECT 
            COUNT(*) as total_properties,
            SUM(baseline_emission) as total_emission,
            SUM(scope1_emission) as total_scope1,
            SUM(scope2_emission) as total_scope2,
            SUM(scope3_emission) as total_scope3,
            AVG(carbon_intensity) as avg_carbon_intensity,
            SUM(area_sqm) as total_area,
            SUM(annual_energy_cost) as total_energy_cost,
            AVG(baseline_emission) as avg_emission_per_property,
            MAX(baseline_emission) as max_emission,
            MIN(baseline_emission) as min_emission
        FROM properties
        WHERE {where_clause}
        """
        
        results = self.db.execute_query(query, params)
        return results[0] if results else {}
    
    def get_emission_kpis(self, portfolio_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get emission-related KPIs
        
        Args:
            portfolio_id: Optional portfolio filter
        
        Returns:
            Emission KPIs
        """
        metrics = self.get_portfolio_metrics(portfolio_id)
        
        # Calculate additional KPIs
        if metrics:
            total_emission = metrics.get('total_emission', 0)
            total_area = metrics.get('total_area', 0)
            
            kpis = {
                **metrics,
                'scope1_percentage': (metrics.get('total_scope1', 0) / total_emission * 100) if total_emission > 0 else 0,
                'scope2_percentage': (metrics.get('total_scope2', 0) / total_emission * 100) if total_emission > 0 else 0,
                'scope3_percentage': (metrics.get('total_scope3', 0) / total_emission * 100) if total_emission > 0 else 0,
                'portfolio_carbon_intensity': (total_emission / total_area) if total_area > 0 else 0
            }
            
            return kpis
        
        return {}
    
    def get_building_type_distribution(self, portfolio_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get distribution of properties by building type
        
        Args:
            portfolio_id: Optional portfolio filter
        
        Returns:
            List of building type counts
        """
        where_clause = "is_active = 1"
        params = None
        
        if portfolio_id:
            where_clause += " AND portfolio_id = ?"
            params = (portfolio_id,)
        
        query = f"""
        SELECT 
            building_type,
            COUNT(*) as count,
            SUM(baseline_emission) as total_emission,
            SUM(area_sqm) as total_area,
            AVG(carbon_intensity) as avg_carbon_intensity
        FROM properties
        WHERE {where_clause}
        GROUP BY building_type
        ORDER BY count DESC
        """
        
        return self.db.execute_query(query, params)
    
    def get_retrofit_potential_distribution(self, portfolio_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get distribution by retrofit potential
        
        Args:
            portfolio_id: Optional portfolio filter
        
        Returns:
            List of retrofit potential counts
        """
        where_clause = "is_active = 1"
        params = None
        
        if portfolio_id:
            where_clause += " AND portfolio_id = ?"
            params = (portfolio_id,)
        
        query = f"""
        SELECT 
            retrofit_potential,
            COUNT(*) as count,
            SUM(baseline_emission) as total_emission,
            AVG(baseline_emission) as avg_emission
        FROM properties
        WHERE {where_clause}
        GROUP BY retrofit_potential
        ORDER BY 
            CASE retrofit_potential
                WHEN 'Critical' THEN 1
                WHEN 'High' THEN 2
                WHEN 'Medium' THEN 3
                WHEN 'Low' THEN 4
                ELSE 5
            END
        """
        
        return self.db.execute_query(query, params)
    
    def get_regional_breakdown(self, portfolio_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get breakdown by region
        
        Args:
            portfolio_id: Optional portfolio filter
        
        Returns:
            List of regional metrics
        """
        where_clause = "is_active = 1"
        params = None
        
        if portfolio_id:
            where_clause += " AND portfolio_id = ?"
            params = (portfolio_id,)
        
        query = f"""
        SELECT 
            region,
            COUNT(*) as property_count,
            SUM(baseline_emission) as total_emission,
            SUM(area_sqm) as total_area,
            AVG(carbon_intensity) as avg_carbon_intensity,
            SUM(annual_energy_cost) as total_energy_cost
        FROM properties
        WHERE {where_clause} AND region IS NOT NULL
        GROUP BY region
        ORDER BY total_emission DESC
        """
        
        return self.db.execute_query(query, params)
    
    def get_top_emitters(self, limit: int = 10, portfolio_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get top emitting properties
        
        Args:
            limit: Number of properties to return
            portfolio_id: Optional portfolio filter
        
        Returns:
            List of top emitting properties
        """
        where_clause = "is_active = 1"
        params = []
        
        if portfolio_id:
            where_clause += " AND portfolio_id = ?"
            params.append(portfolio_id)
        
        query = f"""
        SELECT TOP {limit}
            property_id, name, building_type, retrofit_potential,
            baseline_emission, scope1_emission, scope2_emission, scope3_emission,
            carbon_intensity, area_sqm, region
        FROM properties
        WHERE {where_clause}
        ORDER BY baseline_emission DESC
        """
        
        return self.db.execute_query(query, tuple(params) if params else None)
    
    def get_reduction_opportunities(self, portfolio_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Identify reduction opportunities
        
        Args:
            portfolio_id: Optional portfolio filter
        
        Returns:
            Reduction opportunity analysis
        """
        where_clause = "is_active = 1"
        params = None
        
        if portfolio_id:
            where_clause += " AND portfolio_id = ?"
            params = (portfolio_id,)
        
        query = f"""
        SELECT 
            COUNT(CASE WHEN retrofit_potential = 'Critical' THEN 1 END) as critical_count,
            COUNT(CASE WHEN retrofit_potential = 'High' THEN 1 END) as high_count,
            COUNT(CASE WHEN retrofit_potential = 'Medium' THEN 1 END) as medium_count,
            SUM(CASE WHEN retrofit_potential IN ('Critical', 'High') THEN baseline_emission ELSE 0 END) as high_priority_emissions,
            SUM(baseline_emission) as total_emissions
        FROM properties
        WHERE {where_clause}
        """
        
        results = self.db.execute_query(query, params)
        if results:
            result = results[0]
            total = result.get('total_emissions', 0)
            high_priority = result.get('high_priority_emissions', 0)
            
            return {
                **result,
                'high_priority_percentage': (high_priority / total * 100) if total > 0 else 0
            }
        
        return {}
    
    def get_scenario_comparison(self, scenario_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Compare multiple scenarios
        
        Args:
            scenario_ids: List of scenario IDs
        
        Returns:
            Scenario comparison data
        """
        if not scenario_ids:
            return []
        
        placeholders = ','.join(['?'] * len(scenario_ids))
        query = f"""
        SELECT 
            scenario_id, name, target_year, reduction_percentage,
            baseline_emission, target_emission,
            total_capex, total_opex, strategy_type, sbt_aligned,
            reduction_rate_2030, reduction_rate_2040, reduction_rate_2050
        FROM milestone_scenarios
        WHERE scenario_id IN ({placeholders})
        ORDER BY reduction_percentage DESC
        """
        
        return self.db.execute_query(query, tuple(scenario_ids))
    
    def get_financial_summary(self, portfolio_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get financial summary
        
        Args:
            portfolio_id: Optional portfolio filter
        
        Returns:
            Financial metrics
        """
        where_clause = "is_active = 1"
        params = None
        
        if portfolio_id:
            where_clause += " AND portfolio_id = ?"
            params = (portfolio_id,)
        
        query = f"""
        SELECT 
            SUM(annual_energy_cost) as total_energy_cost,
            AVG(annual_energy_cost) as avg_energy_cost_per_property,
            SUM(annual_energy_cost / NULLIF(area_sqm, 0)) as energy_cost_per_sqm
        FROM properties
        WHERE {where_clause}
        """
        
        results = self.db.execute_query(query, params)
        return results[0] if results else {}
    
    def get_performance_trends(self, property_id: str, years: int = 3) -> Dict[str, Any]:
        """
        Get performance trends for a property
        
        Args:
            property_id: Property ID
            years: Number of years to analyze
        
        Returns:
            Performance trend data
        """
        current_year = datetime.now().year
        start_year = current_year - years
        
        query = """
        SELECT 
            year, 
            SUM(total_emission) as annual_emission,
            SUM(energy_consumption) as annual_consumption,
            AVG(carbon_intensity) as avg_carbon_intensity,
            SUM(energy_cost) as annual_cost
        FROM historical_performance
        WHERE property_id = ? AND year >= ?
        GROUP BY year
        ORDER BY year
        """
        
        return {
            'property_id': property_id,
            'trend_data': self.db.execute_query(query, (property_id, start_year))
        }
    
    def get_complete_dashboard_data(self, portfolio_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get complete dashboard data in one call
        
        Args:
            portfolio_id: Optional portfolio filter
        
        Returns:
            Complete dashboard data
        """
        return {
            'portfolio_metrics': self.get_portfolio_metrics(portfolio_id),
            'emission_kpis': self.get_emission_kpis(portfolio_id),
            'building_type_distribution': self.get_building_type_distribution(portfolio_id),
            'retrofit_distribution': self.get_retrofit_potential_distribution(portfolio_id),
            'regional_breakdown': self.get_regional_breakdown(portfolio_id),
            'top_emitters': self.get_top_emitters(10, portfolio_id),
            'reduction_opportunities': self.get_reduction_opportunities(portfolio_id),
            'financial_summary': self.get_financial_summary(portfolio_id),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_property_rankings(self, portfolio_id: Optional[str] = None, 
                            sort_by: str = 'emission', limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get property rankings by various metrics
        
        Args:
            portfolio_id: Optional portfolio filter
            sort_by: Sort metric ('emission', 'intensity', 'cost')
            limit: Number of properties to return
        
        Returns:
            Ranked list of properties
        """
        where_clause = "is_active = 1"
        params = []
        
        if portfolio_id:
            where_clause += " AND portfolio_id = ?"
            params.append(portfolio_id)
        
        order_by = {
            'emission': 'baseline_emission DESC',
            'intensity': 'carbon_intensity DESC',
            'cost': 'annual_energy_cost DESC',
            'area': 'area_sqm DESC'
        }.get(sort_by, 'baseline_emission DESC')
        
        query = f"""
        SELECT TOP {limit}
            property_id, name, building_type, region,
            baseline_emission, carbon_intensity, area_sqm,
            annual_energy_cost, retrofit_potential,
            RANK() OVER (ORDER BY {order_by}) as rank
        FROM properties
        WHERE {where_clause}
        ORDER BY {order_by}
        """
        
        return self.db.execute_query(query, tuple(params) if params else None)
    
    def get_target_progress(self, scenario_id: str) -> Dict[str, Any]:
        """
        Get progress towards scenario targets
        
        Args:
            scenario_id: Scenario ID
        
        Returns:
            Progress metrics
        """
        query = """
        SELECT 
            COUNT(*) as total_properties,
            SUM(baseline_emission) as total_baseline,
            SUM(target_emission) as total_target,
            AVG(reduction_percentage) as avg_reduction,
            SUM(allocated_budget) as total_budget,
            COUNT(CASE WHEN implementation_priority <= 2 THEN 1 END) as high_priority_count
        FROM property_targets
        WHERE scenario_id = ?
        """
        
        results = self.db.execute_query(query, (scenario_id,))
        if results:
            result = results[0]
            baseline = result.get('total_baseline', 0)
            target = result.get('total_target', 0)
            
            return {
                **result,
                'total_reduction_required': baseline - target,
                'progress_percentage': 0  # Would need actual performance data
            }
        
        return {}
