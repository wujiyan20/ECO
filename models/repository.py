# models/repository.py - Data Access Layer (Repository Pattern)
"""
Repository pattern implementation for SQL Server data access.
Provides CRUD operations and queries for all model types.
Integrates with the DatabaseManager from database/manager.py.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import json

from .property import Property, PropertyFilter, Portfolio
from .emission import BaselineDataRecord, EmissionFactor, EmissionCalculation
from .milestone import MilestoneScenario, MilestoneTarget, MilestoneProgress
from .reduction import ReductionOption, ReductionStrategy, ImplementationPlan
from .cost import CostProjection, CostSchedule
from .enums import *

logger = logging.getLogger(__name__)

# =============================================================================
# BASE REPOSITORY
# =============================================================================

class BaseRepository:
    """
    Base repository with common database operations
    """
    
    def __init__(self, db_manager):
        """
        Initialize repository with database manager
        
        Args:
            db_manager: DatabaseManager instance from database/manager.py
        """
        self.db = db_manager
    
    def _dict_to_model(self, data: Dict, model_class):
        """Convert dictionary to model instance"""
        try:
            return model_class(**data)
        except Exception as e:
            logger.error(f"Error converting dict to {model_class.__name__}: {e}")
            return None

# =============================================================================
# PROPERTY REPOSITORY
# =============================================================================

class PropertyRepository(BaseRepository):
    """Repository for Property data access"""
    
    def get_all(self, filter_obj: Optional[PropertyFilter] = None) -> List[Property]:
        """
        Get all properties with optional filtering
        """
        query = """
        SELECT p.*, 
               pe.scope1_emission, pe.scope2_emission, pe.scope3_emission,
               pe.baseline_emission, pe.emission_intensity
        FROM properties p
        LEFT JOIN property_emissions pe ON p.property_id = pe.property_id
        WHERE 1=1
        """
        params = []
        
        if filter_obj:
            if filter_obj.property_ids:
                placeholders = ','.join('?' * len(filter_obj.property_ids))
                query += f" AND p.property_id IN ({placeholders})"
                params.extend(filter_obj.property_ids)
            
            if filter_obj.building_types:
                placeholders = ','.join('?' * len(filter_obj.building_types))
                query += f" AND p.building_type IN ({placeholders})"
                params.extend([bt.value for bt in filter_obj.building_types])
            
            if filter_obj.min_area:
                query += " AND p.area_sqm >= ?"
                params.append(filter_obj.min_area)
            
            if filter_obj.max_area:
                query += " AND p.area_sqm <= ?"
                params.append(filter_obj.max_area)
            
            if filter_obj.cities:
                placeholders = ','.join('?' * len(filter_obj.cities))
                query += f" AND p.city IN ({placeholders})"
                params.extend(filter_obj.cities)
            
            if filter_obj.status:
                query += " AND p.status = ?"
                params.append(filter_obj.status.value)
        
        results = self.db.execute_query(query, tuple(params) if params else None)
        return [Property(**row) for row in results]
    
    def get_by_id(self, property_id: str) -> Optional[Property]:
        """Get property by ID"""
        query = """
        SELECT p.*, 
               pe.scope1_emission, pe.scope2_emission, pe.scope3_emission,
               pe.baseline_emission, pe.emission_intensity
        FROM properties p
        LEFT JOIN property_emissions pe ON p.property_id = pe.property_id
        WHERE p.property_id = ?
        """
        results = self.db.execute_query(query, (property_id,))
        return Property(**results[0]) if results else None
    
    def create(self, property_obj: Property) -> str:
        """Create new property"""
        query = """
        INSERT INTO properties (
            property_id, name, address, city, state, postal_code, country,
            area_sqm, gross_floor_area, building_type, year_built,
            status, occupancy_rate, retrofit_potential,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            property_obj.property_id,
            property_obj.name,
            property_obj.address,
            property_obj.city,
            property_obj.state,
            property_obj.postal_code,
            property_obj.country,
            property_obj.area_sqm,
            property_obj.gross_floor_area,
            property_obj.building_type.value,
            property_obj.year_built,
            property_obj.status.value,
            property_obj.occupancy_rate,
            property_obj.retrofit_potential.value,
            property_obj.created_at,
            property_obj.updated_at
        )
        
        self.db.execute_non_query(query, params)
        logger.info(f"Created property: {property_obj.property_id}")
        return property_obj.property_id
    
    def update(self, property_obj: Property) -> bool:
        """Update existing property"""
        query = """
        UPDATE properties SET
            name = ?, address = ?, city = ?, state = ?,
            area_sqm = ?, building_type = ?, status = ?,
            occupancy_rate = ?, retrofit_potential = ?,
            updated_at = ?
        WHERE property_id = ?
        """
        params = (
            property_obj.name,
            property_obj.address,
            property_obj.city,
            property_obj.state,
            property_obj.area_sqm,
            property_obj.building_type.value,
            property_obj.status.value,
            property_obj.occupancy_rate,
            property_obj.retrofit_potential.value,
            datetime.now(),
            property_obj.property_id
        )
        
        rows = self.db.execute_non_query(query, params)
        return rows > 0
    
    def delete(self, property_id: str) -> bool:
        """Delete property (soft delete)"""
        query = "UPDATE properties SET is_deleted = 1, deleted_at = ? WHERE property_id = ?"
        rows = self.db.execute_non_query(query, (datetime.now(), property_id))
        return rows > 0
    
    def get_portfolio_metrics(self, property_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate portfolio-level metrics"""
        query = """
        SELECT 
            COUNT(*) as property_count,
            SUM(area_sqm) as total_area,
            AVG(emission_intensity) as avg_emission_intensity,
            SUM(baseline_emission) as total_emissions
        FROM properties p
        LEFT JOIN property_emissions pe ON p.property_id = pe.property_id
        WHERE p.is_deleted = 0
        """
        params = None
        
        if property_ids:
            placeholders = ','.join('?' * len(property_ids))
            query += f" AND p.property_id IN ({placeholders})"
            params = tuple(property_ids)
        
        results = self.db.execute_query(query, params)
        return results[0] if results else {}

# =============================================================================
# EMISSION REPOSITORY
# =============================================================================

class EmissionRepository(BaseRepository):
    """Repository for Emission and Baseline data"""
    
    def get_baseline_data(self, property_id: str, start_year: Optional[int] = None, 
                         end_year: Optional[int] = None) -> List[BaselineDataRecord]:
        """Get baseline emission data for a property"""
        query = """
        SELECT * FROM baseline_emissions
        WHERE property_id = ?
        """
        params = [property_id]
        
        if start_year:
            query += " AND year >= ?"
            params.append(start_year)
        
        if end_year:
            query += " AND year <= ?"
            params.append(end_year)
        
        query += " ORDER BY year"
        
        results = self.db.execute_query(query, tuple(params))
        return [BaselineDataRecord(**row) for row in results]
    
    def create_baseline_record(self, baseline: BaselineDataRecord) -> str:
        """Create new baseline emission record"""
        query = """
        INSERT INTO baseline_emissions (
            id, year, property_id, scope1_emissions, scope2_emissions, scope3_emissions,
            total_emissions, total_consumption, total_cost, unit, data_quality,
            measurement_method, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            baseline.id,
            baseline.year,
            baseline.property_id,
            baseline.scope1_emissions,
            baseline.scope2_emissions,
            baseline.scope3_emissions or 0,
            baseline.total_emissions,
            baseline.total_consumption,
            baseline.total_cost,
            baseline.unit.value,
            baseline.data_quality.value,
            baseline.measurement_method.value,
            baseline.created_at,
            baseline.updated_at
        )
        
        self.db.execute_non_query(query, params)
        return baseline.id
    
    def get_emission_factors(self, fuel_type: Optional[FuelType] = None, 
                           year: Optional[int] = None) -> List[EmissionFactor]:
        """Get emission factors"""
        query = "SELECT * FROM emission_factors WHERE 1=1"
        params = []
        
        if fuel_type:
            query += " AND fuel_type = ?"
            params.append(fuel_type.value)
        
        if year:
            query += " AND valid_from_year <= ? AND (valid_to_year IS NULL OR valid_to_year >= ?)"
            params.extend([year, year])
        
        results = self.db.execute_query(query, tuple(params) if params else None)
        return [EmissionFactor(**row) for row in results]

# =============================================================================
# MILESTONE REPOSITORY
# =============================================================================

class MilestoneRepository(BaseRepository):
    """Repository for Milestone and Scenario data"""
    
    def get_all_scenarios(self, property_ids: Optional[List[str]] = None) -> List[MilestoneScenario]:
        """Get all milestone scenarios"""
        query = "SELECT * FROM milestone_scenarios WHERE is_deleted = 0"
        params = None
        
        if property_ids:
            # Need to join with scenario_properties table
            query = """
            SELECT DISTINCT ms.* FROM milestone_scenarios ms
            JOIN scenario_properties sp ON ms.scenario_id = sp.scenario_id
            WHERE ms.is_deleted = 0 AND sp.property_id IN ({})
            """.format(','.join('?' * len(property_ids)))
            params = tuple(property_ids)
        
        results = self.db.execute_query(query, params)
        scenarios = []
        
        for row in results:
            scenario = MilestoneScenario(**row)
            # Load yearly targets
            scenario.yearly_targets = self._load_yearly_targets(scenario.scenario_id)
            scenarios.append(scenario)
        
        return scenarios
    
    def get_by_id(self, scenario_id: str) -> Optional[MilestoneScenario]:
        """Get scenario by ID"""
        query = "SELECT * FROM milestone_scenarios WHERE scenario_id = ?"
        results = self.db.execute_query(query, (scenario_id,))
        
        if not results:
            return None
        
        scenario = MilestoneScenario(**results[0])
        scenario.yearly_targets = self._load_yearly_targets(scenario_id)
        return scenario
    
    def create(self, scenario: MilestoneScenario) -> str:
        """Create new milestone scenario"""
        query = """
        INSERT INTO milestone_scenarios (
            scenario_id, scenario_name, description, scenario_type,
            base_year, mid_term_year, long_term_year,
            baseline_emission, mid_term_target, long_term_target,
            mid_term_reduction_percentage, long_term_reduction_percentage,
            total_capex, total_opex, cumulative_cost,
            feasibility_score, risk_score, overall_score,
            approval_status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            scenario.scenario_id,
            scenario.scenario_name,
            scenario.description,
            scenario.scenario_type.value,
            scenario.base_year,
            scenario.mid_term_year,
            scenario.long_term_year,
            scenario.baseline_emission,
            scenario.mid_term_target,
            scenario.long_term_target,
            scenario.mid_term_reduction_percentage,
            scenario.long_term_reduction_percentage,
            scenario.total_capex,
            scenario.total_opex,
            scenario.cumulative_cost,
            scenario.feasibility_score,
            scenario.risk_score,
            scenario.overall_score,
            scenario.approval_status.value,
            scenario.created_at,
            scenario.updated_at
        )
        
        self.db.execute_non_query(query, params)
        
        # Save yearly targets
        self._save_yearly_targets(scenario.scenario_id, scenario.yearly_targets)
        
        # Save property associations
        if scenario.property_ids:
            self._save_scenario_properties(scenario.scenario_id, scenario.property_ids)
        
        return scenario.scenario_id
    
    def _load_yearly_targets(self, scenario_id: str) -> Dict[int, float]:
        """Load yearly targets for a scenario"""
        query = """
        SELECT year, target_emission
        FROM scenario_yearly_targets
        WHERE scenario_id = ?
        ORDER BY year
        """
        results = self.db.execute_query(query, (scenario_id,))
        return {row['year']: row['target_emission'] for row in results}
    
    def _save_yearly_targets(self, scenario_id: str, yearly_targets: Dict[int, float]):
        """Save yearly targets for a scenario"""
        if not yearly_targets:
            return
        
        query = """
        INSERT INTO scenario_yearly_targets (scenario_id, year, target_emission)
        VALUES (?, ?, ?)
        """
        params_list = [
            (scenario_id, year, target)
            for year, target in yearly_targets.items()
        ]
        
        self.db.execute_many(query, params_list)
    
    def _save_scenario_properties(self, scenario_id: str, property_ids: List[str]):
        """Save property associations for scenario"""
        query = "INSERT INTO scenario_properties (scenario_id, property_id) VALUES (?, ?)"
        params_list = [(scenario_id, prop_id) for prop_id in property_ids]
        self.db.execute_many(query, params_list)

# =============================================================================
# REDUCTION OPTION REPOSITORY
# =============================================================================

class ReductionOptionRepository(BaseRepository):
    """Repository for Reduction Options"""
    
    def get_all(self, property_id: Optional[str] = None,
                strategy_type: Optional[StrategyType] = None) -> List[ReductionOption]:
        """Get all reduction options"""
        query = "SELECT * FROM reduction_options WHERE is_deleted = 0"
        params = []
        
        if property_id:
            query += " AND property_id = ?"
            params.append(property_id)
        
        if strategy_type:
            query += " AND strategy_type = ?"
            params.append(strategy_type.value)
        
        results = self.db.execute_query(query, tuple(params) if params else None)
        return [ReductionOption(**row) for row in results]
    
    def create(self, option: ReductionOption) -> str:
        """Create new reduction option"""
        query = """
        INSERT INTO reduction_options (
            option_id, name, description, strategy_type,
            annual_co2_reduction, capex, annual_opex, annual_savings,
            implementation_time_months, expected_lifetime_years,
            priority, risk_level, simple_payback_years,
            cost_per_tonne_co2, status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            option.option_id,
            option.name,
            option.description,
            option.strategy_type.value,
            option.annual_co2_reduction,
            option.capex,
            option.annual_opex,
            option.annual_savings,
            option.implementation_time_months,
            option.expected_lifetime_years,
            option.priority.value,
            option.risk_level.value,
            option.simple_payback_years,
            option.cost_per_tonne_co2,
            option.status.value,
            option.created_at,
            option.updated_at
        )
        
        self.db.execute_non_query(query, params)
        return option.option_id

# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'BaseRepository',
    'PropertyRepository',
    'EmissionRepository',
    'MilestoneRepository',
    'ReductionOptionRepository'
]
