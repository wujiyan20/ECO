"""
Database Manager for EcoAssist
Handles all database operations for Modules 1-4
"""
import pyodbc
import logging
import json
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date
from contextlib import contextmanager
from uuid import UUID, uuid4

from .database_config import DatabaseConfig, get_config


logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Central database manager for EcoAssist
    Handles connections and CRUD operations for all modules
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database manager
        
        Args:
            config: DatabaseConfig instance (uses development config if None)
        """
        self.config = config or get_config('development')
        self.connection_string = self.config.get_connection_string()
        logger.info(f"DatabaseManager initialized for server: {self.config.server}, database: {self.config.database}")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections
        Ensures proper connection cleanup
        """
        conn = None
        try:
            conn = pyodbc.connect(
                self.connection_string,
                timeout=self.config.connection_timeout
            )
            conn.timeout = self.config.command_timeout
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[pyodbc.Row]:
        """
        Execute SELECT query and return results
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
        
        Returns:
            List of result rows
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return results
    
    def execute_non_query(self, query: str, params: Optional[Tuple] = None) -> int:
        """
        Execute INSERT/UPDATE/DELETE query
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
        
        Returns:
            Number of affected rows
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            affected = cursor.rowcount
            cursor.close()
            return affected
    
    def execute_scalar(self, query: str, params: Optional[Tuple] = None) -> Any:
        """
        Execute query and return single value
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
        
        Returns:
            Single scalar value
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else None
    
    # =========================================================================
    # MODULE 1: MILESTONE SCENARIOS
    # =========================================================================
    
    def create_scenario(self, scenario_data: Dict[str, Any]) -> str:
        """
        Create a new milestone scenario
        
        Args:
            scenario_data: Dictionary with scenario details
        
        Returns:
            scenario_id (UUID string)
        """
        scenario_id = str(uuid4())
        
        query = """
        INSERT INTO milestone_scenarios (
            scenario_id, user_id, scenario_name, baseline_year, target_year,
            baseline_emission, target_reduction_percentage, scenario_type,
            description, target_emission, reduction_required, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            scenario_id,
            scenario_data.get('user_id'),
            scenario_data['scenario_name'],
            scenario_data['baseline_year'],
            scenario_data['target_year'],
            scenario_data['baseline_emission'],
            scenario_data['target_reduction_percentage'],
            scenario_data.get('scenario_type', 'balanced'),
            scenario_data.get('description'),
            scenario_data['target_emission'],
            scenario_data['reduction_required'],
            scenario_data.get('status', 'draft')
        )
        
        self.execute_non_query(query, params)
        logger.info(f"Created scenario: {scenario_id}")
        return scenario_id
    
    def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Get scenario by ID"""
        query = """
        SELECT scenario_id, user_id, scenario_name, baseline_year, target_year,
               baseline_emission, target_reduction_percentage, scenario_type,
               description, target_emission, reduction_required, status,
               created_at, updated_at
        FROM milestone_scenarios
        WHERE scenario_id = ?
        """
        
        results = self.execute_query(query, (scenario_id,))
        if not results:
            return None
        
        row = results[0]
        return {
            'scenario_id': str(row.scenario_id),
            'user_id': str(row.user_id) if row.user_id else None,
            'scenario_name': row.scenario_name,
            'baseline_year': row.baseline_year,
            'target_year': row.target_year,
            'baseline_emission': float(row.baseline_emission),
            'target_reduction_percentage': float(row.target_reduction_percentage),
            'scenario_type': row.scenario_type,
            'description': row.description,
            'target_emission': float(row.target_emission),
            'reduction_required': float(row.reduction_required),
            'status': row.status,
            'created_at': row.created_at.isoformat() if row.created_at else None,
            'updated_at': row.updated_at.isoformat() if row.updated_at else None
        }
    
    def update_scenario_status(self, scenario_id: str, status: str, registered_by: Optional[str] = None) -> bool:
        """Update scenario status"""
        if status == 'registered' and registered_by:
            query = """
            UPDATE milestone_scenarios
            SET status = ?, registered_at = GETDATE(), registered_by = ?
            WHERE scenario_id = ?
            """
            params = (status, registered_by, scenario_id)
        else:
            query = """
            UPDATE milestone_scenarios
            SET status = ?
            WHERE scenario_id = ?
            """
            params = (status, scenario_id)
        
        affected = self.execute_non_query(query, params)
        return affected > 0
    
    def create_scenario_milestones(self, scenario_id: str, milestones: List[Dict[str, Any]]) -> int:
        """Create milestone records for a scenario"""
        query = """
        INSERT INTO scenario_milestones (
            milestone_id, scenario_id, year, target_emission,
            reduction_from_baseline, reduction_percentage,
            cumulative_reduction, annual_reduction
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        count = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for milestone in milestones:
                params = (
                    str(uuid4()),
                    scenario_id,
                    milestone['year'],
                    milestone['target_emission'],
                    milestone['reduction_from_baseline'],
                    milestone['reduction_percentage'],
                    milestone['cumulative_reduction'],
                    milestone['annual_reduction']
                )
                cursor.execute(query, params)
                count += 1
            cursor.close()
        
        logger.info(f"Created {count} milestones for scenario {scenario_id}")
        return count
    
    def get_scenario_milestones(self, scenario_id: str) -> List[Dict[str, Any]]:
        """Get all milestones for a scenario"""
        query = """
        SELECT year, target_emission, reduction_from_baseline,
               reduction_percentage, cumulative_reduction, annual_reduction
        FROM scenario_milestones
        WHERE scenario_id = ?
        ORDER BY year
        """
        
        results = self.execute_query(query, (scenario_id,))
        return [
            {
                'year': row.year,
                'target_emission': float(row.target_emission),
                'reduction_from_baseline': float(row.reduction_from_baseline),
                'reduction_percentage': float(row.reduction_percentage),
                'cumulative_reduction': float(row.cumulative_reduction),
                'annual_reduction': float(row.annual_reduction)
            }
            for row in results
        ]
    
    def update_scenario_status(
        self, 
        scenario_id: str, 
        status: str,
        registered_by: str = None
    ) -> bool:
        """
        Update milestone scenario registration status
        
        Args:
            scenario_id: Scenario UUID
            status: New status ('registered', 'approved', etc.)
            registered_by: User ID who registered it
        
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            query = """
                UPDATE milestone_scenarios
                SET 
                    status = ?,
                    registered_at = GETDATE(),
                    registered_by = ?,
                    updated_at = GETDATE()
                WHERE scenario_id = ?
            """
            
            # Convert registered_by to UUID if provided
            registered_by_uuid = None
            if registered_by:
                try:
                    registered_by_uuid = uuid.UUID(registered_by) if isinstance(registered_by, str) else registered_by
                except:
                    registered_by_uuid = None
            
            self.execute_non_query(query, (status, registered_by_uuid, scenario_id))
            
            logger.info(f"Updated scenario {scenario_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update scenario status: {e}")
            return False
    
    # =========================================================================
    # MODULE 2: TARGET ALLOCATIONS
    # =========================================================================
    
    def create_allocation(self, allocation_data: Dict[str, Any]) -> str:
        """Create a new target allocation"""
        allocation_id = str(uuid4())
        
        query = """
        INSERT INTO target_allocations (
            allocation_id, scenario_id, user_id, allocation_name,
            allocation_strategy, optimization_method,
            total_properties, total_baseline_emission, total_target_reduction,
            fairness_score, feasibility_score, optimization_score, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            allocation_id,
            allocation_data['scenario_id'],
            allocation_data.get('user_id'),
            allocation_data['allocation_name'],
            allocation_data.get('allocation_strategy', 'proportional'),
            allocation_data.get('optimization_method'),
            allocation_data.get('total_properties', 0),
            allocation_data['total_baseline_emission'],
            allocation_data['total_target_reduction'],
            allocation_data.get('fairness_score'),
            allocation_data.get('feasibility_score'),
            allocation_data.get('optimization_score'),
            allocation_data.get('status', 'draft')
        )
        
        self.execute_non_query(query, params)
        logger.info(f"Created allocation: {allocation_id}")
        return allocation_id
    
    def get_allocation(self, allocation_id: str) -> Optional[Dict[str, Any]]:
        """Get allocation by ID"""
        query = """
        SELECT allocation_id, scenario_id, user_id, allocation_name,
               allocation_strategy, optimization_method,
               total_properties, total_baseline_emission, total_target_reduction,
               fairness_score, feasibility_score, optimization_score, status,
               created_at, updated_at
        FROM target_allocations
        WHERE allocation_id = ?
        """
        
        results = self.execute_query(query, (allocation_id,))
        if not results:
            return None
        
        row = results[0]
        return {
            'allocation_id': str(row.allocation_id),
            'scenario_id': str(row.scenario_id),
            'user_id': str(row.user_id) if row.user_id else None,
            'allocation_name': row.allocation_name,
            'allocation_strategy': row.allocation_strategy,
            'optimization_method': row.optimization_method,
            'total_properties': row.total_properties,
            'total_baseline_emission': float(row.total_baseline_emission),
            'total_target_reduction': float(row.total_target_reduction),
            'fairness_score': float(row.fairness_score) if row.fairness_score else None,
            'feasibility_score': float(row.feasibility_score) if row.feasibility_score else None,
            'optimization_score': float(row.optimization_score) if row.optimization_score else None,
            'status': row.status,
            'created_at': row.created_at.isoformat() if row.created_at else None,
            'updated_at': row.updated_at.isoformat() if row.updated_at else None
        }
    
    def create_property_targets(self, allocation_id: str, property_targets: List[Dict[str, Any]]) -> int:
        """Create property target records"""
        query = """
        INSERT INTO property_targets (
            property_target_id, allocation_id, property_id, property_name,
            building_type, area_sqm, baseline_emission, baseline_intensity,
            target_emission, reduction_target, reduction_percentage,
            allocated_budget, priority_level, feasibility_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        count = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for target in property_targets:
                params = (
                    str(uuid4()),
                    allocation_id,
                    target['property_id'],
                    target.get('property_name'),
                    target.get('building_type'),
                    target.get('area_sqm'),
                    target['baseline_emission'],
                    target.get('baseline_intensity'),
                    target['target_emission'],
                    target['reduction_target'],
                    target['reduction_percentage'],
                    target.get('allocated_budget'),
                    target.get('priority_level'),
                    target.get('feasibility_score')
                )
                cursor.execute(query, params)
                count += 1
            cursor.close()
        
        logger.info(f"Created {count} property targets for allocation {allocation_id}")
        return count
    
    def get_property_targets(self, allocation_id: str) -> List[Dict[str, Any]]:
        """Get all property targets for an allocation"""
        query = """
        SELECT property_id, property_name, building_type, area_sqm,
               baseline_emission, baseline_intensity,
               target_emission, reduction_target, reduction_percentage,
               allocated_budget, priority_level, feasibility_score
        FROM property_targets
        WHERE allocation_id = ?
        ORDER BY priority_level, property_id
        """
        
        results = self.execute_query(query, (allocation_id,))
        return [
            {
                'property_id': row.property_id,
                'property_name': row.property_name,
                'building_type': row.building_type,
                'area_sqm': float(row.area_sqm) if row.area_sqm else None,
                'baseline_emission': float(row.baseline_emission),
                'baseline_intensity': float(row.baseline_intensity) if row.baseline_intensity else None,
                'target_emission': float(row.target_emission),
                'reduction_target': float(row.reduction_target),
                'reduction_percentage': float(row.reduction_percentage),
                'allocated_budget': float(row.allocated_budget) if row.allocated_budget else None,
                'priority_level': row.priority_level,
                'feasibility_score': float(row.feasibility_score) if row.feasibility_score else None
            }
            for row in results
        ]
    
    # =========================================================================
    # MODULE 3: LONG-TERM PLANS
    # =========================================================================
    
    def create_plan(self, plan_data: Dict[str, Any]) -> str:
        """Create a new long-term plan"""
        plan_id = str(uuid4())
        
        query = """
        INSERT INTO long_term_plans (
            plan_id, scenario_id, allocation_id, user_id, plan_name,
            pattern_type, start_year, end_year, planning_horizon_years,
            total_investment, total_capex, total_opex,
            total_reduction_target, overall_roi_years, breakeven_year,
            total_actions, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            plan_id,
            plan_data['scenario_id'],
            plan_data['allocation_id'],
            plan_data.get('user_id'),
            plan_data['plan_name'],
            plan_data.get('pattern_type', 'balanced'),
            plan_data['start_year'],
            plan_data['end_year'],
            plan_data['planning_horizon_years'],
            plan_data['total_investment'],
            plan_data['total_capex'],
            plan_data['total_opex'],
            plan_data['total_reduction_target'],
            plan_data.get('overall_roi_years'),
            plan_data.get('breakeven_year'),
            plan_data.get('total_actions', 0),
            plan_data.get('status', 'draft')
        )
        
        self.execute_non_query(query, params)
        logger.info(f"Created plan: {plan_id}")
        return plan_id
    
    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get plan by ID"""
        query = """
        SELECT plan_id, scenario_id, allocation_id, user_id, plan_name,
               pattern_type, start_year, end_year, planning_horizon_years,
               total_investment, total_capex, total_opex,
               total_reduction_target, overall_roi_years, breakeven_year,
               total_actions, status, approved_by, approval_date,
               created_at, updated_at
        FROM long_term_plans
        WHERE plan_id = ?
        """
        
        results = self.execute_query(query, (plan_id,))
        if not results:
            return None
        
        row = results[0]
        return {
            'plan_id': str(row.plan_id),
            'scenario_id': str(row.scenario_id),
            'allocation_id': str(row.allocation_id),
            'user_id': str(row.user_id) if row.user_id else None,
            'plan_name': row.plan_name,
            'pattern_type': row.pattern_type,
            'start_year': row.start_year,
            'end_year': row.end_year,
            'planning_horizon_years': row.planning_horizon_years,
            'total_investment': float(row.total_investment),
            'total_capex': float(row.total_capex),
            'total_opex': float(row.total_opex),
            'total_reduction_target': float(row.total_reduction_target),
            'overall_roi_years': float(row.overall_roi_years) if row.overall_roi_years else None,
            'breakeven_year': row.breakeven_year,
            'total_actions': row.total_actions,
            'status': row.status,
            'approved_by': str(row.approved_by) if row.approved_by else None,
            'approval_date': row.approval_date.isoformat() if row.approval_date else None,
            'created_at': row.created_at.isoformat() if row.created_at else None,
            'updated_at': row.updated_at.isoformat() if row.updated_at else None
        }
    
    def update_plan_status(self, plan_id: str, status: str, approved_by: Optional[str] = None) -> bool:
        """Update plan status"""
        if status == 'registered' and approved_by:
            query = """
            UPDATE long_term_plans
            SET status = ?, registered_at = GETDATE(), approved_by = ?, approval_date = CAST(GETDATE() AS DATE)
            WHERE plan_id = ?
            """
            params = (status, approved_by, plan_id)
        else:
            query = """
            UPDATE long_term_plans
            SET status = ?
            WHERE plan_id = ?
            """
            params = (status, plan_id)
        
        affected = self.execute_non_query(query, params)
        return affected > 0
    
    def create_annual_plans(self, plan_id: str, annual_plans: List[Dict[str, Any]]) -> int:
        """Create annual action plan records"""
        query = """
        INSERT INTO annual_action_plans (
            annual_plan_id, plan_id, year, total_investment, capex, opex,
            target_reduction, action_count, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        count = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for annual_plan in annual_plans:
                params = (
                    str(uuid4()),
                    plan_id,
                    annual_plan['year'],
                    annual_plan['total_investment'],
                    annual_plan['capex'],
                    annual_plan['opex'],
                    annual_plan['target_reduction'],
                    annual_plan.get('action_count', 0),
                    annual_plan.get('status', 'planned')
                )
                cursor.execute(query, params)
                count += 1
            cursor.close()
        
        logger.info(f"Created {count} annual plans for plan {plan_id}")
        return count
    
    def get_annual_plans(self, plan_id: str) -> List[Dict[str, Any]]:
        """Get all annual plans for a long-term plan"""
        query = """
        SELECT year, total_investment, capex, opex, target_reduction,
               action_count, status
        FROM annual_action_plans
        WHERE plan_id = ?
        ORDER BY year
        """
        
        results = self.execute_query(query, (plan_id,))
        return [
            {
                'year': row.year,
                'total_investment': float(row.total_investment),
                'capex': float(row.capex),
                'opex': float(row.opex),
                'target_reduction': float(row.target_reduction),
                'action_count': row.action_count,
                'status': row.status
            }
            for row in results
        ]
    
    # =========================================================================
    # MODULE 4: REOPTIMIZATIONS
    # =========================================================================
    
    def create_reoptimization(self, reopt_data: Dict[str, Any]) -> str:
        """Create a new reoptimization record"""
        reopt_id = str(uuid4())
        
        query = """
        INSERT INTO reoptimizations (
            reoptimization_id, plan_id, user_id, target_year,
            analysis_start_date, analysis_end_date, frequency,
            planned_emission, actual_emission, variance, variance_percentage,
            planned_cost, actual_cost, cost_variance, cost_variance_percentage,
            performance_status, performance_trend, pattern_type, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            reopt_id,
            reopt_data['plan_id'],
            reopt_data.get('user_id'),
            reopt_data['target_year'],
            reopt_data['analysis_start_date'],
            reopt_data['analysis_end_date'],
            reopt_data.get('frequency', 'quarterly'),
            reopt_data['planned_emission'],
            reopt_data['actual_emission'],
            reopt_data['variance'],
            reopt_data['variance_percentage'],
            reopt_data.get('planned_cost'),
            reopt_data.get('actual_cost'),
            reopt_data.get('cost_variance'),
            reopt_data.get('cost_variance_percentage'),
            reopt_data['performance_status'],
            reopt_data.get('performance_trend'),
            reopt_data['pattern_type'],
            reopt_data.get('status', 'draft')
        )
        
        self.execute_non_query(query, params)
        logger.info(f"Created reoptimization: {reopt_id}")
        return reopt_id
    
    def get_reoptimization(self, reopt_id: str) -> Optional[Dict[str, Any]]:
        """Get reoptimization by ID"""
        query = """
        SELECT reoptimization_id, plan_id, user_id, target_year,
               analysis_start_date, analysis_end_date, frequency,
               planned_emission, actual_emission, variance, variance_percentage,
               planned_cost, actual_cost, cost_variance, cost_variance_percentage,
               performance_status, performance_trend, pattern_type, status,
               approved_by, approval_date, created_at, updated_at
        FROM reoptimizations
        WHERE reoptimization_id = ?
        """
        
        results = self.execute_query(query, (reopt_id,))
        if not results:
            return None
        
        row = results[0]
        return {
            'reoptimization_id': str(row.reoptimization_id),
            'plan_id': str(row.plan_id),
            'user_id': str(row.user_id) if row.user_id else None,
            'target_year': row.target_year,
            'analysis_start_date': row.analysis_start_date.isoformat() if row.analysis_start_date else None,
            'analysis_end_date': row.analysis_end_date.isoformat() if row.analysis_end_date else None,
            'frequency': row.frequency,
            'planned_emission': float(row.planned_emission),
            'actual_emission': float(row.actual_emission),
            'variance': float(row.variance),
            'variance_percentage': float(row.variance_percentage),
            'planned_cost': float(row.planned_cost) if row.planned_cost else None,
            'actual_cost': float(row.actual_cost) if row.actual_cost else None,
            'cost_variance': float(row.cost_variance) if row.cost_variance else None,
            'cost_variance_percentage': float(row.cost_variance_percentage) if row.cost_variance_percentage else None,
            'performance_status': row.performance_status,
            'performance_trend': row.performance_trend,
            'pattern_type': row.pattern_type,
            'status': row.status,
            'approved_by': str(row.approved_by) if row.approved_by else None,
            'approval_date': row.approval_date.isoformat() if row.approval_date else None,
            'created_at': row.created_at.isoformat() if row.created_at else None,
            'updated_at': row.updated_at.isoformat() if row.updated_at else None
        }
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT @@VERSION")
                version = cursor.fetchone()[0]
                cursor.close()
                logger.info(f"Database connection successful: {version[:50]}")
                return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def get_table_count(self, table_name: str) -> int:
        """Get record count for a table"""
        query = f"SELECT COUNT(*) FROM {table_name}"
        return self.execute_scalar(query) or 0
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check
        Returns status and table counts
        """
        try:
            connected = self.test_connection()
            
            if not connected:
                return {
                    'status': 'error',
                    'connected': False,
                    'message': 'Cannot connect to database'
                }
            
            tables = [
                'users', 'milestone_scenarios', 'scenario_milestones',
                'target_allocations', 'property_targets',
                'long_term_plans', 'annual_action_plans', 'plan_actions',
                'reoptimizations', 'performance_tracking'
            ]
            
            table_counts = {}
            for table in tables:
                try:
                    table_counts[table] = self.get_table_count(table)
                except:
                    table_counts[table] = -1  # Error getting count
            
            return {
                'status': 'healthy',
                'connected': True,
                'server': self.config.server,
                'database': self.config.database,
                'table_counts': table_counts
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'connected': False,
                'message': str(e)
            }


# Convenience function for getting database manager instance
_db_manager_instance: Optional[DatabaseManager] = None

def get_db_manager(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """
    Get singleton DatabaseManager instance
    
    Args:
        config: DatabaseConfig (creates new instance if provided)
    
    Returns:
        DatabaseManager instance
    """
    global _db_manager_instance
    
    if config is not None or _db_manager_instance is None:
        _db_manager_instance = DatabaseManager(config)
    
    return _db_manager_instance
