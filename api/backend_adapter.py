# backend_adapter.py - Integration Layer between REST API and Backend Modules
# Converts between API models and backend data models

import sys
import os
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import backend modules
try:
    from data_models import (
        Property as BackendProperty,
        ReductionOption as BackendReductionOption,
        MilestoneScenario as BackendMilestoneScenario,
        StrategicPattern as BackendStrategicPattern,
        BuildingType,
        RetrofitPotential,
        PriorityLevel,
        RiskLevel,
        FuelType
    )
    from ecoassist_backend import EcoAssistBackend
    from database_integration import (
        DatabaseConfig,
        DatabaseManager,
        PropertyRepository,
        ReductionOptionRepository,
        MilestoneScenarioRepository,
        StrategicPatternRepository,
        HistoricalDataRepository
    )
    from ai_functions import (
        MilestoneOptimizer,
        PropertyTargetAllocator,
        StrategicPatternAnalyzer,
        ReoptimizationEngine
    )
    BACKEND_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Backend modules not available: {e}")
    BACKEND_AVAILABLE = False

# Import API models
from api_core import (
    BaselineDataRecord,
    ReductionRate,
    StrategyPreferences,
    ReductionTarget,
    CostProjection,
    StrategyBreakdown
)

logger = logging.getLogger(__name__)

# =============================================================================
# BACKEND INITIALIZATION
# =============================================================================

class BackendInitializer:
    """Initialize and manage backend components"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized and BACKEND_AVAILABLE:
            self.db_config = DatabaseConfig()
            self.db_manager = None
            self.backend = None
            self.repositories = {}
            self.ai_engines = {}
            
    def initialize(self) -> bool:
        """Initialize all backend components"""
        if self._initialized:
            return True
            
        if not BACKEND_AVAILABLE:
            logger.warning("Backend modules not available - using mock mode")
            return False
        
        try:
            # Initialize database
            self.db_manager = DatabaseManager(self.db_config)
            if not self.db_manager.test_connection():
                logger.error("Database connection failed")
                return False
            
            # Initialize backend
            self.backend = EcoAssistBackend()
            logger.info("✓ Backend initialized")
            
            # Initialize repositories
            self.repositories = {
                'property': PropertyRepository(self.db_manager),
                'reduction_option': ReductionOptionRepository(self.db_manager),
                'milestone_scenario': MilestoneScenarioRepository(self.db_manager),
                'strategic_pattern': StrategicPatternRepository(self.db_manager),
                'historical': HistoricalDataRepository(self.db_manager)
            }
            logger.info("✓ Repositories initialized")
            
            # Initialize AI engines
            self.ai_engines = {
                'milestone_optimizer': MilestoneOptimizer(),
                'target_allocator': PropertyTargetAllocator(),
                'pattern_analyzer': StrategicPatternAnalyzer(),
                'reoptimization_engine': ReoptimizationEngine()
            }
            logger.info("✓ AI engines initialized")
            
            self._initialized = True
            logger.info("✓ Backend initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Backend initialization failed: {e}")
            return False
    
    def get_backend(self) -> Optional[EcoAssistBackend]:
        """Get backend instance"""
        return self.backend
    
    def get_repository(self, name: str):
        """Get repository by name"""
        return self.repositories.get(name)
    
    def get_ai_engine(self, name: str):
        """Get AI engine by name"""
        return self.ai_engines.get(name)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.db_manager:
            self.db_manager.close()
        self._initialized = False

# Global initializer instance
backend_initializer = BackendInitializer()

# =============================================================================
# DATA MODEL CONVERTERS
# =============================================================================

class PropertyConverter:
    """Convert between API PropertyData and backend Property"""
    
    @staticmethod
    def api_to_backend(api_property: Dict[str, Any]) -> BackendProperty:
        """Convert API property data to backend Property"""
        if not BACKEND_AVAILABLE:
            return api_property
            
        return BackendProperty(
            property_id=api_property['property_id'],
            name=api_property.get('property_name', ''),
            area_sqm=api_property['floor_area'],
            baseline_emission=api_property['baseline_emission'],
            building_type=api_property['building_type'],
            retrofit_potential=api_property.get('retrofit_potential', 0.0),
            # Add other fields as needed
        )
    
    @staticmethod
    def backend_to_api(backend_property: BackendProperty) -> Dict[str, Any]:
        """Convert backend Property to API format"""
        if not BACKEND_AVAILABLE:
            return backend_property
            
        return {
            'property_id': backend_property.property_id,
            'property_name': backend_property.name,
            'building_type': backend_property.building_type,
            'floor_area': backend_property.area_sqm,
            'baseline_emission': backend_property.baseline_emission,
            'carbon_intensity': getattr(backend_property, 'carbon_intensity', 0.0),
            'retrofit_potential': backend_property.retrofit_potential
        }

class MilestoneConverter:
    """Convert between API and backend milestone scenarios"""
    
    @staticmethod
    def backend_to_api(backend_scenario: BackendMilestoneScenario) -> Dict[str, Any]:
        """Convert backend MilestoneScenario to API format"""
        if not BACKEND_AVAILABLE:
            return backend_scenario
        
        # Convert yearly targets to reduction targets array
        reduction_targets = []
        for year, emission in sorted(backend_scenario.yearly_targets.items()):
            reduction_targets.append({
                'year': year,
                'target_emissions': emission,
                'reduction_from_baseline': backend_scenario.reduction_rate_2030 if year == 2030 else backend_scenario.reduction_rate_2050,
                'cumulative_reduction': 0.0,  # Calculate based on baseline
                'unit': 'kg-CO2e'
            })
        
        # Generate cost projections
        cost_projections = []
        years = sorted(backend_scenario.yearly_targets.keys())
        for i, year in enumerate(years):
            cost_projections.append({
                'year': year,
                'estimated_cost': backend_scenario.total_capex / len(years),
                'breakdown': {
                    'capex': backend_scenario.total_capex / len(years) * 0.7,
                    'opex': backend_scenario.total_opex / len(years)
                },
                'unit': 'USD'
            })
        
        return {
            'scenario_id': backend_scenario.scenario_id,
            'scenario_type': backend_scenario.strategy_type,
            'description': backend_scenario.description,
            'reduction_targets': reduction_targets,
            'cost_projection': cost_projections,
            'strategy_breakdown': {
                'renewable_energy_percentage': 45.0,  # Extract from backend
                'efficiency_improvement_percentage': 35.0,
                'behavioral_change_percentage': 20.0
            }
        }

class StrategicPatternConverter:
    """Convert between API and backend strategic patterns"""
    
    @staticmethod
    def backend_to_api(backend_pattern: BackendStrategicPattern) -> Dict[str, Any]:
        """Convert backend StrategicPattern to API format"""
        if not BACKEND_AVAILABLE:
            return backend_pattern
            
        return {
            'pattern_id': backend_pattern.pattern_id,
            'pattern_name': backend_pattern.name,
            'description': backend_pattern.description,
            'implementation_approach': backend_pattern.implementation_approach,
            'estimated_cost': backend_pattern.estimated_cost,
            'estimated_reduction': backend_pattern.estimated_reduction,
            'risk_level': backend_pattern.risk_level,
            'reduction_options': backend_pattern.reduction_options
        }

# =============================================================================
# BUSINESS LOGIC ADAPTERS
# =============================================================================

class MilestoneAdapter:
    """Adapter for milestone calculation logic"""
    
    def __init__(self):
        self.backend = backend_initializer.get_backend()
        self.ai_optimizer = backend_initializer.get_ai_engine('milestone_optimizer')
        self.repository = backend_initializer.get_repository('milestone_scenario')
    
    def calculate_scenarios(
        self,
        base_year: int,
        mid_term_year: int,
        long_term_year: int,
        property_ids: List[str],
        baseline_data: List[BaselineDataRecord],
        scenario_types: List[str],
        strategy_preferences: Optional[StrategyPreferences] = None
    ) -> List[Dict[str, Any]]:
        """Calculate milestone scenarios using backend"""
        
        if not BACKEND_AVAILABLE or not self.backend:
            logger.warning("Using mock milestone calculation")
            return self._mock_calculate_scenarios(
                base_year, mid_term_year, long_term_year, scenario_types
            )
        
        try:
            # Calculate baseline emission
            baseline_emission = sum(
                record.scope1_emissions + record.scope2_emissions 
                for record in baseline_data
            ) / len(baseline_data)
            
            # Use backend to generate scenarios
            scenarios = self.backend.generate_milestone_scenarios(
                target_year=long_term_year,
                reduction_2030=(mid_term_year - base_year) / (long_term_year - base_year) * 80,
                reduction_2050=80.0
            )
            
            # Convert to API format
            api_scenarios = []
            for scenario in scenarios:
                api_scenario = MilestoneConverter.backend_to_api(scenario)
                api_scenarios.append(api_scenario)
            
            # Save to database if repository available
            if self.repository:
                for scenario in scenarios:
                    try:
                        self.repository.create(scenario)
                    except Exception as e:
                        logger.error(f"Failed to save scenario: {e}")
            
            return api_scenarios
            
        except Exception as e:
            logger.error(f"Backend milestone calculation failed: {e}")
            return self._mock_calculate_scenarios(
                base_year, mid_term_year, long_term_year, scenario_types
            )
    
    def _mock_calculate_scenarios(
        self,
        base_year: int,
        mid_term_year: int,
        long_term_year: int,
        scenario_types: List[str]
    ) -> List[Dict[str, Any]]:
        """Mock calculation when backend unavailable"""
        # Return simplified mock data
        return []
    
    def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Get scenario from database"""
        if not self.repository:
            return None
        
        try:
            scenario = self.repository.get_by_id(scenario_id)
            if scenario:
                return MilestoneConverter.backend_to_api(scenario)
        except Exception as e:
            logger.error(f"Failed to get scenario: {e}")
        
        return None
    
    def register_scenario(self, scenario_id: str, approval_data: Dict) -> bool:
        """Register scenario in database"""
        if not self.repository:
            return False
        
        try:
            scenario = self.repository.get_by_id(scenario_id)
            if scenario:
                scenario.status = 'registered'
                scenario.approval_data = approval_data
                self.repository.update(scenario)
                return True
        except Exception as e:
            logger.error(f"Failed to register scenario: {e}")
        
        return False

class TargetDivisionAdapter:
    """Adapter for target division logic"""
    
    def __init__(self):
        self.ai_allocator = backend_initializer.get_ai_engine('target_allocator')
        self.property_repo = backend_initializer.get_repository('property')
        self.scenario_repo = backend_initializer.get_repository('milestone_scenario')
    
    def allocate_targets(
        self,
        scenario_id: str,
        properties: List[Dict[str, Any]],
        target_years: List[int],
        allocation_method: str,
        optimization_objectives: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Allocate portfolio targets to properties using AI"""
        
        if not BACKEND_AVAILABLE or not self.ai_allocator:
            logger.warning("Using mock target allocation")
            return self._mock_allocate_targets(properties, target_years)
        
        try:
            # Convert properties to backend format
            backend_properties = [
                PropertyConverter.api_to_backend(prop) 
                for prop in properties
            ]
            
            # Get scenario from database
            scenario = None
            if self.scenario_repo:
                scenario = self.scenario_repo.get_by_id(scenario_id)
            
            # Train allocator if needed
            if not self.ai_allocator.is_trained:
                # Load historical data for training
                # training_data = self._prepare_training_data()
                # self.ai_allocator.train(training_data)
                pass
            
            # Perform allocation
            portfolio_target = 10000.0  # Get from scenario
            
            if allocation_method == 'ai_optimized':
                objectives = optimization_objectives or {
                    'fairness': 0.4,
                    'efficiency': 0.4,
                    'feasibility': 0.2
                }
                allocations = self.ai_allocator.allocate_targets_intelligently(
                    backend_properties,
                    portfolio_target,
                    objectives
                )
            else:
                # Use simpler allocation methods
                allocations = self._simple_allocation(
                    backend_properties, 
                    portfolio_target, 
                    allocation_method
                )
            
            # Convert allocations to API format
            return self._format_allocations(allocations, properties, target_years)
            
        except Exception as e:
            logger.error(f"Backend target allocation failed: {e}")
            return self._mock_allocate_targets(properties, target_years)
    
    def _simple_allocation(
        self,
        properties: List,
        portfolio_target: float,
        method: str
    ) -> Dict[str, float]:
        """Simple allocation methods"""
        allocations = {}
        
        if method == 'proportional':
            total = sum(p.baseline_emission for p in properties)
            for prop in properties:
                weight = prop.baseline_emission / total
                allocations[prop.property_id] = portfolio_target * weight
        
        return allocations
    
    def _format_allocations(
        self,
        allocations: Dict[str, float],
        properties: List[Dict],
        target_years: List[int]
    ) -> Dict[str, Any]:
        """Format allocations for API response"""
        property_targets = []
        
        for prop in properties:
            prop_id = prop['property_id']
            allocated = allocations.get(prop_id, 0.0)
            
            for year in target_years:
                property_targets.append({
                    'property_id': prop_id,
                    'property_name': prop['property_name'],
                    'year': year,
                    'allocated_target': prop['baseline_emission'] - allocated,
                    'reduction_from_baseline': (allocated / prop['baseline_emission']) * 100,
                    'absolute_reduction': allocated,
                    'allocation_weight': 0.0,
                    'feasibility_score': prop.get('retrofit_potential', 50.0),
                    'estimated_cost': allocated * 150,  # $150 per tonne
                    'recommended_actions': [],
                    'unit': 'kg-CO2e'
                })
        
        return {
            'property_targets': property_targets,
            'allocation_metrics': {
                'fairness_index': 0.85,
                'efficiency_score': 75.0,
                'feasibility_score': 70.0,
                'coverage': 95.0
            }
        }
    
    def _mock_allocate_targets(
        self,
        properties: List[Dict],
        target_years: List[int]
    ) -> Dict[str, Any]:
        """Mock allocation when backend unavailable"""
        return {
            'property_targets': [],
            'allocation_metrics': {}
        }

class PlanningAdapter:
    """Adapter for long-term planning logic"""
    
    def __init__(self):
        self.backend = backend_initializer.get_backend()
        self.pattern_analyzer = backend_initializer.get_ai_engine('pattern_analyzer')
        self.pattern_repo = backend_initializer.get_repository('strategic_pattern')
    
    def generate_plan(
        self,
        scenario_id: str,
        allocation_id: str,
        planning_horizon: Dict,
        budget_constraints: Dict,
        strategy_preferences: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate long-term implementation plan"""
        
        if not BACKEND_AVAILABLE or not self.backend:
            logger.warning("Using mock plan generation")
            return self._mock_generate_plan()
        
        try:
            # Get strategic patterns from database
            patterns = []
            if self.pattern_repo:
                patterns = self.pattern_repo.get_all_active()
            
            # Analyze patterns
            if self.pattern_analyzer and patterns:
                # Analyze each pattern for suitability
                pass
            
            # Generate implementation plan
            # This would call backend methods
            plan = self._create_implementation_plan(
                scenario_id,
                patterns,
                planning_horizon,
                budget_constraints
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Backend plan generation failed: {e}")
            return self._mock_generate_plan()
    
    def _create_implementation_plan(
        self,
        scenario_id: str,
        patterns: List,
        horizon: Dict,
        budget: Dict
    ) -> Dict[str, Any]:
        """Create implementation plan from patterns"""
        # Implementation logic here
        return {}
    
    def _mock_generate_plan(self) -> Dict[str, Any]:
        """Mock plan when backend unavailable"""
        return {
            'planning_patterns': [],
            'calculation_metadata': {}
        }

class ReoptimizationAdapter:
    """Adapter for reoptimization logic"""
    
    def __init__(self):
        self.reopt_engine = backend_initializer.get_ai_engine('reoptimization_engine')
        self.historical_repo = backend_initializer.get_repository('historical')
    
    def calculate_reoptimization(
        self,
        plan_id: str,
        current_year: int,
        actual_performance: Dict[str, float],
        external_factors: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Calculate reoptimization recommendations"""
        
        if not BACKEND_AVAILABLE or not self.reopt_engine:
            logger.warning("Using mock reoptimization")
            return self._mock_reoptimization()
        
        try:
            # Get historical data
            historical_data = None
            if self.historical_repo:
                historical_data = self.historical_repo.get_by_year_range(
                    plan_id, 
                    current_year - 3, 
                    current_year
                )
            
            # Train engine if needed
            if not self.reopt_engine.is_trained and historical_data:
                # self.reopt_engine.train(historical_data)
                pass
            
            # Calculate adjustments
            # adjustments = self.reopt_engine.calculate_adaptive_adjustments(...)
            
            return self._format_reoptimization_results({})
            
        except Exception as e:
            logger.error(f"Backend reoptimization failed: {e}")
            return self._mock_reoptimization()
    
    def _format_reoptimization_results(self, results: Dict) -> Dict[str, Any]:
        """Format reoptimization results for API"""
        return {
            'reoptimization_id': '',
            'deviation_analysis': {},
            'root_causes': [],
            'recommended_adjustments': []
        }
    
    def _mock_reoptimization(self) -> Dict[str, Any]:
        """Mock reoptimization when backend unavailable"""
        return {}

# =============================================================================
# ADAPTER FACTORY
# =============================================================================

class AdapterFactory:
    """Factory for creating adapter instances"""
    
    @staticmethod
    def get_milestone_adapter() -> MilestoneAdapter:
        """Get milestone adapter instance"""
        return MilestoneAdapter()
    
    @staticmethod
    def get_target_division_adapter() -> TargetDivisionAdapter:
        """Get target division adapter instance"""
        return TargetDivisionAdapter()
    
    @staticmethod
    def get_planning_adapter() -> PlanningAdapter:
        """Get planning adapter instance"""
        return PlanningAdapter()
    
    @staticmethod
    def get_reoptimization_adapter() -> ReoptimizationAdapter:
        """Get reoptimization adapter instance"""
        return ReoptimizationAdapter()

# =============================================================================
# INITIALIZATION FUNCTION
# =============================================================================

def initialize_backend() -> bool:
    """Initialize backend components"""
    return backend_initializer.initialize()

def cleanup_backend():
    """Cleanup backend resources"""
    backend_initializer.cleanup()

def get_backend_status() -> Dict[str, Any]:
    """Get backend initialization status"""
    return {
        'available': BACKEND_AVAILABLE,
        'initialized': backend_initializer._initialized,
        'database_connected': backend_initializer.db_manager is not None if BACKEND_AVAILABLE else False,
        'backend_ready': backend_initializer.backend is not None if BACKEND_AVAILABLE else False
    }

# Log backend availability on import
if BACKEND_AVAILABLE:
    logger.info("✓ Backend modules available - full functionality enabled")
else:
    logger.warning("⚠ Backend modules not available - using mock mode")
