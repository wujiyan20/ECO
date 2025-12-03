# models/__init__.py - Models Package
"""
EcoAssist Data Models Package
Comprehensive data models for carbon reduction planning and management.

This package provides all data models needed for the EcoAssist system:
- Property and portfolio management
- Emission tracking and baselines
- Milestone planning and scenarios
- Reduction options and strategies
- Cost projections and financial analysis
- Data access layer (repositories)
"""

__version__ = "2.0.0"
__author__ = "EcoAssist Development Team"

# =============================================================================
# ENUMERATIONS
# =============================================================================

from .enums import (
    # Property & Building
    BuildingType,
    RetrofitPotential,
    PropertyStatus,
    
    # Priority & Risk
    PriorityLevel,
    RiskLevel,
    UrgencyLevel,
    
    # Emissions
    EmissionScope,
    EmissionCategory,
    FuelType,
    
    # Workflow & Status
    ApprovalStatus,
    ImplementationStatus,
    OnTrackStatus,
    
    # Scenario & Strategy
    ScenarioType,
    AllocationMethod,
    StrategyType,
    
    # Data Types
    DataType,
    DataQuality,
    MeasurementMethod,
    
    # Cost & Financial
    CostCategory,
    Currency,
    
    # Reporting
    ReportingPeriod,
    ReportFormat,
    
    # Units
    EmissionUnit,
    EnergyUnit,
    AreaUnit,
    
    # Helper Functions
    get_enum_values,
    get_enum_choices,
    validate_enum_value,
    
    # Mappings
    PRIORITY_NUMERIC_MAP,
    RISK_SCORE_MAP,
    RETROFIT_POTENTIAL_MAP
)

# =============================================================================
# BASE CLASSES
# =============================================================================

from .base import (
    # Base Classes
    BaseModel,
    #TimestampedModel,
    AuditableModel,
    
    # ID Generation
    generate_uuid,
    generate_property_id,
    generate_milestone_id,
    generate_scenario_id,
    generate_calculation_id,
    generate_hash,
    
    # Validation
    ValidationError,
    validate_year,
    validate_year_range,
    validate_percentage,
    validate_positive_number,
    validate_uuid,
    validate_email,
    validate_list_not_empty,
    
    # Conversion
    convert_to_tonnes,
    convert_area,
    calculate_carbon_intensity,
    calculate_percentage_change,
    round_to_decimals,
    
    # Date/Time
    get_current_year,
    get_financial_year,
    get_date_range_for_year,
    days_between,
    add_years,
    
    # Logging
    log_model_change,
    log_validation_error
)

# =============================================================================
# PROPERTY MODELS
# =============================================================================

from .property import (
    Property,
    PropertyEmissionBreakdown,
    PropertyMetrics,
    Portfolio,
    PropertyFilter
)

# =============================================================================
# EMISSION MODELS
# =============================================================================

from .emission import (
    BaselineDataRecord,
    EmissionFactor,
    EmissionCalculation,
    EmissionTrend,
    EmissionProjection,
    EmissionBenchmark
)

# =============================================================================
# MILESTONE MODELS
# =============================================================================

from .milestone import (
    MilestoneTarget,
    MilestoneScenario,
    ScenarioComparison,
    MilestoneProgress,
    MilestoneAlert
)

# =============================================================================
# REDUCTION MODELS
# =============================================================================

from .reduction import (
    ReductionOption,
    ReductionStrategy,
    ImplementationPhase,
    ImplementationPlan,
    ReductionRate,
    StrategyBreakdown
)

# =============================================================================
# COST MODELS
# =============================================================================

from .cost import (
    CostProjection,
    CapexOpex,
    CostSchedule,
    ROICalculation,
    NPVCalculation,
    PaybackPeriod,
    FinancialMetrics,
    BudgetAllocation
)

# =============================================================================
# REPOSITORY (DATA ACCESS LAYER)
# =============================================================================

from .repository import (
    BaseRepository,
    PropertyRepository,
    EmissionRepository,
    MilestoneRepository,
    ReductionOptionRepository
)

# =============================================================================
# GROUPED EXPORTS
# =============================================================================

# All enumerations
ENUMS = [
    BuildingType, RetrofitPotential, PropertyStatus,
    PriorityLevel, RiskLevel, UrgencyLevel,
    EmissionScope, EmissionCategory, FuelType,
    ApprovalStatus, ImplementationStatus, OnTrackStatus,
    ScenarioType, AllocationMethod, StrategyType,
    DataType, DataQuality, MeasurementMethod,
    CostCategory, Currency,
    ReportingPeriod, ReportFormat,
    EmissionUnit, EnergyUnit, AreaUnit
]

# All data models
MODELS = [
    Property, PropertyEmissionBreakdown, PropertyMetrics, Portfolio,
    BaselineDataRecord, EmissionFactor, EmissionCalculation, EmissionTrend,
    MilestoneTarget, MilestoneScenario, ScenarioComparison, MilestoneProgress,
    ReductionOption, ReductionStrategy, ImplementationPlan,
    CostProjection, CostSchedule, FinancialMetrics
]

# All repositories
REPOSITORIES = [
    PropertyRepository,
    EmissionRepository,
    MilestoneRepository,
    ReductionOptionRepository
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_enums():
    """Get list of all enumeration classes"""
    return ENUMS

def get_all_models():
    """Get list of all model classes"""
    return MODELS

def get_all_repositories():
    """Get list of all repository classes"""
    return REPOSITORIES

def create_repositories(db_manager):
    """
    Create all repository instances with a database manager
    
    Args:
        db_manager: DatabaseManager instance from database/manager.py
    
    Returns:
        Dictionary of repository instances
    """
    return {
        'property': PropertyRepository(db_manager),
        'emission': EmissionRepository(db_manager),
        'milestone': MilestoneRepository(db_manager),
        'reduction': ReductionOptionRepository(db_manager)
    }

# =============================================================================
# PACKAGE INFO
# =============================================================================

def get_version():
    """Get package version"""
    return __version__

def get_model_summary():
    """Get summary of available models"""
    return {
        'version': __version__,
        'enumerations': len(ENUMS),
        'models': len(MODELS),
        'repositories': len(REPOSITORIES),
        'categories': {
            'property': 4,
            'emission': 6,
            'milestone': 5,
            'reduction': 6,
            'cost': 8
        }
    }

# =============================================================================
# COMPLETE EXPORT LIST
# =============================================================================

__all__ = [
    # Version
    '__version__',
    
    # Enumerations
    'BuildingType', 'RetrofitPotential', 'PropertyStatus',
    'PriorityLevel', 'RiskLevel', 'UrgencyLevel',
    'EmissionScope', 'EmissionCategory', 'FuelType',
    'ApprovalStatus', 'ImplementationStatus', 'OnTrackStatus',
    'ScenarioType', 'AllocationMethod', 'StrategyType',
    'DataType', 'DataQuality', 'MeasurementMethod',
    'CostCategory', 'Currency',
    'ReportingPeriod', 'ReportFormat',
    'EmissionUnit', 'EnergyUnit', 'AreaUnit',
    
    # Base Classes
    'BaseModel', 'TimestampedModel', 'AuditableModel',
    'ValidationError',
    
    # Property Models
    'Property', 'PropertyEmissionBreakdown', 'PropertyMetrics', 
    'Portfolio', 'PropertyFilter',
    
    # Emission Models
    'BaselineDataRecord', 'EmissionFactor', 'EmissionCalculation',
    'EmissionTrend', 'EmissionProjection', 'EmissionBenchmark',
    
    # Milestone Models
    'MilestoneTarget', 'MilestoneScenario', 'ScenarioComparison',
    'MilestoneProgress', 'MilestoneAlert',
    
    # Reduction Models
    'ReductionOption', 'ReductionStrategy', 'ImplementationPhase',
    'ImplementationPlan', 'ReductionRate', 'StrategyBreakdown',
    
    # Cost Models
    'CostProjection', 'CapexOpex', 'CostSchedule',
    'ROICalculation', 'NPVCalculation', 'PaybackPeriod',
    'FinancialMetrics', 'BudgetAllocation',
    
    # Repositories
    'BaseRepository', 'PropertyRepository', 'EmissionRepository',
    'MilestoneRepository', 'ReductionOptionRepository',
    
    # Utility Functions
    'generate_uuid', 'generate_property_id', 'generate_milestone_id',
    'generate_scenario_id', 'generate_calculation_id',
    'validate_year', 'validate_year_range', 'validate_percentage',
    'validate_positive_number', 'get_enum_values', 'get_enum_choices',
    'calculate_carbon_intensity', 'calculate_percentage_change',
    'convert_to_tonnes', 'convert_area',
    'get_current_year', 'get_financial_year',
    
    # Package Functions
    'get_all_enums', 'get_all_models', 'get_all_repositories',
    'create_repositories', 'get_version', 'get_model_summary',
    
    # Collections
    'ENUMS', 'MODELS', 'REPOSITORIES'
]

# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

import logging
logger = logging.getLogger(__name__)
logger.info(f"EcoAssist Models Package v{__version__} loaded successfully")
logger.info(f"Available: {len(ENUMS)} enums, {len(MODELS)} models, {len(REPOSITORIES)} repositories")
