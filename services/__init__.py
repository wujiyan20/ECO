# services/__init__.py - Services Package
"""
EcoAssist Services Layer Package

This package provides the business logic layer for the EcoAssist system,
implementing the separation between API endpoints and core functionality.

Services:
- PropertyService: Property and portfolio management
- MilestoneService: Milestone calculation and scenario generation
- AllocationService: Target allocation algorithms
- TrackingService: Progress monitoring and deviation analysis
- AIService: AI/ML model management and predictions
- VisualizationService: Charts, reports, and dashboards

Usage:
    from services import (
        PropertyService,
        MilestoneService,
        AllocationService,
        TrackingService,
        AIService,
        VisualizationService,
        get_service_registry
    )
    
    # Initialize services
    registry = get_service_registry()
    
    property_service = PropertyService(db_manager)
    registry.register("property", property_service)
    
    # Use services
    result = property_service.get_property("PROP-001")
    if result.is_success:
        property = result.data
"""

__version__ = "2.0.0"
__author__ = "EcoAssist Development Team"

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# BASE SERVICE COMPONENTS
# =============================================================================

from .base_service import (
    # Result classes
    ServiceResultStatus,
    ServiceResult,
    
    # Caching
    SimpleCache,
    CacheEntry,
    get_cache,
    
    # Decorators
    measure_time,
    cached,
    retry,
    transaction,
    validate_input,
    
    # Base classes
    BaseService,
    ServiceRegistry,
    get_service_registry,
    
    # Context
    ServiceContext,
    set_service_context,
    get_service_context,
    clear_service_context,
    create_service_context,
    
    # Utilities
    batch_operation
)

# =============================================================================
# PROPERTY SERVICE
# =============================================================================

from .property_service import (
    PropertyService,
    PropertySummary,
    PortfolioSummary,
    PropertyUpdateRequest
)

# =============================================================================
# MILESTONE SERVICE
# =============================================================================

from .milestone_service import (
    MilestoneService,
    MilestoneCalculationRequest,
    MilestoneCalculationResult,
    ScenarioConfig,
    DEFAULT_SCENARIOS
)

# =============================================================================
# ALLOCATION SERVICE
# =============================================================================

from .allocation_service import (
    AllocationService,
    AllocationRequest,
    AllocationResult,
    PropertyAllocation,
    AllocationAdjustment
)

# =============================================================================
# TRACKING SERVICE
# =============================================================================

from .tracking_service import (
    TrackingService,
    ProgressData,
    ProgressSummary,
    DeviationAnalysis,
    ReoptimizationRecommendation,
    AlertSeverity,
    TrendDirection
)

# =============================================================================
# AI SERVICE
# =============================================================================

from .ai_service import (
    AIService,
    ModelType,
    ModelStatus,
    ModelInfo,
    PredictionRequest,
    PredictionResult,
    TrainingRequest,
    TrainingResult
)

# =============================================================================
# VISUALIZATION SERVICE
# =============================================================================

from .visualization_service import (
    VisualizationService,
    ChartType,
    ReportFormat,
    ChartConfig,
    ChartData,
    ReportSection,
    Report,
    DashboardData,
    COLOR_SCHEMES
)

# =============================================================================
# SERVICE FACTORY
# =============================================================================

class ServiceFactory:
    """
    Factory for creating and managing service instances.
    
    Provides centralized service creation with dependency injection.
    
    Usage:
        factory = ServiceFactory(db_manager)
        
        # Create all services
        services = factory.create_all()
        
        # Or create specific service
        property_service = factory.create_property_service()
    """
    
    def __init__(self, db_manager=None):
        """
        Initialize service factory.
        
        Args:
            db_manager: Database manager instance to inject into services
        """
        self.db_manager = db_manager
        self._services = {}
    
    def create_property_service(self) -> PropertyService:
        """Create property service instance"""
        if 'property' not in self._services:
            self._services['property'] = PropertyService(self.db_manager)
        return self._services['property']
    
    def create_milestone_service(self) -> MilestoneService:
        """Create milestone service instance"""
        if 'milestone' not in self._services:
            self._services['milestone'] = MilestoneService(self.db_manager)
        return self._services['milestone']
    
    def create_allocation_service(self) -> AllocationService:
        """Create allocation service instance"""
        if 'allocation' not in self._services:
            property_service = self.create_property_service()
            self._services['allocation'] = AllocationService(
                self.db_manager, property_service
            )
        return self._services['allocation']
    
    def create_tracking_service(self) -> TrackingService:
        """Create tracking service instance"""
        if 'tracking' not in self._services:
            milestone_service = self.create_milestone_service()
            self._services['tracking'] = TrackingService(
                self.db_manager, milestone_service
            )
        return self._services['tracking']
    
    def create_ai_service(self) -> AIService:
        """Create AI service instance"""
        if 'ai' not in self._services:
            self._services['ai'] = AIService(self.db_manager)
        return self._services['ai']
    
    def create_visualization_service(self) -> VisualizationService:
        """Create visualization service instance"""
        if 'visualization' not in self._services:
            milestone_service = self.create_milestone_service()
            tracking_service = self.create_tracking_service()
            property_service = self.create_property_service()
            self._services['visualization'] = VisualizationService(
                self.db_manager,
                milestone_service,
                tracking_service,
                property_service
            )
        return self._services['visualization']
    
    def create_all(self) -> dict:
        """
        Create all service instances.
        
        Returns:
            Dictionary of service name to service instance
        """
        return {
            'property': self.create_property_service(),
            'milestone': self.create_milestone_service(),
            'allocation': self.create_allocation_service(),
            'tracking': self.create_tracking_service(),
            'ai': self.create_ai_service(),
            'visualization': self.create_visualization_service()
        }
    
    def initialize_all(self) -> dict:
        """
        Create and initialize all services.
        
        Returns:
            Dictionary of service name to initialization result
        """
        services = self.create_all()
        results = {}
        
        for name, service in services.items():
            results[name] = service.initialize()
            if results[name].is_success:
                logger.info(f"✓ {name} service initialized")
            else:
                logger.error(f"✗ {name} service initialization failed")
        
        return results
    
    def get_service(self, name: str) -> BaseService:
        """
        Get a service by name.
        
        Args:
            name: Service name
            
        Returns:
            Service instance or None
        """
        return self._services.get(name)


def create_services(db_manager=None) -> dict:
    """
    Convenience function to create all services.
    
    Args:
        db_manager: Database manager instance
        
    Returns:
        Dictionary of service name to service instance
    """
    factory = ServiceFactory(db_manager)
    return factory.create_all()


def initialize_services(db_manager=None) -> dict:
    """
    Convenience function to create and initialize all services.
    
    Args:
        db_manager: Database manager instance
        
    Returns:
        Dictionary of service name to initialization result
    """
    factory = ServiceFactory(db_manager)
    return factory.initialize_all()


# =============================================================================
# MODULE SUMMARY
# =============================================================================

def get_service_summary() -> dict:
    """
    Get summary of available services.
    
    Returns:
        Dictionary with service information
    """
    return {
        "version": __version__,
        "services": [
            {
                "name": "PropertyService",
                "description": "Property and portfolio management",
                "operations": ["get_property", "get_properties", "create_property", 
                              "update_property", "delete_property", "get_portfolio_metrics"]
            },
            {
                "name": "MilestoneService", 
                "description": "Milestone calculation and scenario generation",
                "operations": ["calculate_milestones", "save_milestone_scenario",
                              "get_milestone_scenario", "compare_scenarios"]
            },
            {
                "name": "AllocationService",
                "description": "Target allocation across properties",
                "operations": ["allocate_targets", "adjust_allocation",
                              "rebalance_allocation", "validate_allocation"]
            },
            {
                "name": "TrackingService",
                "description": "Progress monitoring and deviation analysis",
                "operations": ["record_actual_data", "get_progress_summary",
                              "analyze_deviations", "generate_alerts"]
            },
            {
                "name": "AIService",
                "description": "AI/ML model management and predictions",
                "operations": ["predict", "train_model", "optimize",
                              "get_model_status"]
            },
            {
                "name": "VisualizationService",
                "description": "Charts, reports, and dashboards",
                "operations": ["generate_milestone_chart", "create_progress_report",
                              "get_dashboard_data", "export_report"]
            }
        ],
        "decorators": ["measure_time", "cached", "retry", "transaction"],
        "utilities": ["ServiceFactory", "ServiceRegistry", "ServiceContext"]
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Version
    '__version__',
    
    # Base service components
    'ServiceResultStatus',
    'ServiceResult',
    'SimpleCache',
    'CacheEntry',
    'get_cache',
    'measure_time',
    'cached',
    'retry',
    'transaction',
    'validate_input',
    'BaseService',
    'ServiceRegistry',
    'get_service_registry',
    'ServiceContext',
    'set_service_context',
    'get_service_context',
    'clear_service_context',
    'create_service_context',
    'batch_operation',
    
    # Property service
    'PropertyService',
    'PropertySummary',
    'PortfolioSummary',
    'PropertyUpdateRequest',
    
    # Milestone service
    'MilestoneService',
    'MilestoneCalculationRequest',
    'MilestoneCalculationResult',
    'ScenarioConfig',
    'DEFAULT_SCENARIOS',
    
    # Allocation service
    'AllocationService',
    'AllocationRequest',
    'AllocationResult',
    'PropertyAllocation',
    'AllocationAdjustment',
    
    # Tracking service
    'TrackingService',
    'ProgressData',
    'ProgressSummary',
    'DeviationAnalysis',
    'ReoptimizationRecommendation',
    'AlertSeverity',
    'TrendDirection',
    
    # AI service
    'AIService',
    'ModelType',
    'ModelStatus',
    'ModelInfo',
    'PredictionRequest',
    'PredictionResult',
    'TrainingRequest',
    'TrainingResult',
    
    # Visualization service
    'VisualizationService',
    'ChartType',
    'ReportFormat',
    'ChartConfig',
    'ChartData',
    'ReportSection',
    'Report',
    'DashboardData',
    'COLOR_SCHEMES',
    
    # Factory and utilities
    'ServiceFactory',
    'create_services',
    'initialize_services',
    'get_service_summary'
]

# Log package initialization
logger.info(f"Services package v{__version__} loaded with {len(__all__)} exports")
