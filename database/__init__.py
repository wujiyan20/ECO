# database/__init__.py - Database Package Initialization
"""
EcoAssist Database Package
Provides database access, models, and repositories for the EcoAssist system
"""

import logging
from typing import Dict, Any

# Import configuration and management
from .config import DatabaseConfig, CommonConfigs, get_default_config, set_default_config
from .manager import DatabaseManager, ConnectionPool

# Import models
from .models import (
    # Core models
    Property,
    ReductionOption,
    MilestoneScenario,
    PropertyTarget,
    StrategicPattern,
    
    # Historical models
    HistoricalConsumption,
    HistoricalEmission,
    HistoricalPerformance,
    
    # Pricing models
    CarbonCreditPrice,
    RenewableEnergyPrice,
    RenewableFuelPrice,
    
    # Dashboard models
    DashboardMetrics,
    PortfolioSummary,
    
    # Enums
    BuildingType,
    RetrofitPotential,
    RiskLevel,
    OptionCategory,
)

# Import repositories
from .repositories import (
    PropertyRepository,
    MilestoneScenarioRepository,
    PropertyTargetRepository,
    ReductionOptionRepository,
    StrategicPatternRepository,
    HistoricalDataRepository,
    CarbonCreditRepository,
    DashboardRepository,
)

logger = logging.getLogger(__name__)

# Package version
__version__ = '1.0.0'

# Export all main classes and functions
__all__ = [
    # Configuration
    'DatabaseConfig',
    'CommonConfigs',
    'get_default_config',
    'set_default_config',
    
    # Manager
    'DatabaseManager',
    'ConnectionPool',
    
    # Models
    'Property',
    'ReductionOption',
    'MilestoneScenario',
    'PropertyTarget',
    'StrategicPattern',
    'HistoricalConsumption',
    'HistoricalEmission',
    'HistoricalPerformance',
    'CarbonCreditPrice',
    'RenewableEnergyPrice',
    'RenewableFuelPrice',
    'DashboardMetrics',
    'PortfolioSummary',
    
    # Enums
    'BuildingType',
    'RetrofitPotential',
    'RiskLevel',
    'OptionCategory',
    
    # Repositories
    'PropertyRepository',
    'MilestoneScenarioRepository',
    'PropertyTargetRepository',
    'ReductionOptionRepository',
    'StrategicPatternRepository',
    'HistoricalDataRepository',
    'CarbonCreditRepository',
    'DashboardRepository',
    
    # Convenience functions
    'quick_setup',
    'create_repositories',
    'EcoAssistDatabase',
]


# ================================================================================
# CONVENIENCE FUNCTIONS
# ================================================================================

def quick_setup(use_pool: bool = True) -> DatabaseManager:
    """
    Quick setup database manager from environment
    
    Args:
        use_pool: Whether to use connection pooling
    
    Returns:
        Configured DatabaseManager instance
    
    Example:
        >>> manager = quick_setup()
        >>> properties = manager.execute_query("SELECT * FROM properties")
    """
    config = DatabaseConfig.from_env()
    manager = DatabaseManager(config, use_pool=use_pool)
    
    # Test connection
    if not manager.test_connection():
        logger.warning("Database connection test failed")
    
    return manager


def create_repositories(db_manager: DatabaseManager) -> Dict[str, Any]:
    """
    Create all repository instances
    
    Args:
        db_manager: DatabaseManager instance
    
    Returns:
        Dictionary of repository instances
    
    Example:
        >>> manager = quick_setup()
        >>> repos = create_repositories(manager)
        >>> properties = repos['property'].get_all_active()
    """
    return {
        'property': PropertyRepository(db_manager),
        'milestone': MilestoneScenarioRepository(db_manager),
        'target': PropertyTargetRepository(db_manager),
        'reduction_option': ReductionOptionRepository(db_manager),
        'strategic_pattern': StrategicPatternRepository(db_manager),
        'historical': HistoricalDataRepository(db_manager),
        'carbon_credit': CarbonCreditRepository(db_manager),
        'dashboard': DashboardRepository(db_manager),
    }


class EcoAssistDatabase:
    """
    High-level database interface with repository access
    
    Example:
        >>> with EcoAssistDatabase() as db:
        ...     properties = db.property.get_all_active()
        ...     metrics = db.dashboard.get_portfolio_metrics()
    """
    
    def __init__(self, config: DatabaseConfig = None, use_pool: bool = True):
        """
        Initialize database interface
        
        Args:
            config: DatabaseConfig instance (uses environment if None)
            use_pool: Whether to use connection pooling
        """
        if config is None:
            config = DatabaseConfig.from_env()
        
        self.config = config
        self.manager = DatabaseManager(config, use_pool=use_pool)
        
        # Initialize repositories
        self.property = PropertyRepository(self.manager)
        self.milestone = MilestoneScenarioRepository(self.manager)
        self.target = PropertyTargetRepository(self.manager)
        self.reduction_option = ReductionOptionRepository(self.manager)
        self.strategic_pattern = StrategicPatternRepository(self.manager)
        self.historical = HistoricalDataRepository(self.manager)
        self.carbon_credit = CarbonCreditRepository(self.manager)
        self.dashboard = DashboardRepository(self.manager)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self.close()
    
    def close(self):
        """Close database manager and cleanup resources"""
        self.manager.close()
    
    def test_connection(self) -> bool:
        """
        Test database connection
        
        Returns:
            True if connection successful
        """
        return self.manager.test_connection()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database usage statistics
        
        Returns:
            Dictionary with statistics
        """
        return self.manager.get_statistics()


# ================================================================================
# INITIALIZATION
# ================================================================================

def initialize_logging(level=logging.INFO):
    """
    Initialize package logging
    
    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.setLevel(level)


# Initialize logging on import
initialize_logging()

logger.info(f"EcoAssist Database Package v{__version__} initialized")
