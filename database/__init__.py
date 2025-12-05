# database/__init__.py - Database Package Initialization
"""
EcoAssist Database Package - Simple Integration
Provides database access for the EcoAssist system
"""

import logging

logger = logging.getLogger(__name__)

# Package version
__version__ = '1.0.0'

# Import only what we have
try:
    from .database_config import (
        DatabaseConfig,
        DEVELOPMENT_CONFIG,
        PRODUCTION_CONFIG,
        get_config
    )
    from .database_manager import (
        DatabaseManager,
        get_db_manager
    )
    
    __all__ = [
        'DatabaseConfig',
        'DEVELOPMENT_CONFIG',
        'PRODUCTION_CONFIG',
        'get_config',
        'DatabaseManager',
        'get_db_manager',
    ]
    
    logger.info(f"EcoAssist Database Package v{__version__} initialized")
    
except ImportError as e:
    logger.error(f"Failed to import database modules: {e}")
    raise
