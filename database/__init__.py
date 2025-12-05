# database/__init__.py - Database Package
"""
EcoAssist Database Layer Package

This package provides database access and management for the EcoAssist system.

Components:
- DatabaseConfig: Configuration management
- DatabaseManager: Database operations and connection management

Usage:
    from database import DatabaseManager, DatabaseConfig, get_db_manager
    
    # Quick setup
    db = get_db_manager()
    
    # Or manual setup
    config = DatabaseConfig(server='jiyan', database='EcoAssist')
    db = DatabaseManager(config)
"""

__version__ = "1.0.0"
__author__ = "EcoAssist Development Team"

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

from .database_config import (
    DatabaseConfig,
    DEVELOPMENT_CONFIG,
    PRODUCTION_CONFIG,
    get_config
)

# =============================================================================
# DATABASE MANAGER
# =============================================================================

from .database_manager import (
    DatabaseManager,
    get_db_manager
)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Version
    '__version__',
    
    # Configuration
    'DatabaseConfig',
    'DEVELOPMENT_CONFIG',
    'PRODUCTION_CONFIG',
    'get_config',
    
    # Manager
    'DatabaseManager',
    'get_db_manager',
]

# Log package initialization
logger.info(f"Database package v{__version__} initialized successfully")
