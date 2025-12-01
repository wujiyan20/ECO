# models/base.py - Base Model Classes and Common Utilities
"""
Base classes and utilities for all EcoAssist data models.
Provides common functionality like UUID generation, timestamps, and validation.
"""

import uuid
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List, Type, TypeVar
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

# Type variable for generic base model
T = TypeVar('T', bound='BaseModel')

# =============================================================================
# UUID AND IDENTIFIER UTILITIES
# =============================================================================

def generate_uuid() -> str:
    """Generate a new UUID v4 as string"""
    return str(uuid.uuid4())

def is_valid_uuid(uuid_string: str) -> bool:
    """Validate if string is a valid UUID"""
    try:
        uuid.UUID(uuid_string)
        return True
    except (ValueError, AttributeError):
        return False

def generate_property_id(prefix: str = "PROP") -> str:
    """Generate property ID with prefix"""
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"

def generate_scenario_id(prefix: str = "SCEN") -> str:
    """Generate scenario ID with prefix"""
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"

def generate_plan_id(prefix: str = "PLAN") -> str:
    """Generate plan ID with prefix"""
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"

def generate_calculation_id() -> str:
    """Generate calculation tracking ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique = uuid.uuid4().hex[:8].upper()
    return f"CALC-{timestamp}-{unique}"

# =============================================================================
# BASE MODEL CLASS
# =============================================================================

@dataclass
class BaseModel(ABC):
    """
    Abstract base class for all data models in EcoAssist system.
    Provides common fields and functionality.
    """
    id: str = field(default_factory=generate_uuid)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dataclass to dictionary with proper serialization.
        Handles datetime objects and nested structures.
        """
        def serialize_value(value):
            if isinstance(value, datetime):
                return value.isoformat()
            elif isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif hasattr(value, 'to_dict'):
                return value.to_dict()
            elif hasattr(value, 'value'):  # For Enums
                return value.value
            else:
                return value
        
        result = {}
        for key, value in asdict(self).items():
            result[key] = serialize_value(value)
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert model to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create model instance from dictionary"""
        # Filter out keys that don't exist in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        # Convert datetime strings back to datetime objects
        for key, value in filtered_data.items():
            if key in ('created_at', 'updated_at') and isinstance(value, str):
                filtered_data[key] = datetime.fromisoformat(value)
        
        return cls(**filtered_data)
    
    def update_timestamp(self, user: Optional[str] = None):
        """Update the updated_at timestamp and user"""
        self.updated_at = datetime.now()
        if user:
            self.updated_by = user
    
    def deactivate(self, user: Optional[str] = None):
        """Mark record as inactive (soft delete)"""
        self.is_active = False
        self.update_timestamp(user)
    
    def activate(self, user: Optional[str] = None):
        """Mark record as active"""
        self.is_active = True
        self.update_timestamp(user)
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata entry"""
        self.metadata[key] = value
        self.update_timestamp()
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value"""
        return self.metadata.get(key, default)
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate model data.
        Returns tuple of (is_valid, error_messages)
        Subclasses should override to add specific validation.
        """
        errors = []
        
        # Validate UUID
        if not is_valid_uuid(self.id):
            errors.append(f"Invalid ID format: {self.id}")
        
        # Validate timestamps
        if self.updated_at < self.created_at:
            errors.append("Updated timestamp cannot be before created timestamp")
        
        return len(errors) == 0, errors
    
    def __repr__(self) -> str:
        """String representation of model"""
        class_name = self.__class__.__name__
        return f"{class_name}(id={self.id[:8]}...)"

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

class ValidationError(Exception):
    """Custom validation error exception"""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {', '.join(errors)}")

def validate_positive(value: float, field_name: str) -> None:
    """Validate that a value is positive"""
    if value < 0:
        raise ValidationError([f"{field_name} must be positive, got {value}"])

def validate_percentage(value: float, field_name: str) -> None:
    """Validate that a value is a valid percentage (0-100)"""
    if not 0 <= value <= 100:
        raise ValidationError([f"{field_name} must be between 0 and 100, got {value}"])

def validate_year(year: int, min_year: int = 2000, max_year: int = 2100) -> None:
    """Validate year is within reasonable range"""
    if not min_year <= year <= max_year:
        raise ValidationError([f"Year must be between {min_year} and {max_year}, got {year}"])

def validate_year_range(start_year: int, end_year: int) -> None:
    """Validate year range"""
    validate_year(start_year)
    validate_year(end_year)
    if end_year <= start_year:
        raise ValidationError([f"End year ({end_year}) must be after start year ({start_year})"])

def validate_non_empty(value: str, field_name: str) -> None:
    """Validate that string is not empty"""
    if not value or not value.strip():
        raise ValidationError([f"{field_name} cannot be empty"])

def validate_list_not_empty(value: List[Any], field_name: str) -> None:
    """Validate that list is not empty"""
    if not value or len(value) == 0:
        raise ValidationError([f"{field_name} cannot be empty"])

# =============================================================================
# DATETIME UTILITIES
# =============================================================================

def get_current_timestamp() -> datetime:
    """Get current timestamp"""
    return datetime.now()

def get_years_between(start_year: int, end_year: int) -> List[int]:
    """Get list of years between start and end (inclusive)"""
    return list(range(start_year, end_year + 1))

# =============================================================================
# NUMERIC UTILITIES
# =============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for division by zero"""
    if denominator == 0:
        return default
    return numerator / denominator

def calculate_percentage(part: float, total: float) -> float:
    """Calculate percentage with safe division"""
    return safe_divide(part, total, 0.0) * 100

__all__ = [
    'BaseModel', 'ValidationError', 'generate_uuid', 'is_valid_uuid',
    'generate_property_id', 'generate_scenario_id', 'generate_plan_id',
    'generate_calculation_id', 'validate_positive', 'validate_percentage',
    'validate_year', 'validate_year_range', 'validate_non_empty',
    'validate_list_not_empty', 'get_current_timestamp', 'get_years_between',
    'safe_divide', 'calculate_percentage'
]
