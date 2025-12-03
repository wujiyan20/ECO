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
    id: str = field(default_factory=generate_uuid, init=False)
    created_at: datetime = field(default_factory=datetime.now, init=False)
    updated_at: datetime = field(default_factory=datetime.now, init=False)
    created_by: Optional[str] = field(default=None, init=False)
    updated_by: Optional[str] = field(default=None, init=False)
    is_active: bool = field(default=True, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Initialize base fields if not already set"""
        if not hasattr(self, 'id') or not self.id:
            self.id = generate_uuid()
        if not hasattr(self, 'created_at'):
            self.created_at = datetime.now()
        if not hasattr(self, 'updated_at'):
            self.updated_at = datetime.now()
        if not hasattr(self, 'created_by'):
            self.created_by = None
        if not hasattr(self, 'updated_by'):
            self.updated_by = None
        if not hasattr(self, 'is_active'):
            self.is_active = True
        if not hasattr(self, 'metadata'):
            self.metadata = {}
    
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


@dataclass
class AuditableModel(BaseModel):
    """
    Extended base model with full audit trail.
    Adds versioning and change tracking capabilities.
    """
    version: int = field(default=1, init=False)
    last_modified_by: Optional[str] = field(default=None, init=False)
    last_modified_at: Optional[datetime] = field(default=None, init=False)
    change_notes: Optional[str] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize audit fields"""
        super().__post_init__()
        if not hasattr(self, 'version'):
            self.version = 1
        if not hasattr(self, 'last_modified_by'):
            self.last_modified_by = None
        if not hasattr(self, 'last_modified_at'):
            self.last_modified_at = None
        if not hasattr(self, 'change_notes'):
            self.change_notes = None
    
    def increment_version(self, user: Optional[str] = None, notes: Optional[str] = None):
        """Increment version and update audit fields"""
        self.version += 1
        self.last_modified_by = user
        self.last_modified_at = datetime.now()
        self.change_notes = notes
        self.update_timestamp(user)

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
        
def validate_positive_number(value: float, field_name: str) -> None:
    """Validate that a number is positive (alias for validate_positive)"""
    validate_positive(value, field_name)

def calculate_carbon_intensity(total_emissions: float, area_sqm: float) -> float:
    """
    Calculate carbon intensity (kg-CO2e per square meter)
    
    Args:
        total_emissions: Total emissions in kg-CO2e
        area_sqm: Area in square meters
        
    Returns:
        Carbon intensity in kg-CO2e per sqm
    """
    return safe_divide(total_emissions, area_sqm, 0.0)

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change (positive = increase, negative = decrease)
        Returns 0.0 if old_value is 0
        
    Example:
        calculate_percentage_change(100, 120) -> 20.0
        calculate_percentage_change(100, 80) -> -20.0
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100

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
    

# =============================================================================
# ADDITIONAL DATE/TIME UTILITIES
# =============================================================================

def get_current_year() -> int:
    """Get current year"""
    return datetime.now().year

def get_financial_year(date: Optional[datetime] = None) -> int:
    """
    Get Australian financial year (July 1 - June 30)
    
    Args:
        date: Date to check (default: today)
        
    Returns:
        Financial year (e.g., 2024 for FY 2024-25)
    """
    if date is None:
        date = datetime.now()
    
    if date.month >= 7:  # July onwards is next FY
        return date.year
    else:
        return date.year - 1

def get_date_range_for_year(year: int) -> tuple[datetime, datetime]:
    """Get start and end dates for a calendar year"""
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31, 23, 59, 59)
    return start, end

def days_between(start: datetime, end: datetime) -> int:
    """Calculate days between two dates"""
    return (end - start).days

def add_years(date: datetime, years: int) -> datetime:
    """Add years to a date"""
    try:
        return date.replace(year=date.year + years)
    except ValueError:
        # Handle leap year edge case (Feb 29)
        return date.replace(year=date.year + years, day=28)

# =============================================================================
# ADDITIONAL ID GENERATION
# =============================================================================

def generate_milestone_id(prefix: str = "MILE") -> str:
    """Generate milestone ID with prefix"""
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"

def generate_hash(text: str) -> str:
    """Generate MD5 hash of text"""
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()

# =============================================================================
# ADDITIONAL VALIDATION UTILITIES
# =============================================================================

def validate_uuid(value: str, field_name: str) -> None:
    """Validate UUID format"""
    if not is_valid_uuid(value):
        raise ValidationError([f"{field_name} must be a valid UUID, got {value}"])

def validate_email(email: str, field_name: str = "Email") -> None:
    """Basic email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValidationError([f"{field_name} must be a valid email address"])

# =============================================================================
# CONVERSION UTILITIES
# =============================================================================

def convert_to_tonnes(kg: float) -> float:
    """Convert kg to tonnes"""
    return kg / 1000.0

def convert_area(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert area between units
    
    Args:
        value: Area value
        from_unit: Source unit ('sqm', 'sqft', 'acres', 'hectares')
        to_unit: Target unit
        
    Returns:
        Converted value
    """
    # Conversion factors to square meters
    to_sqm = {
        'sqm': 1.0,
        'sqft': 0.092903,
        'acres': 4046.86,
        'hectares': 10000.0
    }
    
    if from_unit not in to_sqm or to_unit not in to_sqm:
        raise ValueError(f"Invalid units: {from_unit} or {to_unit}")
    
    # Convert to sqm first, then to target unit
    sqm = value * to_sqm[from_unit]
    return sqm / to_sqm[to_unit]

def round_to_decimals(value: float, decimals: int = 2) -> float:
    """Round to specified decimal places"""
    return round(value, decimals)

# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def log_model_change(model: BaseModel, action: str, user: Optional[str] = None):
    """Log model changes"""
    logger.info(
        f"Model {action}: {model.__class__.__name__} "
        f"(id={model.id[:8]}..., user={user or 'system'})"
    )

def log_validation_error(model: BaseModel, errors: List[str]):
    """Log validation errors"""
    logger.error(
        f"Validation failed for {model.__class__.__name__} "
        f"(id={model.id[:8]}...): {', '.join(errors)}"
    )

__all__ = [
    'BaseModel', 'AuditableModel', 'ValidationError', 
    'generate_uuid', 'is_valid_uuid', 'generate_property_id', 'generate_scenario_id', 
    'generate_plan_id', 'generate_calculation_id', 'generate_milestone_id', 'generate_hash',
    'validate_positive', 'validate_positive_number', 'validate_percentage', 
    'validate_year', 'validate_year_range', 'validate_non_empty', 'validate_list_not_empty',
    'validate_uuid', 'validate_email',
    'get_current_timestamp', 'get_years_between', 'get_current_year', 'get_financial_year',
    'get_date_range_for_year', 'days_between', 'add_years',
    'safe_divide', 'calculate_percentage', 'calculate_percentage_change', 'calculate_carbon_intensity',
    'convert_to_tonnes', 'convert_area', 'round_to_decimals',
    'log_model_change', 'log_validation_error'
]