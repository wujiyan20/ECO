# models/reoptimization.py - Annual Reoptimization Data Models
"""
Data models for annual reoptimization functionality.

Models:
- ReoptimizationRequest: Request for reoptimization calculation
- PerformanceComparison: Actual vs planned performance
- AdjustedAction: Modified or new action
- ChangesFromOriginal: Summary of changes
- ImpactAnalysis: Analysis of reoptimization impact
- ReoptimizationPattern: Complete reoptimization strategy
- ReoptimizationResult: Complete reoptimization result
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class ReoptimizationStatus(Enum):
    """Status of reoptimization"""
    DRAFT = "draft"
    CALCULATED = "calculated"
    REGISTERED = "registered"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class FrequencyType(Enum):
    """Data aggregation frequency"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class PerformanceStatus(Enum):
    """Performance status relative to targets"""
    EXCEEDING = "exceeding"
    ON_TRACK = "on_track"
    BEHIND = "behind"
    CRITICAL = "critical"


class ActionChange(Enum):
    """Type of action change"""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    ACCELERATED = "accelerated"
    DELAYED = "delayed"


# =============================================================================
# REQUEST MODELS
# =============================================================================

@dataclass
class BudgetAdjustment:
    """Budget adjustment for reoptimization"""
    annual_budget_limit: Optional[float] = None
    currency: str = "USD"
    reason: Optional[str] = None


@dataclass
class ReoptimizationRequest:
    """Request for annual reoptimization calculation"""
    plan_id: str
    target_year: int
    start_date: str  # ISO 8601 format YYYY-MM-DD
    end_date: str    # ISO 8601 format YYYY-MM-DD
    frequency: str = "quarterly"
    budget_adjustment: Optional[BudgetAdjustment] = None
    
    def __post_init__(self):
        # Validate dates
        try:
            start = datetime.fromisoformat(self.start_date)
            end = datetime.fromisoformat(self.end_date)
            if end <= start:
                raise ValueError("end_date must be after start_date")
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}")
        
        # Validate frequency
        valid_frequencies = ["daily", "weekly", "monthly", "quarterly", "yearly"]
        if self.frequency not in valid_frequencies:
            raise ValueError(f"frequency must be one of: {valid_frequencies}")


# =============================================================================
# PERFORMANCE MODELS
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for a period"""
    planned_emission: float
    actual_emission: float
    variance: float
    variance_percentage: float
    emission_unit: str = "kg-CO2e"


@dataclass
class CostMetrics:
    """Cost metrics for a period"""
    planned_cost: float
    actual_cost: float
    variance: float
    variance_percentage: float
    cost_unit: str = "USD"


@dataclass
class PerformanceComparison:
    """Comparison of planned vs actual performance"""
    period: str  # e.g., "Q1-2025" or "2025"
    performance_metrics: PerformanceMetrics
    cost_metrics: CostMetrics
    status: str = "on_track"
    notes: Optional[str] = None


@dataclass
class HistoricalPerformance:
    """Historical performance data"""
    property_id: str
    year: int
    month: Optional[int] = None
    actual_emission: float = 0.0
    actual_cost: float = 0.0
    target_emission: Optional[float] = None
    target_cost: Optional[float] = None
    emission_unit: str = "kg-CO2e"
    cost_unit: str = "USD"


# =============================================================================
# ACTION ADJUSTMENT MODELS
# =============================================================================

@dataclass
class AdjustedAction:
    """Modified or new action in reoptimization"""
    action_id: str = ""
    action_type: str = "other"
    target_properties: List[str] = field(default_factory=list)
    expected_reduction: float = 0.0
    investment_required: float = 0.0
    priority: str = "medium"
    rationale: str = ""
    change_type: str = "added"  # added, removed, modified
    emission_unit: str = "kg-CO2e"
    cost_unit: str = "USD"
    
    def __post_init__(self):
        if not self.action_id:
            from models.base import generate_uuid
            self.action_id = generate_uuid()


@dataclass
class ChangesFromOriginal:
    """Summary of changes from original plan"""
    actions_added: int = 0
    actions_removed: int = 0
    actions_modified: int = 0
    budget_change: float = 0.0
    expected_additional_reduction: float = 0.0
    cost_unit: str = "USD"
    emission_unit: str = "kg-CO2e"


@dataclass
class AdjustedAnnualPlan:
    """Adjusted annual plan for reoptimization"""
    year: int
    original_actions: List[Dict[str, Any]] = field(default_factory=list)
    recommended_actions: List[AdjustedAction] = field(default_factory=list)
    changes_from_original: Optional[ChangesFromOriginal] = None


# =============================================================================
# IMPACT ANALYSIS MODELS
# =============================================================================

@dataclass
class ImpactAnalysis:
    """Analysis of reoptimization impact"""
    long_term_target_impact: str = ""
    financial_impact: str = ""
    risk_impact: str = ""
    resource_impact: Optional[str] = None
    timeline_impact: Optional[str] = None


# =============================================================================
# REOPTIMIZATION PATTERN MODELS
# =============================================================================

@dataclass
class ReoptimizationPattern:
    """Complete reoptimization strategy"""
    pattern_id: str = ""
    pattern_name: str = ""
    rationale: str = ""
    adjusted_annual_plan: List[AdjustedAnnualPlan] = field(default_factory=list)
    impact_analysis: Optional[ImpactAnalysis] = None
    performance_comparison: Optional[PerformanceComparison] = None
    plan_id: str = ""
    
    def __post_init__(self):
        if not self.pattern_id:
            from models.base import generate_uuid
            self.pattern_id = generate_uuid()


@dataclass
class ReoptimizationResult:
    """Complete reoptimization calculation result"""
    calculation_id: str = ""
    reoptimization_patterns: List[ReoptimizationPattern] = field(default_factory=list)
    calculation_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    
    def __post_init__(self):
        if not self.calculation_id:
            from models.base import generate_uuid
            self.calculation_id = generate_uuid()
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


# =============================================================================
# REGISTRATION MODELS
# =============================================================================

@dataclass
class ReoptimizationApprovalInfo:
    """Approval information for reoptimization registration"""
    approved_by: str
    approval_date: str
    comments: Optional[str] = None


@dataclass
class ReoptimizationRegistrationRequest:
    """Request to register reoptimization"""
    reoptimization_pattern_id: str
    plan_id: str
    approval_info: ReoptimizationApprovalInfo


@dataclass
class ReoptimizationRegistrationResult:
    """Result of reoptimization registration"""
    reoptimization_id: str = ""
    updated_plan_id: str = ""
    registration_timestamp: str = ""
    status: str = "active"
    changes_applied: Optional[ChangesFromOriginal] = None
    
    def __post_init__(self):
        if not self.reoptimization_id:
            from models.base import generate_uuid
            self.reoptimization_id = generate_uuid()
        if not self.registration_timestamp:
            self.registration_timestamp = datetime.utcnow().isoformat()


# =============================================================================
# VISUALIZATION MODELS
# =============================================================================

@dataclass
class MonthlyPerformanceData:
    """Monthly performance data for visualization"""
    month: str  # Format: YYYY-MM
    actual_emission: float
    target_emission: float
    variance: float
    emission_unit: str = "kg-CO2e"


@dataclass
class PerformanceTrends:
    """Performance trends over time"""
    monthly_data: List[MonthlyPerformanceData] = field(default_factory=list)
    overall_trend: str = "stable"  # improving, stable, declining
    trend_analysis: Optional[str] = None


@dataclass
class PlanComparison:
    """Comparison of original vs adjusted plan"""
    original_total_reduction: float
    adjusted_total_reduction: float
    difference: float
    emission_unit: str = "kg-CO2e"
    original_total_cost: float = 0.0
    adjusted_total_cost: float = 0.0
    cost_difference: float = 0.0
    cost_unit: str = "USD"


@dataclass
class KeyMetrics:
    """Key performance metrics"""
    target_achievement_rate: float
    cost_efficiency_improvement: float
    recommended_adjustments: int
    implementation_progress: float


@dataclass
class ReoptimizationVisualizationData:
    """Complete visualization data for reoptimization"""
    property_id: str
    year: int
    performance_trends: Optional[PerformanceTrends] = None
    comparison_charts: Optional[Dict[str, PlanComparison]] = None
    key_metrics: Optional[KeyMetrics] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_default_reoptimization_request(plan_id: str, target_year: int) -> ReoptimizationRequest:
    """Create a default reoptimization request"""
    return ReoptimizationRequest(
        plan_id=plan_id,
        target_year=target_year,
        start_date=f"{target_year-1}-01-01",
        end_date=f"{target_year-1}-12-31",
        frequency="quarterly"
    )


def validate_reoptimization_request(request: ReoptimizationRequest) -> List[str]:
    """Validate reoptimization request and return list of errors"""
    errors = []
    
    if not request.plan_id:
        errors.append("plan_id is required")
    
    if request.target_year < 2020 or request.target_year > 2100:
        errors.append("target_year must be between 2020 and 2100")
    
    try:
        start = datetime.fromisoformat(request.start_date)
        end = datetime.fromisoformat(request.end_date)
        if end <= start:
            errors.append("end_date must be after start_date")
    except ValueError:
        errors.append("Invalid date format (use YYYY-MM-DD)")
    
    return errors


def calculate_variance_percentage(actual: float, planned: float) -> float:
    """Calculate variance percentage"""
    if planned == 0:
        return 0.0
    return ((actual - planned) / planned) * 100


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'ReoptimizationStatus',
    'FrequencyType',
    'PerformanceStatus',
    'ActionChange',
    
    # Request models
    'BudgetAdjustment',
    'ReoptimizationRequest',
    
    # Performance models
    'PerformanceMetrics',
    'CostMetrics',
    'PerformanceComparison',
    'HistoricalPerformance',
    
    # Action models
    'AdjustedAction',
    'ChangesFromOriginal',
    'AdjustedAnnualPlan',
    
    # Impact models
    'ImpactAnalysis',
    
    # Result models
    'ReoptimizationPattern',
    'ReoptimizationResult',
    'ReoptimizationRegistrationRequest',
    'ReoptimizationRegistrationResult',
    'ReoptimizationApprovalInfo',
    
    # Visualization models
    'MonthlyPerformanceData',
    'PerformanceTrends',
    'PlanComparison',
    'KeyMetrics',
    'ReoptimizationVisualizationData',
    
    # Helper functions
    'create_default_reoptimization_request',
    'validate_reoptimization_request',
    'calculate_variance_percentage'
]
