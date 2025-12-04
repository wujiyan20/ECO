# models/planning.py - Long-term Planning Data Models
"""
Data models for long-term planning functionality.

Models:
- PlanningHorizon: Time period for planning
- BudgetConstraints: Financial constraints
- StrategyPreferences: Strategy priority weights
- ImplementationConstraints: Physical/operational limits
- ActionPlan: Individual reduction action
- AnnualPlan: Plan for a specific year
- FinancialSummary: Cost and ROI summary
- RiskAssessment: Risk analysis
- PlanningPattern: Complete planning strategy
- PlanningRequest: Request for planning calculation
- PlanningResult: Complete planning result
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class PlanningStatus(Enum):
    """Status of planning calculation"""
    DRAFT = "draft"
    CALCULATED = "calculated"
    REGISTERED = "registered"
    ACTIVE = "active"
    ARCHIVED = "archived"


class ActionType(Enum):
    """Types of reduction actions"""
    SOLAR_PANEL = "solar_panel_installation"
    HVAC_UPGRADE = "hvac_upgrade"
    LED_LIGHTING = "led_lighting_upgrade"
    INSULATION = "building_insulation"
    SMART_CONTROLS = "smart_controls"
    RENEWABLE_ENERGY = "renewable_energy_procurement"
    BEHAVIORAL = "behavioral_change_program"
    OTHER = "other"


class PriorityLevel(Enum):
    """Priority levels for actions"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskLevel(Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# PLANNING REQUEST MODELS
# =============================================================================

@dataclass
class PlanningHorizon:
    """Planning time period definition"""
    start_year: int
    end_year: int
    evaluation_intervals: str = "annual"
    
    def __post_init__(self):
        if self.end_year <= self.start_year:
            raise ValueError("end_year must be after start_year")


@dataclass
class BudgetConstraints:
    """Financial constraints for planning"""
    total_budget: Optional[float] = None
    annual_budget_limit: Optional[float] = None
    currency: str = "USD"
    cost_escalation_rate: float = 3.0
    
    def __post_init__(self):
        if self.cost_escalation_rate < 0:
            raise ValueError("cost_escalation_rate cannot be negative")


@dataclass
class StrategyPreferences:
    """Strategy priority weights"""
    renewable_energy_priority: float = 0.33
    energy_efficiency_priority: float = 0.33
    behavioral_change_priority: float = 0.34
    
    def __post_init__(self):
        # Normalize to sum to 1.0
        total = (self.renewable_energy_priority + 
                 self.energy_efficiency_priority + 
                 self.behavioral_change_priority)
        if total > 0:
            self.renewable_energy_priority /= total
            self.energy_efficiency_priority /= total
            self.behavioral_change_priority /= total


@dataclass
class ImplementationConstraints:
    """Physical and operational constraints"""
    max_simultaneous_projects: Optional[int] = None
    required_roi_years: Optional[float] = None
    minimum_reduction_per_action: Optional[float] = None
    technology_restrictions: List[str] = field(default_factory=list)


@dataclass
class PlanningRequest:
    """Request for long-term planning calculation"""
    scenario_id: str
    allocation_id: str
    planning_horizon: PlanningHorizon
    budget_constraints: Optional[BudgetConstraints] = None
    strategy_preferences: Optional[StrategyPreferences] = None
    implementation_constraints: Optional[ImplementationConstraints] = None


# =============================================================================
# ACTION AND PLAN MODELS
# =============================================================================

@dataclass
class ActionPlan:
    """Individual reduction action"""
    action_id: str = ""
    action_type: str = "other"
    target_properties: List[str] = field(default_factory=list)
    expected_reduction: float = 0.0
    investment_required: float = 0.0
    roi_years: float = 0.0
    priority: str = "medium"
    rationale: str = ""
    emission_unit: str = "kg-CO2e"
    cost_unit: str = "USD"
    implementation_duration_months: int = 12
    
    def __post_init__(self):
        if not self.action_id:
            from models.base import generate_uuid
            self.action_id = generate_uuid()


@dataclass
class AnnualPlan:
    """Plan for a specific year"""
    year: int
    actions: List[ActionPlan] = field(default_factory=list)
    total_investment: float = 0.0
    total_reduction: float = 0.0
    cumulative_progress: float = 0.0
    cost_unit: str = "USD"
    emission_unit: str = "kg-CO2e"
    
    def calculate_totals(self):
        """Calculate total investment and reduction from actions"""
        self.total_investment = sum(a.investment_required for a in self.actions)
        self.total_reduction = sum(a.expected_reduction for a in self.actions)


@dataclass
class FinancialSummary:
    """Financial summary for planning pattern"""
    total_investment: float = 0.0
    total_savings: float = 0.0
    net_benefit: float = 0.0
    overall_roi_years: float = 0.0
    currency: str = "USD"
    payback_period_years: Optional[float] = None
    npv: Optional[float] = None
    irr: Optional[float] = None


@dataclass
class RiskAssessment:
    """Risk assessment for planning pattern"""
    implementation_risk: str = "medium"
    financial_risk: str = "medium"
    technology_risk: str = "medium"
    regulatory_risk: str = "low"
    overall_risk_level: str = "medium"
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)


# =============================================================================
# PLANNING PATTERN AND RESULT MODELS
# =============================================================================

@dataclass
class PlanningPattern:
    """Complete planning strategy"""
    pattern_id: str = ""
    pattern_name: str = ""
    description: str = ""
    annual_plan: List[AnnualPlan] = field(default_factory=list)
    financial_summary: Optional[FinancialSummary] = None
    risk_assessment: Optional[RiskAssessment] = None
    scenario_id: str = ""
    allocation_id: str = ""
    
    def __post_init__(self):
        if not self.pattern_id:
            from models.base import generate_uuid
            self.pattern_id = generate_uuid()


@dataclass
class PlanningResult:
    """Complete planning calculation result"""
    calculation_id: str = ""
    planning_patterns: List[PlanningPattern] = field(default_factory=list)
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
class ApprovalInfo:
    """Approval information for registration"""
    approved_by: str
    approval_date: str
    comments: Optional[str] = None


@dataclass
class RegistrationRequest:
    """Request to register a planning pattern"""
    pattern_id: str
    plan_name: str
    approval_info: ApprovalInfo


@dataclass
class RegistrationResult:
    """Result of planning registration"""
    plan_id: str = ""
    registration_timestamp: str = ""
    status: str = "registered"
    next_steps: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.plan_id:
            from models.base import generate_uuid
            self.plan_id = generate_uuid()
        if not self.registration_timestamp:
            self.registration_timestamp = datetime.utcnow().isoformat()


# =============================================================================
# VISUALIZATION MODELS
# =============================================================================

@dataclass
class TimelineDataPoint:
    """Data point for timeline visualization"""
    year: int
    cumulative_reduction: float
    cumulative_investment: float
    cumulative_savings: float
    emission_unit: str = "kg-CO2e"
    cost_unit: str = "USD"


@dataclass
class ActionDistribution:
    """Distribution of actions by type and year"""
    by_type: Dict[str, int] = field(default_factory=dict)
    by_year: Dict[str, int] = field(default_factory=dict)
    by_priority: Dict[str, int] = field(default_factory=dict)


@dataclass
class CostBenefitAnalysis:
    """Cost-benefit analysis data"""
    breakeven_year: Optional[int] = None
    total_roi_percentage: float = 0.0
    npv: float = 0.0
    currency: str = "USD"


@dataclass
class VisualizationData:
    """Complete visualization data for a planning pattern"""
    pattern_id: str
    pattern_name: str
    timeline_data: List[TimelineDataPoint] = field(default_factory=list)
    action_distribution: Optional[ActionDistribution] = None
    cost_benefit_analysis: Optional[CostBenefitAnalysis] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_default_planning_request(scenario_id: str, allocation_id: str) -> PlanningRequest:
    """Create a default planning request"""
    return PlanningRequest(
        scenario_id=scenario_id,
        allocation_id=allocation_id,
        planning_horizon=PlanningHorizon(
            start_year=2025,
            end_year=2050
        )
    )


def validate_planning_request(request: PlanningRequest) -> List[str]:
    """Validate planning request and return list of errors"""
    errors = []
    
    if not request.scenario_id:
        errors.append("scenario_id is required")
    
    if not request.allocation_id:
        errors.append("allocation_id is required")
    
    if request.planning_horizon.end_year <= request.planning_horizon.start_year:
        errors.append("planning_horizon.end_year must be after start_year")
    
    if request.budget_constraints:
        if request.budget_constraints.total_budget is not None:
            if request.budget_constraints.total_budget < 0:
                errors.append("total_budget cannot be negative")
    
    return errors


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'PlanningStatus',
    'ActionType',
    'PriorityLevel',
    'RiskLevel',
    
    # Request models
    'PlanningHorizon',
    'BudgetConstraints',
    'StrategyPreferences',
    'ImplementationConstraints',
    'PlanningRequest',
    
    # Action and plan models
    'ActionPlan',
    'AnnualPlan',
    'FinancialSummary',
    'RiskAssessment',
    
    # Result models
    'PlanningPattern',
    'PlanningResult',
    'RegistrationRequest',
    'RegistrationResult',
    'ApprovalInfo',
    
    # Visualization models
    'TimelineDataPoint',
    'ActionDistribution',
    'CostBenefitAnalysis',
    'VisualizationData',
    
    # Helper functions
    'create_default_planning_request',
    'validate_planning_request'
]
