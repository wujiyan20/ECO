# models/reduction.py - Reduction Options and Strategy Models
"""
CO2 reduction options, strategies, and implementation models.
Handles reduction measures, cost-benefit analysis, and implementation planning.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from .base import (
    BaseModel,
    AuditableModel,
    validate_positive_number,
    calculate_percentage_change,
    ValidationError
)
from .enums import (
    StrategyType,
    PriorityLevel,
    RiskLevel,
    ImplementationStatus,
    EmissionScope
)

# =============================================================================
# REDUCTION OPTION MODELS
# =============================================================================

@dataclass
class ReductionOption(AuditableModel):
    """
    Specific reduction measure or technology option
    """
    option_id: str
    name: str
    description: Optional[str] = None
    strategy_type: StrategyType = StrategyType.ENERGY_EFFICIENCY
    
    # Impact metrics
    annual_co2_reduction: float = 0.0  # kg-CO2e per year
    reduction_percentage: float = 0.0  # % of baseline
    emission_scope: EmissionScope = EmissionScope.SCOPE_2
    
    # Financial data
    capex: float = 0.0  # Capital expenditure (AUD)
    annual_opex: float = 0.0  # Operating expenditure per year (AUD)
    annual_savings: float = 0.0  # Energy cost savings per year (AUD)
    
    # Implementation
    implementation_time_months: int = 0
    expected_lifetime_years: int = 0
    maintenance_frequency_months: int = 12
    
    # Priority and risk
    priority: PriorityLevel = PriorityLevel.MEDIUM
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    # Calculated metrics
    simple_payback_years: float = 0.0
    npv: float = 0.0  # Net Present Value
    irr: float = 0.0  # Internal Rate of Return
    cost_per_tonne_co2: float = 0.0
    
    # Applicability
    applicable_property_types: List[str] = field(default_factory=list)
    applicable_property_ids: List[str] = field(default_factory=list)
    
    # Dependencies
    prerequisites: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    
    # Additional benefits
    co_benefits: List[str] = field(default_factory=list)  # e.g., "Improved comfort", "Health benefits"
    
    # Implementation status
    status: ImplementationStatus = ImplementationStatus.PLANNED
    implementation_progress: float = 0.0  # 0-100%
    
    # Metadata
    data_source: Optional[str] = None
    data_quality: str = "Medium"
    last_reviewed: Optional[datetime] = None
    
    def calculate_simple_payback(self):
        """Calculate simple payback period in years"""
        if self.annual_savings > 0:
            self.simple_payback_years = self.capex / self.annual_savings
        else:
            self.simple_payback_years = 999.0  # Effectively infinite
    
    def calculate_cost_per_tonne(self):
        """Calculate cost per tonne of CO2 reduced"""
        if self.annual_co2_reduction > 0:
            # Convert to tonnes
            annual_reduction_tonnes = self.annual_co2_reduction / 1000
            # Total cost over lifetime
            total_cost = self.capex + (self.annual_opex * self.expected_lifetime_years)
            # Total reduction over lifetime
            total_reduction = annual_reduction_tonnes * self.expected_lifetime_years
            self.cost_per_tonne_co2 = total_cost / total_reduction if total_reduction > 0 else 0
    
    def calculate_npv(self, discount_rate: float = 0.05):
        """
        Calculate Net Present Value
        
        Args:
            discount_rate: Annual discount rate (default 5%)
        """
        if self.expected_lifetime_years == 0:
            self.npv = -self.capex
            return
        
        # Initial investment (negative)
        npv = -self.capex
        
        # Add discounted cash flows for each year
        for year in range(1, self.expected_lifetime_years + 1):
            annual_net_savings = self.annual_savings - self.annual_opex
            discount_factor = (1 + discount_rate) ** year
            npv += annual_net_savings / discount_factor
        
        self.npv = npv
    
    def is_financially_viable(self, max_payback_years: float = 10.0) -> bool:
        """Check if option is financially viable"""
        return (
            self.simple_payback_years <= max_payback_years and
            self.npv > 0
        )
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate reduction option"""
        is_valid, errors = super().validate()
        
        # Validate reduction
        if self.annual_co2_reduction < 0:
            errors.append("CO2 reduction cannot be negative")
        
        # Validate costs
        if self.capex < 0:
            errors.append("CAPEX cannot be negative")
        if self.annual_opex < 0:
            errors.append("Annual OPEX cannot be negative")
        if self.annual_savings < 0:
            errors.append("Annual savings cannot be negative")
        
        # Validate timeline
        if self.implementation_time_months < 0:
            errors.append("Implementation time cannot be negative")
        if self.expected_lifetime_years <= 0:
            errors.append("Expected lifetime must be positive")
        
        return len(errors) == 0, errors

@dataclass
class ReductionStrategy(BaseModel):
    """
    Collection of reduction options forming a strategy
    """
    strategy_id: str
    name: str
    description: Optional[str] = None
    strategy_type: StrategyType = StrategyType.ENERGY_EFFICIENCY
    
    # Included options
    option_ids: List[str] = field(default_factory=list)
    options: List[ReductionOption] = field(default_factory=list)
    
    # Aggregated metrics
    total_co2_reduction: float = 0.0
    total_capex: float = 0.0
    total_annual_opex: float = 0.0
    total_annual_savings: float = 0.0
    
    # Timeline
    start_date: Optional[datetime] = None
    expected_completion_date: Optional[datetime] = None
    
    # Allocation
    properties_allocated: List[str] = field(default_factory=list)
    allocation_percentage: float = 0.0  # % of total portfolio strategy
    
    def calculate_aggregated_metrics(self):
        """Calculate total metrics from all options"""
        self.total_co2_reduction = sum(opt.annual_co2_reduction for opt in self.options)
        self.total_capex = sum(opt.capex for opt in self.options)
        self.total_annual_opex = sum(opt.annual_opex for opt in self.options)
        self.total_annual_savings = sum(opt.annual_savings for opt in self.options)
    
    def get_high_priority_options(self) -> List[ReductionOption]:
        """Get high priority options in this strategy"""
        return [
            opt for opt in self.options
            if opt.priority in [PriorityLevel.VERY_HIGH, PriorityLevel.HIGH]
        ]
    
    def get_quick_wins(self, max_payback_years: float = 3.0) -> List[ReductionOption]:
        """Get options with quick payback (quick wins)"""
        return [
            opt for opt in self.options
            if opt.simple_payback_years <= max_payback_years and opt.simple_payback_years > 0
        ]

# =============================================================================
# IMPLEMENTATION PLANNING MODELS
# =============================================================================

@dataclass
class ImplementationPhase:
    """
    Phase in implementation timeline
    """
    phase_id: str
    phase_number: int
    name: str
    description: Optional[str] = None
    
    # Timeline
    start_date: datetime
    end_date: datetime
    duration_months: int = 0
    
    # Options in this phase
    option_ids: List[str] = field(default_factory=list)
    
    # Metrics
    phase_capex: float = 0.0
    phase_co2_reduction: float = 0.0
    
    # Status
    status: ImplementationStatus = ImplementationStatus.PLANNED
    completion_percentage: float = 0.0
    
    def calculate_duration(self):
        """Calculate phase duration in months"""
        delta = self.end_date - self.start_date
        self.duration_months = int(delta.days / 30)

@dataclass
class ImplementationPlan(AuditableModel):
    """
    Complete implementation plan with phased rollout
    """
    plan_id: str
    name: str
    description: Optional[str] = None
    scenario_id: str
    
    # Timeline
    start_date: datetime
    end_date: datetime
    
    # Phases
    phases: List[ImplementationPhase] = field(default_factory=list)
    
    # All options
    all_options: List[ReductionOption] = field(default_factory=list)
    
    # Totals
    total_options: int = 0
    total_capex: float = 0.0
    total_annual_reduction: float = 0.0
    
    # Resource requirements
    required_budget: float = 0.0
    required_personnel: int = 0
    external_contractors: List[str] = field(default_factory=list)
    
    # Risk assessment
    overall_risk_level: RiskLevel = RiskLevel.MEDIUM
    key_risks: List[Dict[str, str]] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    
    # Progress
    overall_progress: float = 0.0  # 0-100%
    completed_options: int = 0
    in_progress_options: int = 0
    
    def add_phase(self, phase: ImplementationPhase):
        """Add phase to plan"""
        self.phases.append(phase)
        self.phases.sort(key=lambda p: p.phase_number)
    
    def calculate_totals(self):
        """Calculate total metrics"""
        self.total_options = len(self.all_options)
        self.total_capex = sum(opt.capex for opt in self.all_options)
        self.total_annual_reduction = sum(opt.annual_co2_reduction for opt in self.all_options)
    
    def calculate_progress(self):
        """Calculate overall implementation progress"""
        if not self.all_options:
            self.overall_progress = 0.0
            return
        
        total_weight = len(self.all_options)
        weighted_progress = sum(opt.implementation_progress for opt in self.all_options)
        self.overall_progress = weighted_progress / total_weight
        
        # Count status
        self.completed_options = sum(
            1 for opt in self.all_options
            if opt.status == ImplementationStatus.COMPLETED
        )
        self.in_progress_options = sum(
            1 for opt in self.all_options
            if opt.status == ImplementationStatus.IN_PROGRESS
        )

@dataclass
class ReductionRate:
    """
    Custom reduction rate for specific years
    """
    year: int
    target_reduction_percentage: float  # % from baseline
    target_emission: float = 0.0  # kg-CO2e
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate reduction rate"""
        errors = []
        
        # Validate year
        current_year = datetime.now().year
        if self.year < current_year:
            errors.append("Year cannot be in the past")
        
        # Validate percentage
        if not (0 <= self.target_reduction_percentage <= 100):
            errors.append("Reduction percentage must be between 0 and 100")
        
        return len(errors) == 0, errors

@dataclass
class StrategyBreakdown:
    """
    Breakdown of strategy allocation
    """
    strategy_type: StrategyType
    allocation_percentage: float  # % of total reduction effort
    estimated_reduction: float  # kg-CO2e
    estimated_cost: float  # AUD
    
    # Specific measures
    measures: List[str] = field(default_factory=list)
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate strategy breakdown"""
        errors = []
        
        if not (0 <= self.allocation_percentage <= 100):
            errors.append("Allocation percentage must be between 0 and 100")
        
        if self.estimated_reduction < 0:
            errors.append("Estimated reduction cannot be negative")
        
        if self.estimated_cost < 0:
            errors.append("Estimated cost cannot be negative")
        
        return len(errors) == 0, errors

# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'ReductionOption',
    'ReductionStrategy',
    'ImplementationPhase',
    'ImplementationPlan',
    'ReductionRate',
    'StrategyBreakdown'
]
