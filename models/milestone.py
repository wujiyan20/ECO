# models/milestone.py - Milestone and Scenario Data Models
"""
Milestone planning and scenario models for the EcoAssist system.
Handles milestone targets, scenarios, and long-term planning.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base import (
    BaseModel,
    AuditableModel,
    validate_year_range,
    validate_percentage,
    calculate_percentage_change,
    ValidationError
)
from .enums import (
    ScenarioType,
    ApprovalStatus,
    AllocationMethod,
    OnTrackStatus
)

# =============================================================================
# MILESTONE TARGET MODELS
# =============================================================================

@dataclass
class MilestoneTarget(BaseModel):
    """
    Single milestone target for a specific year
    """
    target_id: str
    property_id: Optional[str] = None
    year: int = 0
    
    # Target values
    target_emission: float = 0.0  # kg-CO2e
    baseline_emission: float = 0.0  # kg-CO2e
    reduction_amount: float = 0.0  # kg-CO2e
    reduction_percentage: float = 0.0  # %
    
    # Breakdown by scope
    scope1_target: float = 0.0
    scope2_target: float = 0.0
    scope3_target: Optional[float] = 0.0
    
    # Metadata
    target_type: str = "Absolute"  # "Absolute" or "Intensity"
    unit: str = "kg-CO2e"
    confidence_level: float = 0.85
    
    def calculate_reduction(self):
        """Calculate reduction amount and percentage"""
        self.reduction_amount = self.baseline_emission - self.target_emission
        if self.baseline_emission > 0:
            self.reduction_percentage = (self.reduction_amount / self.baseline_emission) * 100
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate milestone target"""
        is_valid, errors = super().validate()
        
        # Validate year
        current_year = datetime.now().year
        if self.year < current_year:
            errors.append(f"Target year {self.year} is in the past")
        
        # Validate emissions
        if self.target_emission < 0:
            errors.append("Target emission cannot be negative")
        if self.baseline_emission < 0:
            errors.append("Baseline emission cannot be negative")
        
        # Validate reduction
        if self.reduction_percentage < 0:
            errors.append("Reduction percentage cannot be negative")
        if self.reduction_percentage > 100:
            errors.append("Reduction percentage cannot exceed 100%")
        
        return len(errors) == 0, errors

@dataclass
class MilestoneScenario(AuditableModel):
    """
    Complete milestone scenario with yearly targets
    """
    scenario_id: str
    scenario_name: str
    description: Optional[str] = None
    scenario_type: ScenarioType = ScenarioType.STANDARD
    
    # Time period
    base_year: int = 0
    mid_term_year: int = 2030
    long_term_year: int = 2050
    
    # Baseline
    baseline_emission: float = 0.0  # kg-CO2e
    
    # Key targets
    mid_term_target: float = 0.0  # kg-CO2e for 2030
    mid_term_reduction_percentage: float = 0.0
    long_term_target: float = 0.0  # kg-CO2e for 2050
    long_term_reduction_percentage: float = 0.0
    
    # Yearly targets (interpolated)
    yearly_targets: Dict[int, float] = field(default_factory=dict)  # year: emission
    yearly_reductions: Dict[int, float] = field(default_factory=dict)  # year: reduction %
    
    # Cost projections
    total_capex: float = 0.0
    total_opex: float = 0.0
    cumulative_cost: float = 0.0
    cost_per_tonne_reduced: float = 0.0
    
    # Strategy breakdown
    strategy_allocation: Dict[str, float] = field(default_factory=dict)  # strategy: %
    
    # Performance metrics
    feasibility_score: float = 0.0  # 0-100
    risk_score: float = 0.0  # 0-100
    cost_efficiency_score: float = 0.0  # 0-100
    overall_score: float = 0.0  # 0-100
    
    # Approval workflow
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT
    approved_by: Optional[str] = None
    approved_date: Optional[datetime] = None
    comments: Optional[str] = None
    
    # Properties included
    property_ids: List[str] = field(default_factory=list)
    property_count: int = 0
    
    def calculate_reduction_percentages(self):
        """Calculate reduction percentages for key milestones"""
        if self.baseline_emission > 0:
            self.mid_term_reduction_percentage = (
                (self.baseline_emission - self.mid_term_target) / self.baseline_emission * 100
            )
            self.long_term_reduction_percentage = (
                (self.baseline_emission - self.long_term_target) / self.baseline_emission * 100
            )
    
    def interpolate_yearly_targets(self):
        """
        Interpolate yearly targets between key milestones
        Creates smooth transition path
        """
        if not self.baseline_emission or not self.mid_term_target or not self.long_term_target:
            return
        
        # Clear existing
        self.yearly_targets = {}
        self.yearly_reductions = {}
        
        # Base year
        self.yearly_targets[self.base_year] = self.baseline_emission
        self.yearly_reductions[self.base_year] = 0.0
        
        # Interpolate to mid-term
        years_to_midterm = self.mid_term_year - self.base_year
        annual_reduction = (self.baseline_emission - self.mid_term_target) / years_to_midterm
        
        for year in range(self.base_year + 1, self.mid_term_year):
            years_elapsed = year - self.base_year
            target = self.baseline_emission - (annual_reduction * years_elapsed)
            self.yearly_targets[year] = target
            self.yearly_reductions[year] = ((self.baseline_emission - target) / self.baseline_emission * 100)
        
        # Mid-term milestone
        self.yearly_targets[self.mid_term_year] = self.mid_term_target
        self.yearly_reductions[self.mid_term_year] = self.mid_term_reduction_percentage
        
        # Interpolate to long-term
        years_to_longterm = self.long_term_year - self.mid_term_year
        annual_reduction = (self.mid_term_target - self.long_term_target) / years_to_longterm
        
        for year in range(self.mid_term_year + 1, self.long_term_year):
            years_elapsed = year - self.mid_term_year
            target = self.mid_term_target - (annual_reduction * years_elapsed)
            self.yearly_targets[year] = target
            self.yearly_reductions[year] = ((self.baseline_emission - target) / self.baseline_emission * 100)
        
        # Long-term milestone
        self.yearly_targets[self.long_term_year] = self.long_term_target
        self.yearly_reductions[self.long_term_year] = self.long_term_reduction_percentage
    
    def calculate_cost_efficiency(self):
        """Calculate cost per tonne of CO2 reduced"""
        total_reduction = self.baseline_emission - self.long_term_target
        if total_reduction > 0:
            # Convert to tonnes
            total_reduction_tonnes = total_reduction / 1000
            self.cost_per_tonne_reduced = self.cumulative_cost / total_reduction_tonnes
    
    def calculate_overall_score(self):
        """Calculate overall scenario score"""
        # Weighted average of different scores
        weights = {
            'feasibility': 0.35,
            'cost_efficiency': 0.35,
            'risk': 0.30  # Lower risk is better, so invert
        }
        
        risk_score_inverted = 100 - self.risk_score
        
        self.overall_score = (
            self.feasibility_score * weights['feasibility'] +
            self.cost_efficiency_score * weights['cost_efficiency'] +
            risk_score_inverted * weights['risk']
        )
    
    def get_target_for_year(self, year: int) -> Optional[float]:
        """Get emission target for specific year"""
        return self.yearly_targets.get(year)
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate milestone scenario"""
        is_valid, errors = super().validate()
        
        # Validate years
        try:
            validate_year_range(self.base_year, self.mid_term_year)
            validate_year_range(self.mid_term_year, self.long_term_year)
        except ValidationError as e:
            errors.append(str(e))
        
        # Validate targets
        if self.mid_term_target > self.baseline_emission:
            errors.append("Mid-term target cannot exceed baseline")
        if self.long_term_target > self.mid_term_target:
            errors.append("Long-term target cannot exceed mid-term target")
        
        # Validate reductions
        try:
            validate_percentage(self.mid_term_reduction_percentage)
            validate_percentage(self.long_term_reduction_percentage)
        except ValidationError as e:
            errors.append(str(e))
        
        if self.long_term_reduction_percentage < self.mid_term_reduction_percentage:
            errors.append("Long-term reduction must be greater than mid-term")
        
        return len(errors) == 0, errors

@dataclass
class ScenarioComparison:
    """
    Comparison between multiple scenarios
    """
    comparison_id: str
    base_year: int
    scenarios: List[MilestoneScenario] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)
    
    def get_best_scenario(self, criteria: str = "overall_score") -> Optional[MilestoneScenario]:
        """Get best scenario based on criteria"""
        if not self.scenarios:
            return None
        
        if criteria == "overall_score":
            return max(self.scenarios, key=lambda s: s.overall_score)
        elif criteria == "cost":
            return min(self.scenarios, key=lambda s: s.cumulative_cost)
        elif criteria == "reduction":
            return max(self.scenarios, key=lambda s: s.long_term_reduction_percentage)
        elif criteria == "feasibility":
            return max(self.scenarios, key=lambda s: s.feasibility_score)
        
        return None
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """Get summary comparison of all scenarios"""
        if not self.scenarios:
            return {}
        
        return {
            'scenario_count': len(self.scenarios),
            'scenarios': [
                {
                    'name': s.scenario_name,
                    'type': s.scenario_type.value,
                    'mid_term_reduction': s.mid_term_reduction_percentage,
                    'long_term_reduction': s.long_term_reduction_percentage,
                    'total_cost': s.cumulative_cost,
                    'cost_per_tonne': s.cost_per_tonne_reduced,
                    'overall_score': s.overall_score,
                    'risk_score': s.risk_score
                }
                for s in self.scenarios
            ]
        }

# =============================================================================
# MILESTONE PROGRESS TRACKING
# =============================================================================

@dataclass
class MilestoneProgress:
    """
    Track progress against milestone targets
    """
    progress_id: str
    scenario_id: str
    property_id: Optional[str] = None
    tracking_date: datetime = field(default_factory=datetime.now)
    
    # Current status
    current_year: int = 0
    target_emission: float = 0.0
    actual_emission: float = 0.0
    variance: float = 0.0
    variance_percentage: float = 0.0
    
    # Progress metrics
    on_track_status: OnTrackStatus = OnTrackStatus.ON_TRACK
    cumulative_reduction_achieved: float = 0.0
    cumulative_reduction_target: float = 0.0
    progress_percentage: float = 0.0  # % of target achieved
    
    # Future outlook
    projected_year_end_emission: float = 0.0
    risk_of_missing_target: bool = False
    recommended_actions: List[str] = field(default_factory=list)
    
    def calculate_variance(self):
        """Calculate variance between target and actual"""
        self.variance = self.actual_emission - self.target_emission
        if self.target_emission > 0:
            self.variance_percentage = (self.variance / self.target_emission) * 100
        
        # Determine on-track status
        if self.variance_percentage <= -5:  # 5% better than target
            self.on_track_status = OnTrackStatus.AHEAD
        elif self.variance_percentage <= 5:  # Within 5% of target
            self.on_track_status = OnTrackStatus.ON_TRACK
        elif self.variance_percentage <= 15:  # 5-15% worse than target
            self.on_track_status = OnTrackStatus.AT_RISK
        elif self.variance_percentage <= 25:  # 15-25% worse than target
            self.on_track_status = OnTrackStatus.OFF_TRACK
        else:  # More than 25% worse than target
            self.on_track_status = OnTrackStatus.CRITICAL
    
    def calculate_progress_percentage(self):
        """Calculate overall progress towards target"""
        if self.cumulative_reduction_target > 0:
            self.progress_percentage = (
                self.cumulative_reduction_achieved / self.cumulative_reduction_target * 100
            )

@dataclass
class MilestoneAlert:
    """
    Alert for milestone tracking issues
    """
    alert_id: str
    scenario_id: str
    property_id: Optional[str] = None
    alert_type: str  # "variance", "risk", "delay"
    severity: str  # "info", "warning", "critical"
    
    title: str
    description: str
    detected_date: datetime = field(default_factory=datetime.now)
    
    # Metrics
    current_value: float = 0.0
    target_value: float = 0.0
    variance: float = 0.0
    
    # Actions
    recommended_actions: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None
    status: str = "open"  # "open", "acknowledged", "resolved"
    resolved_date: Optional[datetime] = None

# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'MilestoneTarget',
    'MilestoneScenario',
    'ScenarioComparison',
    'MilestoneProgress',
    'MilestoneAlert'
]
