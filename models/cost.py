# models/cost.py - Cost and Financial Models
"""
Cost projection and financial analysis models for the EcoAssist system.
Handles CAPEX, OPEX, ROI calculations, and financial projections.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base import (
    BaseModel,
    validate_positive_number,
    ValidationError
)
from .enums import (
    CostCategory,
    Currency
)

# =============================================================================
# COST PROJECTION MODELS
# =============================================================================

@dataclass
class CostProjection(BaseModel):
    """
    Cost projection for a specific year or period
    """
    projection_id: str = ""
    year: int = 0
    property_id: Optional[str] = None
    scenario_id: Optional[str] = None
    
    # Capital expenditure
    capex_total: float = 0.0
    capex_energy_efficiency: float = 0.0
    capex_renewable_energy: float = 0.0
    capex_building_retrofit: float = 0.0
    capex_technology_upgrade: float = 0.0
    capex_other: float = 0.0
    
    # Operating expenditure
    opex_total: float = 0.0
    opex_maintenance: float = 0.0
    opex_energy: float = 0.0
    opex_monitoring: float = 0.0
    opex_staff: float = 0.0
    opex_consulting: float = 0.0
    opex_other: float = 0.0
    
    # Savings
    energy_cost_savings: float = 0.0
    operational_savings: float = 0.0
    avoided_costs: float = 0.0  # e.g., carbon tax
    incentives_received: float = 0.0
    total_savings: float = 0.0
    
    # Net cost
    net_cost: float = 0.0  # CAPEX + OPEX - Savings
    cumulative_cost: float = 0.0
    
    # Currency
    currency: Currency = Currency.AUD
    
    # Confidence
    confidence_level: float = 0.75
    uncertainty_range: float = 0.15  # +/- 15%
    
    def calculate_totals(self):
        """Calculate total CAPEX, OPEX, and net cost"""
        self.capex_total = (
            self.capex_energy_efficiency +
            self.capex_renewable_energy +
            self.capex_building_retrofit +
            self.capex_technology_upgrade +
            self.capex_other
        )
        
        self.opex_total = (
            self.opex_maintenance +
            self.opex_energy +
            self.opex_monitoring +
            self.opex_staff +
            self.opex_consulting +
            self.opex_other
        )
        
        self.total_savings = (
            self.energy_cost_savings +
            self.operational_savings +
            self.avoided_costs +
            self.incentives_received
        )
        
        self.net_cost = self.capex_total + self.opex_total - self.total_savings
    
    def get_cost_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get detailed cost breakdown"""
        return {
            'capex': {
                'energy_efficiency': self.capex_energy_efficiency,
                'renewable_energy': self.capex_renewable_energy,
                'building_retrofit': self.capex_building_retrofit,
                'technology_upgrade': self.capex_technology_upgrade,
                'other': self.capex_other,
                'total': self.capex_total
            },
            'opex': {
                'maintenance': self.opex_maintenance,
                'energy': self.opex_energy,
                'monitoring': self.opex_monitoring,
                'staff': self.opex_staff,
                'consulting': self.opex_consulting,
                'other': self.opex_other,
                'total': self.opex_total
            },
            'savings': {
                'energy_cost_savings': self.energy_cost_savings,
                'operational_savings': self.operational_savings,
                'avoided_costs': self.avoided_costs,
                'incentives': self.incentives_received,
                'total': self.total_savings
            }
        }

@dataclass
class CapexOpex:
    """
    Simple CAPEX/OPEX data structure
    """
    capex: float = 0.0
    opex: float = 0.0
    currency: Currency = Currency.AUD
    
    def get_total(self) -> float:
        """Get total cost"""
        return self.capex + self.opex

@dataclass
class CostSchedule(BaseModel):
    """
    Multi-year cost schedule
    """
    schedule_id: str = ""
    name: str = ""
    scenario_id: str = ""
    
    # Time period
    start_year: int = 0
    end_year: int = 0
    
    # Yearly projections
    yearly_costs: Dict[int, CostProjection] = field(default_factory=dict)
    
    # Totals
    total_capex: float = 0.0
    total_opex: float = 0.0
    total_savings: float = 0.0
    net_total_cost: float = 0.0
    
    # Currency
    currency: Currency = Currency.AUD
    
    def add_year_projection(self, year: int, projection: CostProjection):
        """Add projection for specific year"""
        self.yearly_costs[year] = projection
    
    def calculate_totals(self):
        """Calculate total costs across all years"""
        self.total_capex = sum(proj.capex_total for proj in self.yearly_costs.values())
        self.total_opex = sum(proj.opex_total for proj in self.yearly_costs.values())
        self.total_savings = sum(proj.total_savings for proj in self.yearly_costs.values())
        self.net_total_cost = self.total_capex + self.total_opex - self.total_savings
    
    def get_peak_cost_year(self) -> Optional[int]:
        """Get year with highest cost"""
        if not self.yearly_costs:
            return None
        return max(self.yearly_costs.keys(), key=lambda y: self.yearly_costs[y].net_cost)

# =============================================================================
# FINANCIAL ANALYSIS MODELS
# =============================================================================

@dataclass
class ROICalculation(BaseModel):
    """
    Return on Investment calculation
    """
    calculation_id: str = ""
    name: str  = ""
    
    # Investment
    initial_investment: float = 0.0
    additional_investments: Dict[int, float] = field(default_factory=dict)
    total_investment: float = 0.0
    
    # Returns
    annual_returns: Dict[int, float] = field(default_factory=dict)  # year: return
    total_returns: float = 0.0
    
    # Analysis period
    analysis_period_years: int = 10
    
    # Results
    roi_percentage: float = 0.0
    simple_payback_years: float = 0.0
    
    def calculate_roi(self):
        """Calculate ROI percentage"""
        self.total_investment = self.initial_investment + sum(self.additional_investments.values())
        self.total_returns = sum(self.annual_returns.values())
        
        if self.total_investment > 0:
            self.roi_percentage = ((self.total_returns - self.total_investment) / 
                                  self.total_investment * 100)
    
    def calculate_simple_payback(self):
        """Calculate simple payback period"""
        if not self.annual_returns:
            self.simple_payback_years = 999.0
            return
        
        cumulative = 0
        sorted_years = sorted(self.annual_returns.keys())
        
        for year in sorted_years:
            cumulative += self.annual_returns[year]
            if cumulative >= self.total_investment:
                self.simple_payback_years = year
                return
        
        self.simple_payback_years = 999.0  # Never pays back

@dataclass
class NPVCalculation(BaseModel):
    """
    Net Present Value calculation
    """
    calculation_id: str = ""
    name: str = ""
    
    # Cash flows
    initial_investment: float = 0.0  # Negative value
    annual_cash_flows: Dict[int, float] = field(default_factory=dict)  # year: cash flow
    
    # Parameters
    discount_rate: float = 0.05  # 5% default
    analysis_period_years: int = 20
    
    # Results
    npv: float = 0.0
    irr: float = 0.0  # Internal Rate of Return
    profitability_index: float = 0.0
    
    def calculate_npv(self):
        """Calculate Net Present Value"""
        # Start with initial investment (negative)
        self.npv = -self.initial_investment
        
        # Add discounted cash flows
        for year, cash_flow in self.annual_cash_flows.items():
            discount_factor = (1 + self.discount_rate) ** year
            self.npv += cash_flow / discount_factor
        
        # Calculate profitability index
        if self.initial_investment > 0:
            pv_future_cash_flows = self.npv + self.initial_investment
            self.profitability_index = pv_future_cash_flows / self.initial_investment
    
    def is_viable(self) -> bool:
        """Check if investment is financially viable"""
        return self.npv > 0 and self.profitability_index > 1.0

@dataclass
class PaybackPeriod(BaseModel):
    """
    Payback period analysis
    """
    calculation_id: str
    
    # Investment
    initial_cost: float = 0.0
    
    # Returns
    annual_savings: float = 0.0
    annual_revenue: float = 0.0
    
    # Results
    simple_payback_years: float = 0.0
    discounted_payback_years: float = 0.0
    
    # Parameters
    discount_rate: float = 0.05
    
    def calculate_simple_payback(self):
        """Calculate simple payback period"""
        total_annual_return = self.annual_savings + self.annual_revenue
        
        if total_annual_return > 0:
            self.simple_payback_years = self.initial_cost / total_annual_return
        else:
            self.simple_payback_years = 999.0
    
    def calculate_discounted_payback(self):
        """Calculate discounted payback period"""
        if self.annual_savings + self.annual_revenue <= 0:
            self.discounted_payback_years = 999.0
            return
        
        cumulative_pv = 0
        year = 0
        annual_return = self.annual_savings + self.annual_revenue
        
        while cumulative_pv < self.initial_cost and year < 100:
            year += 1
            discount_factor = (1 + self.discount_rate) ** year
            pv_return = annual_return / discount_factor
            cumulative_pv += pv_return
        
        self.discounted_payback_years = year if cumulative_pv >= self.initial_cost else 999.0

@dataclass
class FinancialMetrics(BaseModel):
    """
    Comprehensive financial metrics
    """
    metrics_id: str
    scenario_id: str
    calculation_date: datetime = field(default_factory=datetime.now)
    
    # Cost metrics
    total_capex: float = 0.0
    total_opex: float = 0.0
    total_cost: float = 0.0
    
    # Return metrics
    total_savings: float = 0.0
    total_revenue: float = 0.0
    
    # Efficiency metrics
    cost_per_tonne_co2: float = 0.0
    savings_per_tonne_co2: float = 0.0
    
    # Investment metrics
    roi_percentage: float = 0.0
    npv: float = 0.0
    irr: float = 0.0
    payback_years: float = 0.0
    profitability_index: float = 0.0
    
    # Risk metrics
    financial_risk_score: float = 0.0  # 0-100
    cost_uncertainty: float = 0.15  # +/- percentage
    
    # Assumptions
    discount_rate: float = 0.05
    analysis_period: int = 20
    currency: Currency = Currency.AUD
    
    def calculate_cost_per_tonne(self, total_co2_reduction: float):
        """Calculate cost per tonne of CO2 reduced"""
        if total_co2_reduction > 0:
            # Convert to tonnes
            reduction_tonnes = total_co2_reduction / 1000
            self.cost_per_tonne_co2 = self.total_cost / reduction_tonnes

    def is_financially_attractive(self) -> bool:
        """Check if scenario is financially attractive"""
        return (
            self.npv > 0 and
            self.roi_percentage > 0 and
            self.payback_years < 15 and
            self.profitability_index > 1.0
        )

# =============================================================================
# BUDGET AND ALLOCATION MODELS
# =============================================================================

@dataclass
class BudgetAllocation:
    """
    Budget allocation by category or property
    """
    allocation_id: str
    name: str
    total_budget: float
    
    # Allocations
    allocations: Dict[str, float] = field(default_factory=dict)  # category/property: amount
    
    # Status
    allocated_amount: float = 0.0
    remaining_budget: float = 0.0
    utilization_percentage: float = 0.0
    
    def calculate_utilization(self):
        """Calculate budget utilization"""
        self.allocated_amount = sum(self.allocations.values())
        self.remaining_budget = self.total_budget - self.allocated_amount
        
        if self.total_budget > 0:
            self.utilization_percentage = (self.allocated_amount / self.total_budget) * 100
    
    def is_over_budget(self) -> bool:
        """Check if over budget"""
        return self.allocated_amount > self.total_budget

# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'CostProjection',
    'CapexOpex',
    'CostSchedule',
    'ROICalculation',
    'NPVCalculation',
    'PaybackPeriod',
    'FinancialMetrics',
    'BudgetAllocation'
]
