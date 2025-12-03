# database/models.py - Data Models
"""
Data models matching EcoAssist database schema
All models use dataclasses for clean structure and easy serialization
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import json


# ================================================================================
# ENUMS
# ================================================================================

class BuildingType(str, Enum):
    """Building types"""
    OFFICE = "Office"
    RETAIL = "Retail"
    INDUSTRIAL = "Industrial"
    RESIDENTIAL = "Residential"
    MIXED_USE = "Mixed Use"
    WAREHOUSE = "Warehouse"
    HOTEL = "Hotel"
    HEALTHCARE = "Healthcare"
    EDUCATION = "Education"


class RetrofitPotential(str, Enum):
    """Retrofit potential levels"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class RiskLevel(str, Enum):
    """Risk level categories"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class OptionCategory(str, Enum):
    """Reduction option categories"""
    ENERGY_EFFICIENCY = "Energy Efficiency"
    RENEWABLE_ENERGY = "Renewable Energy"
    FUEL_SWITCHING = "Fuel Switching"
    PROCESS_IMPROVEMENT = "Process Improvement"
    CARBON_OFFSET = "Carbon Offset"
    BEHAVIOR_CHANGE = "Behavior Change"


# ================================================================================
# BASE MODEL
# ================================================================================

@dataclass
class BaseModel:
    """Base model with common functionality"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        # Convert datetime objects to ISO format strings
        for key, value in result.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create instance from dictionary"""
        return cls(**data)


# ================================================================================
# PROPERTY MODELS
# ================================================================================

@dataclass
class Property(BaseModel):
    """Property data model matching database schema"""
    property_id: str
    name: str
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    area_sqm: float = 0.0
    building_type: str = "Office"
    retrofit_potential: str = "Medium"
    baseline_emission: float = 0.0
    scope1_emission: float = 0.0
    scope2_emission: float = 0.0
    scope3_emission: float = 0.0
    carbon_intensity: Optional[float] = None
    annual_energy_cost: float = 0.0
    portfolio_id: str = "DEFAULT"
    region: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def calculate_total_emission(self) -> float:
        """Calculate total emissions across all scopes"""
        return self.scope1_emission + self.scope2_emission + self.scope3_emission
    
    def calculate_carbon_intensity(self) -> float:
        """Calculate carbon intensity (kg CO2e / m²)"""
        if self.area_sqm > 0:
            return self.calculate_total_emission() / self.area_sqm
        return 0.0
    
    def update_carbon_intensity(self):
        """Update the carbon_intensity field"""
        self.carbon_intensity = self.calculate_carbon_intensity()


@dataclass
class PropertyTarget:
    """Property-specific emission reduction targets"""
    target_id: str
    property_id: str
    scenario_id: str
    target_year: int
    baseline_emission: float
    target_emission: float
    reduction_percentage: float
    allocated_budget: float = 0.0
    implementation_priority: int = 3
    notes: Optional[str] = None
    created_at: Optional[datetime] = None


# ================================================================================
# REDUCTION OPTION MODELS
# ================================================================================

@dataclass
class ReductionOption(BaseModel):
    """Reduction option data model matching database schema"""
    option_id: str
    name: str
    description: Optional[str] = None
    category: str = "Energy Efficiency"
    co2_reduction_potential: float = 0.0
    capex: float = 0.0
    opex: float = 0.0
    annual_savings: float = 0.0
    payback_period_years: Optional[float] = None
    roi_percentage: Optional[float] = None
    priority: int = 3
    implementation_time_months: int = 6
    risk_level: str = "Medium"
    technology_type: Optional[str] = None
    applicable_building_types: Optional[List[str]] = None
    min_building_size_sqm: Optional[float] = None
    max_building_size_sqm: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.applicable_building_types is None:
            self.applicable_building_types = []
        
        # Calculate financial metrics if not set
        if self.payback_period_years is None and self.annual_savings > 0:
            self.payback_period_years = self.capex / self.annual_savings if self.capex > 0 else 0
        
        if self.roi_percentage is None and self.capex > 0:
            lifetime_savings = self.annual_savings * 10  # Assume 10-year lifetime
            self.roi_percentage = ((lifetime_savings - self.capex) / self.capex) * 100
    
    def calculate_cost_effectiveness(self) -> float:
        """Calculate cost per tonne CO2 reduced"""
        if self.co2_reduction_potential > 0:
            total_cost = self.capex + (self.opex * 10)  # 10-year lifecycle
            return total_cost / self.co2_reduction_potential
        return 0.0
    
    def is_suitable_for_property(self, property: Property) -> bool:
        """Check if option is suitable for a property"""
        # Check building type
        if self.applicable_building_types and property.building_type not in self.applicable_building_types:
            return False
        
        # Check building size
        if self.min_building_size_sqm and property.area_sqm < self.min_building_size_sqm:
            return False
        
        if self.max_building_size_sqm and property.area_sqm > self.max_building_size_sqm:
            return False
        
        return True


# ================================================================================
# MILESTONE & SCENARIO MODELS
# ================================================================================

@dataclass
class MilestoneScenario(BaseModel):
    """Milestone scenario data model matching database schema"""
    scenario_id: str
    name: str
    description: Optional[str] = None
    target_year: int = 2050
    baseline_year: int = 2024
    baseline_emission: float = 0.0
    target_emission: float = 0.0
    reduction_percentage: float = 0.0
    yearly_targets: Optional[Dict[str, float]] = None
    scope1_target: float = 0.0
    scope2_target: float = 0.0
    scope3_target: float = 0.0
    total_capex: float = 0.0
    total_opex: float = 0.0
    reduction_rate_2030: float = 0.0
    reduction_rate_2040: float = 0.0
    reduction_rate_2050: float = 0.0
    strategy_type: str = "balanced"
    sbt_aligned: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.yearly_targets is None:
            self.yearly_targets = {}
    
    def get_target_for_year(self, year: int) -> Optional[float]:
        """Get emission target for specific year"""
        return self.yearly_targets.get(str(year))
    
    def calculate_trajectory(self) -> Dict[int, float]:
        """Calculate emission reduction trajectory"""
        trajectory = {}
        years = range(self.baseline_year, self.target_year + 1)
        
        for year in years:
            # Linear interpolation between milestones
            progress = (year - self.baseline_year) / (self.target_year - self.baseline_year)
            emission = self.baseline_emission - (self.baseline_emission - self.target_emission) * progress
            trajectory[year] = round(emission, 2)
        
        return trajectory


# ================================================================================
# STRATEGIC PATTERN MODELS
# ================================================================================

@dataclass
class StrategicPattern(BaseModel):
    """Strategic pattern data model matching database schema"""
    pattern_id: str
    name: str
    description: Optional[str] = None
    reduction_options: Optional[List[str]] = None
    option_priorities: Optional[Dict[str, int]] = None
    estimated_cost: float = 0.0
    estimated_reduction: float = 0.0
    estimated_savings: float = 0.0
    risk_level: str = "Medium"
    implementation_sequence: Optional[List[str]] = None
    total_implementation_time_months: int = 12
    success_rate: float = 0.85
    suitable_building_types: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.reduction_options is None:
            self.reduction_options = []
        if self.option_priorities is None:
            self.option_priorities = {}
        if self.implementation_sequence is None:
            self.implementation_sequence = []
        if self.suitable_building_types is None:
            self.suitable_building_types = []
    
    def calculate_cost_effectiveness(self) -> float:
        """Calculate cost per tonne CO2 reduced"""
        if self.estimated_reduction > 0:
            return self.estimated_cost / self.estimated_reduction
        return 0.0


# ================================================================================
# HISTORICAL DATA MODELS
# ================================================================================

@dataclass
class HistoricalConsumption:
    """Historical energy consumption data"""
    consumption_id: str
    property_id: str
    year: int
    month: int
    fuel_type: str
    consumption_value: float
    unit: str = "kWh"
    cost: float = 0.0
    data_source: Optional[str] = None
    data_quality: str = "Measured"
    created_at: Optional[datetime] = None


@dataclass
class HistoricalEmission:
    """Historical emission data"""
    emission_id: str
    property_id: str
    year: int
    month: int
    scope1_emission: float = 0.0
    scope2_emission: float = 0.0
    scope3_emission: float = 0.0
    total_emission: float = 0.0
    carbon_intensity: Optional[float] = None
    data_source: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def calculate_total(self):
        """Calculate total emissions"""
        self.total_emission = self.scope1_emission + self.scope2_emission + self.scope3_emission


@dataclass
class HistoricalPerformance:
    """Historical performance metrics"""
    performance_id: str
    property_id: str
    year: int
    quarter: int
    energy_consumption: float = 0.0
    total_emission: float = 0.0
    carbon_intensity: float = 0.0
    energy_cost: float = 0.0
    reduction_achieved: float = 0.0
    target_compliance: float = 0.0
    created_at: Optional[datetime] = None


# ================================================================================
# PRICING MODELS
# ================================================================================

@dataclass
class CarbonCreditPrice:
    """Carbon credit pricing data model"""
    credit_type: str
    year: int
    price_per_tonne: float
    currency: str = "AUD"
    market_region: str = "Australia"
    vintage_year: Optional[int] = None
    certification_standard: Optional[str] = None
    project_type: Optional[str] = None
    trading_volume: Optional[float] = None
    price_volatility: Optional[float] = None
    co_benefits: Optional[List[str]] = None
    data_source: str = ""
    data_quality_score: float = 100.0
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.co_benefits is None:
            self.co_benefits = []


@dataclass
class RenewableEnergyPrice:
    """Renewable energy pricing data model"""
    energy_type: str
    year: int
    currency: str = "AUD"
    region: str = "Australia"
    price_per_mwh: Optional[float] = None
    price_per_kw_installed: Optional[float] = None
    technology_specification: Optional[str] = None
    capacity_range: Optional[str] = None
    capex: Optional[float] = None
    opex_annual: Optional[float] = None
    capacity_factor: Optional[float] = None
    expected_lifetime_years: int = 25
    degradation_rate: Optional[float] = None
    incentive_stc_value: Optional[float] = None
    incentive_lgc_value: Optional[float] = None
    feed_in_tariff: Optional[float] = None
    installation_cost: Optional[float] = None
    grid_connection_cost: Optional[float] = None
    data_source: str = ""
    confidence_level: float = 95.0
    notes: Optional[str] = None


@dataclass
class RenewableFuelPrice:
    """Renewable fuel pricing data model"""
    fuel_type: str
    year: int
    price_per_unit: float
    unit_type: str
    currency: str = "AUD"
    region: str = "Australia"
    feedstock_type: Optional[str] = None
    production_pathway: Optional[str] = None
    energy_content_mj: Optional[float] = None
    emission_factor: Optional[float] = None
    carbon_intensity_score: Optional[float] = None
    blend_ratio: Optional[str] = None
    purity_grade: Optional[str] = None
    fossil_fuel_price_equivalent: Optional[float] = None
    price_premium_percentage: Optional[float] = None
    sustainability_certification: Optional[str] = None
    lcfs_credit_value: Optional[float] = None
    feedstock_availability: Optional[str] = None
    storage_requirements: Optional[str] = None
    equipment_compatibility: Optional[str] = None
    data_source: str = ""
    data_quality_score: float = 100.0
    notes: Optional[str] = None


# ================================================================================
# DASHBOARD MODELS
# ================================================================================

@dataclass
class DashboardMetrics:
    """Dashboard metrics summary"""
    total_properties: int = 0
    total_emission: float = 0.0
    total_scope1: float = 0.0
    total_scope2: float = 0.0
    total_scope3: float = 0.0
    average_carbon_intensity: float = 0.0
    total_area_sqm: float = 0.0
    total_energy_cost: float = 0.0
    reduction_target_2030: float = 0.0
    reduction_target_2050: float = 0.0
    current_reduction_rate: float = 0.0
    on_track_properties: int = 0
    at_risk_properties: int = 0


@dataclass
class PortfolioSummary:
    """Portfolio summary"""
    portfolio_id: str
    total_properties: int
    total_emission: float
    total_area: float
    avg_carbon_intensity: float
    building_type_distribution: Dict[str, int]
    retrofit_potential_distribution: Dict[str, int]
    
    def __post_init__(self):
        if not hasattr(self, 'building_type_distribution'):
            self.building_type_distribution = {}
        if not hasattr(self, 'retrofit_potential_distribution'):
            self.retrofit_potential_distribution = {}


# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def serialize_model(model: BaseModel) -> Dict[str, Any]:
    """Serialize model to dictionary"""
    return model.to_dict()


def deserialize_model(model_class, data: Dict[str, Any]):
    """Deserialize dictionary to model"""
    return model_class.from_dict(data)
