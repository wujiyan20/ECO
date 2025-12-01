# models/property.py - Property Data Models
"""
Property data models for the EcoAssist system.
Handles property information, building characteristics, and portfolio management.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base import (
    BaseModel,
    AuditableModel,
    validate_positive_number,
    validate_year,
    calculate_carbon_intensity,
    ValidationError
)
from .enums import (
    BuildingType,
    RetrofitPotential,
    PropertyStatus,
    EmissionScope,
    AreaUnit
)

# =============================================================================
# PROPERTY MODELS
# =============================================================================

@dataclass
class Property(AuditableModel):
    """
    Comprehensive property data model
    Represents a building or facility in the portfolio
    """
    # Basic Information
    property_id: str = ""  # External property identifier
    name: str = ""
    description: Optional[str] = None
    
    # Location
    address: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country: str = "Australia"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    # Physical Characteristics
    area_sqm: float = 0.0
    gross_floor_area: float = 0.0
    net_lettable_area: float = 0.0
    building_type: BuildingType = BuildingType.OFFICE
    year_built: Optional[int] = None
    year_renovated: Optional[int] = None
    number_of_floors: Optional[int] = None
    
    # Operational Status
    status: PropertyStatus = PropertyStatus.ACTIVE
    occupancy_rate: float = 0.0  # Percentage 0-100
    occupant_count: Optional[int] = None
    operating_hours_per_week: Optional[float] = None
    
    # Emission Data
    baseline_emission: float = 0.0  # kg-CO2e
    scope1_emission: float = 0.0
    scope2_emission: float = 0.0
    scope3_emission: float = 0.0
    emission_intensity: float = 0.0  # kg-CO2e per sqm
    
    # Energy Data
    annual_energy_consumption: float = 0.0  # kWh
    annual_energy_cost: float = 0.0  # AUD
    energy_intensity: float = 0.0  # kWh per sqm
    
    # Retrofit Information
    retrofit_potential: RetrofitPotential = RetrofitPotential.MEDIUM
    retrofit_priority_score: float = 0.0  # 0-100
    last_energy_audit_date: Optional[datetime] = None
    
    # Target Allocation (calculated fields)
    allocated_2030_target: float = 0.0  # kg-CO2e
    allocated_2050_target: float = 0.0  # kg-CO2e
    reduction_potential_percentage: float = 0.0
    
    # Financial
    property_value: Optional[float] = None
    annual_operating_cost: Optional[float] = None
    
    # Metadata
    portfolio_id: Optional[str] = None
    business_unit: Optional[str] = None
    manager_name: Optional[str] = None
    manager_email: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_total_emission(self) -> float:
        """Calculate total emission across all scopes"""
        return self.scope1_emission + self.scope2_emission + self.scope3_emission
    
    def calculate_emission_intensity(self) -> float:
        """Calculate emission intensity (kg-CO2e per sqm)"""
        if self.area_sqm <= 0:
            return 0.0
        return calculate_carbon_intensity(self.calculate_total_emission(), self.area_sqm)
    
    def calculate_energy_intensity(self) -> float:
        """Calculate energy intensity (kWh per sqm)"""
        if self.area_sqm <= 0:
            return 0.0
        return self.annual_energy_consumption / self.area_sqm
    
    def update_intensities(self):
        """Update calculated intensity fields"""
        self.emission_intensity = self.calculate_emission_intensity()
        self.energy_intensity = self.calculate_energy_intensity()
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate property data"""
        is_valid, errors = super().validate()
        
        # Required fields
        if not self.property_id:
            errors.append("Property ID is required")
        
        if not self.name:
            errors.append("Property name is required")
        
        # Physical attributes
        if self.area_sqm <= 0:
            errors.append("Area must be greater than zero")
        
        if self.gross_floor_area > 0 and self.gross_floor_area < self.area_sqm:
            errors.append("Gross floor area cannot be less than area")
        
        # Year validation
        if self.year_built:
            try:
                validate_year(self.year_built, min_year=1800, max_year=datetime.now().year)
            except ValidationError as e:
                errors.append(str(e))
        
        if self.year_renovated and self.year_built:
            if self.year_renovated < self.year_built:
                errors.append("Renovation year cannot be before build year")
        
        # Occupancy
        if not (0 <= self.occupancy_rate <= 100):
            errors.append("Occupancy rate must be between 0 and 100")
        
        # Emissions
        if self.scope1_emission < 0 or self.scope2_emission < 0 or self.scope3_emission < 0:
            errors.append("Emission values cannot be negative")
        
        return len(errors) == 0, errors
    
    def get_age(self) -> Optional[int]:
        """Get building age in years"""
        if self.year_built:
            return datetime.now().year - self.year_built
        return None
    
    def is_high_priority(self) -> bool:
        """Check if property is high priority for retrofits"""
        return self.retrofit_potential in [RetrofitPotential.CRITICAL, RetrofitPotential.HIGH]
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Get summary information for property"""
        return {
            'property_id': self.property_id,
            'name': self.name,
            'building_type': self.building_type.value,
            'area_sqm': self.area_sqm,
            'total_emission': self.calculate_total_emission(),
            'emission_intensity': self.emission_intensity,
            'retrofit_potential': self.retrofit_potential.value,
            'status': self.status.value
        }

@dataclass
class PropertyEmissionBreakdown:
    """Detailed emission breakdown for a property"""
    property_id: str
    year: int
    
    # Scope 1 breakdown
    scope1_total: float = 0.0
    scope1_natural_gas: float = 0.0
    scope1_diesel: float = 0.0
    scope1_other_fuels: float = 0.0
    scope1_fugitive: float = 0.0
    
    # Scope 2 breakdown
    scope2_total: float = 0.0
    scope2_electricity: float = 0.0
    scope2_heating: float = 0.0
    scope2_cooling: float = 0.0
    
    # Scope 3 breakdown (optional)
    scope3_total: float = 0.0
    scope3_business_travel: float = 0.0
    scope3_waste: float = 0.0
    scope3_water: float = 0.0
    scope3_other: float = 0.0
    
    total_emissions: float = 0.0
    unit: str = "kg-CO2e"
    
    def calculate_totals(self):
        """Calculate total emissions"""
        self.scope1_total = (
            self.scope1_natural_gas +
            self.scope1_diesel +
            self.scope1_other_fuels +
            self.scope1_fugitive
        )
        
        self.scope2_total = (
            self.scope2_electricity +
            self.scope2_heating +
            self.scope2_cooling
        )
        
        self.scope3_total = (
            self.scope3_business_travel +
            self.scope3_waste +
            self.scope3_water +
            self.scope3_other
        )
        
        self.total_emissions = self.scope1_total + self.scope2_total + self.scope3_total
    
    def get_scope_percentages(self) -> Dict[str, float]:
        """Get percentage breakdown by scope"""
        if self.total_emissions == 0:
            return {'scope1': 0, 'scope2': 0, 'scope3': 0}
        
        return {
            'scope1': (self.scope1_total / self.total_emissions) * 100,
            'scope2': (self.scope2_total / self.total_emissions) * 100,
            'scope3': (self.scope3_total / self.total_emissions) * 100
        }

@dataclass
class PropertyMetrics:
    """Performance metrics for a property"""
    property_id: str
    calculation_date: datetime = field(default_factory=datetime.now)
    
    # Emission metrics
    emission_intensity: float = 0.0  # kg-CO2e per sqm
    emission_per_occupant: float = 0.0  # kg-CO2e per person
    year_over_year_change: float = 0.0  # Percentage
    
    # Energy metrics
    energy_intensity: float = 0.0  # kWh per sqm
    energy_cost_intensity: float = 0.0  # AUD per sqm
    
    # Performance scores
    retrofit_priority_score: float = 0.0  # 0-100
    carbon_reduction_potential: float = 0.0  # Percentage
    cost_efficiency_score: float = 0.0  # 0-100
    
    # Comparison to benchmarks
    vs_portfolio_average: float = 0.0  # Percentage difference
    vs_industry_benchmark: float = 0.0  # Percentage difference
    vs_best_in_class: float = 0.0  # Percentage difference
    
    # Risk assessment
    climate_risk_score: float = 0.0  # 0-100
    regulatory_risk_score: float = 0.0  # 0-100
    financial_risk_score: float = 0.0  # 0-100

@dataclass
class Portfolio:
    """Portfolio of properties"""
    portfolio_id: str
    name: str
    description: Optional[str] = None
    properties: List[Property] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def get_total_area(self) -> float:
        """Get total area of all properties"""
        return sum(p.area_sqm for p in self.properties)
    
    def get_total_emissions(self) -> float:
        """Get total emissions across portfolio"""
        return sum(p.calculate_total_emission() for p in self.properties)
    
    def get_average_emission_intensity(self) -> float:
        """Get portfolio-wide emission intensity"""
        total_area = self.get_total_area()
        if total_area == 0:
            return 0.0
        return self.get_total_emissions() / total_area
    
    def get_properties_by_type(self, building_type: BuildingType) -> List[Property]:
        """Get properties of specific building type"""
        return [p for p in self.properties if p.building_type == building_type]
    
    def get_high_priority_properties(self) -> List[Property]:
        """Get properties with high retrofit potential"""
        return [p for p in self.properties if p.is_high_priority()]
    
    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics"""
        if not self.properties:
            return {}
        
        return {
            'property_count': len(self.properties),
            'total_area_sqm': self.get_total_area(),
            'total_emissions': self.get_total_emissions(),
            'average_emission_intensity': self.get_average_emission_intensity(),
            'building_type_distribution': self._get_building_type_distribution(),
            'retrofit_potential_summary': self._get_retrofit_potential_summary(),
            'total_allocated_2030_target': sum(p.allocated_2030_target for p in self.properties),
            'total_allocated_2050_target': sum(p.allocated_2050_target for p in self.properties)
        }
    
    def _get_building_type_distribution(self) -> Dict[str, int]:
        """Get count of properties by building type"""
        distribution = {}
        for prop in self.properties:
            type_name = prop.building_type.value
            distribution[type_name] = distribution.get(type_name, 0) + 1
        return distribution
    
    def _get_retrofit_potential_summary(self) -> Dict[str, int]:
        """Get count of properties by retrofit potential"""
        summary = {}
        for prop in self.properties:
            potential = prop.retrofit_potential.value
            summary[potential] = summary.get(potential, 0) + 1
        return summary

# =============================================================================
# PROPERTY FILTER AND QUERY MODELS
# =============================================================================

@dataclass
class PropertyFilter:
    """Filter criteria for querying properties"""
    property_ids: Optional[List[str]] = None
    building_types: Optional[List[BuildingType]] = None
    retrofit_potentials: Optional[List[RetrofitPotential]] = None
    min_area: Optional[float] = None
    max_area: Optional[float] = None
    min_emission_intensity: Optional[float] = None
    max_emission_intensity: Optional[float] = None
    cities: Optional[List[str]] = None
    states: Optional[List[str]] = None
    portfolio_id: Optional[str] = None
    status: Optional[PropertyStatus] = None
    tags: Optional[List[str]] = None

# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'Property',
    'PropertyEmissionBreakdown',
    'PropertyMetrics',
    'Portfolio',
    'PropertyFilter'
]
