# models/emission.py - Emission Data Models
"""
Emission and baseline data models for the EcoAssist system.
Handles emission calculations, baselines, and historical data.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base import (
    BaseModel,
    validate_year,
    validate_positive_number,
    calculate_percentage_change,
    ValidationError
)
from .enums import (
    EmissionScope,
    EmissionCategory,
    FuelType,
    EmissionUnit,
    DataQuality,
    MeasurementMethod
)

# =============================================================================
# BASELINE EMISSION MODELS
# =============================================================================

@dataclass
class BaselineDataRecord(BaseModel):
    """
    Single year of baseline emission data
    Used for establishing historical emission patterns
    """
    year: int = 0
    property_id: Optional[str] = None
    
    # Emission data
    scope1_emissions: float = 0.0  # kg-CO2e
    scope2_emissions: float = 0.0  # kg-CO2e
    scope3_emissions: Optional[float] = 0.0  # kg-CO2e (optional)
    total_emissions: float = 0.0  # kg-CO2e
    
    # Energy consumption
    total_consumption: float = 0.0  # kWh
    electricity_consumption: float = 0.0  # kWh
    gas_consumption: float = 0.0  # kWh or cubic meters
    other_fuel_consumption: float = 0.0  # kWh
    
    # Cost data
    total_cost: float = 0.0  # AUD
    electricity_cost: float = 0.0  # AUD
    gas_cost: float = 0.0  # AUD
    other_fuel_cost: float = 0.0  # AUD
    
    # Metadata
    unit: EmissionUnit = EmissionUnit.KG_CO2E
    data_quality: DataQuality = DataQuality.MEDIUM
    measurement_method: MeasurementMethod = MeasurementMethod.CALCULATED
    data_source: Optional[str] = None
    notes: Optional[str] = None
    verified: bool = False
    verified_by: Optional[str] = None
    verified_date: Optional[datetime] = None
    
    def calculate_total_emissions(self) -> float:
        """Calculate total emissions across all scopes"""
        scope3 = self.scope3_emissions or 0.0
        total = self.scope1_emissions + self.scope2_emissions + scope3
        self.total_emissions = total
        return total
    
    def calculate_emission_intensity(self, area_sqm: float) -> float:
        """Calculate emission intensity per area"""
        if area_sqm <= 0:
            return 0.0
        return self.calculate_total_emissions() / area_sqm
    
    def calculate_energy_intensity(self, area_sqm: float) -> float:
        """Calculate energy intensity per area"""
        if area_sqm <= 0:
            return 0.0
        return self.total_consumption / area_sqm
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate baseline data"""
        is_valid, errors = super().validate()
        
        # Validate year
        try:
            validate_year(self.year, min_year=2000, max_year=datetime.now().year + 1)
        except ValidationError as e:
            errors.append(str(e))
        
        # Validate emissions
        if self.scope1_emissions < 0:
            errors.append("Scope 1 emissions cannot be negative")
        if self.scope2_emissions < 0:
            errors.append("Scope 2 emissions cannot be negative")
        if self.scope3_emissions and self.scope3_emissions < 0:
            errors.append("Scope 3 emissions cannot be negative")
        
        # Validate consumption
        if self.total_consumption < 0:
            errors.append("Total consumption cannot be negative")
        
        # Validate costs
        if self.total_cost < 0:
            errors.append("Total cost cannot be negative")
        
        # Check consistency
        calculated_total = self.scope1_emissions + self.scope2_emissions + (self.scope3_emissions or 0)
        if self.total_emissions > 0 and abs(calculated_total - self.total_emissions) > 0.01:
            errors.append("Total emissions doesn't match sum of scopes")
        
        return len(errors) == 0, errors

@dataclass
class EmissionFactor:
    """
    Emission factor for converting activity data to emissions
    Based on GHG Protocol and regional factors
    """
    factor_id: str
    name: str
    fuel_type: FuelType
    emission_scope: EmissionScope
    
    # Factor value
    factor_value: float  # kg-CO2e per unit
    unit: str  # e.g., "kWh", "L", "kg", "m3"
    
    # Geographic and temporal scope
    country: str = "Australia"
    region: Optional[str] = None
    valid_from_year: int = 2020
    valid_to_year: Optional[int] = None
    
    # Source information
    data_source: str = ""
    source_url: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Additional factors (optional)
    ch4_factor: Optional[float] = None  # Methane
    n2o_factor: Optional[float] = None  # Nitrous oxide
    
    def is_valid_for_year(self, year: int) -> bool:
        """Check if factor is valid for given year"""
        if year < self.valid_from_year:
            return False
        if self.valid_to_year and year > self.valid_to_year:
            return False
        return True
    
    def calculate_emissions(self, activity_amount: float) -> float:
        """Calculate emissions from activity amount"""
        return activity_amount * self.factor_value

@dataclass
class EmissionCalculation:
    """
    Record of emission calculation with detailed breakdown
    """
    calculation_id: str
    property_id: str
    calculation_date: datetime = field(default_factory=datetime.now)
    year: int = 0
    
    # Scope 1 calculations
    scope1_natural_gas: float = 0.0
    scope1_natural_gas_activity: float = 0.0  # m3 or kWh
    scope1_diesel: float = 0.0
    scope1_diesel_activity: float = 0.0  # L
    scope1_gasoline: float = 0.0
    scope1_gasoline_activity: float = 0.0  # L
    scope1_other: float = 0.0
    scope1_total: float = 0.0
    
    # Scope 2 calculations
    scope2_electricity: float = 0.0
    scope2_electricity_activity: float = 0.0  # kWh
    scope2_heating: float = 0.0
    scope2_heating_activity: float = 0.0  # kWh
    scope2_cooling: float = 0.0
    scope2_cooling_activity: float = 0.0  # kWh
    scope2_total: float = 0.0
    
    # Scope 3 calculations (optional)
    scope3_travel: float = 0.0
    scope3_waste: float = 0.0
    scope3_water: float = 0.0
    scope3_other: float = 0.0
    scope3_total: float = 0.0
    
    # Totals
    total_emissions: float = 0.0
    
    # Factors used
    emission_factors_used: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    calculation_method: str = "GHG Protocol"
    data_quality: DataQuality = DataQuality.MEDIUM
    uncertainty_percentage: float = 10.0
    notes: Optional[str] = None
    
    def calculate_totals(self):
        """Calculate total emissions for each scope"""
        self.scope1_total = (
            self.scope1_natural_gas +
            self.scope1_diesel +
            self.scope1_gasoline +
            self.scope1_other
        )
        
        self.scope2_total = (
            self.scope2_electricity +
            self.scope2_heating +
            self.scope2_cooling
        )
        
        self.scope3_total = (
            self.scope3_travel +
            self.scope3_waste +
            self.scope3_water +
            self.scope3_other
        )
        
        self.total_emissions = self.scope1_total + self.scope2_total + self.scope3_total
    
    def get_scope_breakdown(self) -> Dict[str, float]:
        """Get percentage breakdown by scope"""
        if self.total_emissions == 0:
            return {'scope1': 0, 'scope2': 0, 'scope3': 0}
        
        return {
            'scope1': (self.scope1_total / self.total_emissions) * 100,
            'scope2': (self.scope2_total / self.total_emissions) * 100,
            'scope3': (self.scope3_total / self.total_emissions) * 100
        }

# =============================================================================
# EMISSION TREND AND ANALYSIS MODELS
# =============================================================================

@dataclass
class EmissionTrend:
    """
    Emission trend analysis over multiple years
    """
    property_id: str
    start_year: int
    end_year: int
    baseline_data: List[BaselineDataRecord] = field(default_factory=list)
    
    # Calculated metrics
    average_annual_emissions: float = 0.0
    total_emissions_period: float = 0.0
    trend_direction: str = "stable"  # "increasing", "decreasing", "stable"
    average_annual_change_percentage: float = 0.0
    
    # Volatility
    coefficient_of_variation: float = 0.0
    
    def calculate_metrics(self):
        """Calculate trend metrics from baseline data"""
        if not self.baseline_data:
            return
        
        # Sort by year
        sorted_data = sorted(self.baseline_data, key=lambda x: x.year)
        
        # Calculate totals
        emissions = [d.total_emissions for d in sorted_data]
        self.total_emissions_period = sum(emissions)
        self.average_annual_emissions = self.total_emissions_period / len(emissions)
        
        # Calculate trend
        if len(emissions) > 1:
            changes = []
            for i in range(1, len(emissions)):
                if emissions[i-1] > 0:
                    change = calculate_percentage_change(emissions[i-1], emissions[i])
                    changes.append(change)
            
            if changes:
                self.average_annual_change_percentage = sum(changes) / len(changes)
                
                # Determine trend direction
                if self.average_annual_change_percentage > 2:
                    self.trend_direction = "increasing"
                elif self.average_annual_change_percentage < -2:
                    self.trend_direction = "decreasing"
                else:
                    self.trend_direction = "stable"
        
        # Calculate volatility
        if len(emissions) > 1 and self.average_annual_emissions > 0:
            variance = sum((x - self.average_annual_emissions) ** 2 for x in emissions) / len(emissions)
            std_dev = variance ** 0.5
            self.coefficient_of_variation = (std_dev / self.average_annual_emissions) * 100

@dataclass
class EmissionProjection:
    """
    Projected future emissions under different scenarios
    """
    property_id: str
    projection_year: int
    base_year: int
    baseline_emission: float
    
    # Scenario projections
    business_as_usual: float = 0.0  # No reduction measures
    standard_scenario: float = 0.0  # Standard reduction path
    aggressive_scenario: float = 0.0  # Aggressive reduction path
    
    # Reduction from baseline
    bau_reduction_percentage: float = 0.0
    standard_reduction_percentage: float = 0.0
    aggressive_reduction_percentage: float = 0.0
    
    # Confidence intervals
    confidence_level: float = 0.95
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    
    # Assumptions
    assumptions: Dict[str, Any] = field(default_factory=dict)
    methodology: str = "AI-based projection"
    
    def calculate_reduction_percentages(self):
        """Calculate reduction percentages for each scenario"""
        if self.baseline_emission > 0:
            self.bau_reduction_percentage = calculate_percentage_change(
                self.baseline_emission, self.business_as_usual
            )
            self.standard_reduction_percentage = calculate_percentage_change(
                self.baseline_emission, self.standard_scenario
            )
            self.aggressive_reduction_percentage = calculate_percentage_change(
                self.baseline_emission, self.aggressive_scenario
            )

@dataclass
class EmissionBenchmark:
    """
    Benchmark data for comparison
    """
    benchmark_id: str
    building_type: str
    country: str = "Australia"
    region: Optional[str] = None
    year: int = 0
    
    # Benchmark values
    median_emission_intensity: float = 0.0  # kg-CO2e per sqm
    percentile_25: float = 0.0
    percentile_75: float = 0.0
    best_in_class: float = 0.0  # Top 10%
    worst_in_class: float = 0.0  # Bottom 10%
    
    # Sample information
    sample_size: int = 0
    data_source: str = ""
    
    def get_performance_rating(self, emission_intensity: float) -> str:
        """Get performance rating based on emission intensity"""
        if emission_intensity <= self.best_in_class:
            return "Excellent"
        elif emission_intensity <= self.percentile_25:
            return "Good"
        elif emission_intensity <= self.median_emission_intensity:
            return "Average"
        elif emission_intensity <= self.percentile_75:
            return "Below Average"
        else:
            return "Poor"

# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'BaselineDataRecord',
    'EmissionFactor',
    'EmissionCalculation',
    'EmissionTrend',
    'EmissionProjection',
    'EmissionBenchmark'
]
