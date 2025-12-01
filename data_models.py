# data_models.py - Data Models and Database Layer

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Union, Any
from enum import Enum
import uuid
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BuildingType(Enum):
    """Enumeration for building types"""
    OFFICE = "Office"
    RETAIL = "Retail"
    RESIDENTIAL = "Residential"
    INDUSTRIAL = "Industrial"
    MIXED_USE = "Mixed-Use"
    EDUCATIONAL = "Educational"
    WAREHOUSE = "Warehouse"
    HEALTHCARE = "Healthcare"
    HOSPITALITY = "Hospitality"

class RetrofitPotential(Enum):
    """Enumeration for retrofit potential levels"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    CRITICAL = "Critical"

class PriorityLevel(Enum):
    """Enumeration for priority levels"""
    VERY_HIGH = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    VERY_LOW = 1

class RiskLevel(Enum):
    """Enumeration for risk levels"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class EmissionScope(Enum):
    """Enumeration for emission scopes"""
    SCOPE_1 = "Scope 1"
    SCOPE_2 = "Scope 2"
    SCOPE_3 = "Scope 3"

class FuelType(Enum):
    """Enumeration for fuel types"""
    NATURAL_GAS = "Natural Gas"
    ELECTRICITY = "Electricity"
    DIESEL = "Diesel"
    GASOLINE = "Gasoline"
    PROPANE = "Propane"
    HEATING_OIL = "Heating Oil"
    RENEWABLE = "Renewable"
    BIOFUEL = "Biofuel"
    HYDROGEN = "Hydrogen"

@dataclass
class BaseModel:
    """Base model with common functionality"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary"""
        return asdict(self)
    
    def update_timestamp(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.now()

@dataclass
class Property(BaseModel):
    """Enhanced Property data structure with comprehensive attributes"""
    property_id: str = ""
    name: str = ""
    address: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country: str = "Australia"
    
    # Physical characteristics
    area_sqm: float = 0.0
    gross_floor_area: float = 0.0
    net_lettable_area: float = 0.0
    number_of_floors: int = 1
    year_built: int = 2000
    year_renovated: Optional[int] = None
    
    # Operational data
    occupancy_rate: float = 0.0
    operating_hours_per_day: float = 8.0
    operating_days_per_week: int = 5
    peak_occupancy: int = 0
    
    # Building characteristics
    building_type: BuildingType = BuildingType.OFFICE
    retrofit_potential: RetrofitPotential = RetrofitPotential.MEDIUM
    building_certification: Optional[str] = None
    energy_star_rating: Optional[float] = None
    nabers_rating: Optional[float] = None
    
    # Emission data
    baseline_emission: float = 0.0
    scope1_emission: float = 0.0
    scope2_emission: float = 0.0
    scope3_emission: float = 0.0
    carbon_intensity: float = 0.0  # tCO2e per sqm
    
    # Financial data
    annual_energy_cost: float = 0.0
    maintenance_cost: float = 0.0
    insurance_value: float = 0.0
    
    # Ownership and management
    owner: str = ""
    manager: str = ""
    tenant: str = ""
    lease_expiry: Optional[datetime] = None
    
    # Portfolio grouping
    portfolio_id: str = ""
    business_unit: str = ""
    region: str = ""
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    is_active: bool = True
    
    def calculate_carbon_intensity(self) -> float:
        """Calculate carbon intensity per square meter"""
        if self.area_sqm > 0:
            self.carbon_intensity = self.baseline_emission / self.area_sqm
        return self.carbon_intensity
    
    def get_total_emission(self) -> float:
        """Get total emission across all scopes"""
        return self.scope1_emission + self.scope2_emission + self.scope3_emission
    
    def get_building_efficiency_score(self) -> float:
        """Calculate building efficiency score (0-100)"""
        base_score = 50
        
        # Adjust for building age
        age = datetime.now().year - self.year_built
        age_penalty = min(age * 0.5, 20)  # Max 20 point penalty
        
        # Adjust for retrofit potential
        retrofit_bonus = {
            RetrofitPotential.HIGH: 20,
            RetrofitPotential.MEDIUM: 10,
            RetrofitPotential.LOW: 0,
            RetrofitPotential.CRITICAL: -10
        }
        
        # Adjust for certifications
        cert_bonus = 0
        if self.nabers_rating:
            cert_bonus += self.nabers_rating * 2
        if self.energy_star_rating:
            cert_bonus += self.energy_star_rating
            
        efficiency_score = base_score - age_penalty + retrofit_bonus.get(self.retrofit_potential, 0) + cert_bonus
        return max(0, min(100, efficiency_score))

@dataclass
class ReductionOption(BaseModel):
    """Enhanced CO2 reduction option with detailed specifications"""
    option_id: str = ""
    name: str = ""
    description: str = ""
    category: str = ""
    
    # Impact metrics
    co2_reduction_potential: float = 0.0  # tCO2e per year
    co2_reduction_percentage: float = 0.0  # % reduction
    energy_savings_kwh: float = 0.0
    energy_savings_percentage: float = 0.0
    
    # Financial metrics
    capex: float = 0.0  # Capital expenditure
    opex: float = 0.0   # Operating expenditure per year
    maintenance_cost: float = 0.0  # Annual maintenance
    financing_cost: float = 0.0    # Financing/interest
    
    # ROI calculations
    payback_period_years: float = 0.0
    net_present_value: float = 0.0
    internal_rate_of_return: float = 0.0
    cost_per_tonne_co2: float = 0.0
    
    # Implementation details
    priority: PriorityLevel = PriorityLevel.MEDIUM
    implementation_time_months: int = 6
    implementation_complexity: str = "Medium"
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    # Technical specifications
    technology_type: str = ""
    vendor: str = ""
    warranty_years: int = 5
    expected_lifetime_years: int = 15
    maintenance_frequency: str = "Annual"
    
    # Prerequisites and constraints
    minimum_building_size: float = 0.0
    suitable_building_types: List[BuildingType] = field(default_factory=list)
    climate_zones: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    # Performance tracking
    installation_date: Optional[datetime] = None
    commissioning_date: Optional[datetime] = None
    actual_performance: Dict[str, float] = field(default_factory=dict)
    performance_variance: float = 0.0
    
    # Regulatory and compliance
    regulatory_requirements: List[str] = field(default_factory=list)
    compliance_standards: List[str] = field(default_factory=list)
    incentives_available: List[str] = field(default_factory=list)
    
    def calculate_cost_effectiveness(self) -> Dict[str, float]:
        """Calculate comprehensive cost effectiveness metrics"""
        total_cost = self.capex + (self.opex * self.expected_lifetime_years)
        total_co2_savings = self.co2_reduction_potential * self.expected_lifetime_years
        
        metrics = {
            "cost_per_tonne_co2": total_cost / total_co2_savings if total_co2_savings > 0 else float('inf'),
            "annual_cost_savings": self.energy_savings_kwh * 0.25,  # Assume $0.25/kWh
            "payback_period": self.capex / (self.energy_savings_kwh * 0.25) if self.energy_savings_kwh > 0 else float('inf'),
            "lifetime_savings": (self.energy_savings_kwh * 0.25 * self.expected_lifetime_years) - total_cost
        }
        
        self.cost_per_tonne_co2 = metrics["cost_per_tonne_co2"]
        self.payback_period_years = metrics["payback_period"]
        
        return metrics
    
    def is_suitable_for_property(self, property: Property) -> bool:
        """Check if this reduction option is suitable for a given property"""
        if self.minimum_building_size > 0 and property.area_sqm < self.minimum_building_size:
            return False
            
        if self.suitable_building_types and property.building_type not in self.suitable_building_types:
            return False
            
        return True

@dataclass
class MilestoneScenario(BaseModel):
    """Enhanced milestone scenario with detailed tracking"""
    scenario_id: str = ""
    name: str = ""
    description: str = ""
    scenario_type: str = "Standard"
    
    # Target specifications
    target_year: int = 2050
    baseline_year: int = 2025
    
    # Reduction targets by year
    yearly_targets: Dict[int, float] = field(default_factory=dict)
    yearly_percentage_targets: Dict[int, float] = field(default_factory=dict)
    
    # Scope-specific targets
    scope1_targets: Dict[int, float] = field(default_factory=dict)
    scope2_targets: Dict[int, float] = field(default_factory=dict)
    scope3_targets: Dict[int, float] = field(default_factory=dict)
    
    # Financial projections
    total_capex: float = 0.0
    total_opex: float = 0.0
    annual_capex: Dict[int, float] = field(default_factory=dict)
    annual_opex: Dict[int, float] = field(default_factory=dict)
    
    # Performance metrics
    reduction_rate_2030: float = 0.0
    reduction_rate_2050: float = 0.0
    strategy_type: str = "Balanced"
    
    # Implementation details
    key_milestones: List[Dict[str, Any]] = field(default_factory=list)
    critical_path_items: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
    # Monitoring and verification
    monitoring_frequency: str = "Quarterly"
    verification_method: str = "Third-party"
    reporting_schedule: List[str] = field(default_factory=list)
    
    # Stakeholder information
    approval_status: str = "Draft"
    approved_by: str = ""
    approval_date: Optional[datetime] = None
    
    # Additional metadata
    assumptions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def calculate_trajectory(self) -> Dict[int, Dict[str, float]]:
        """Calculate detailed emission trajectory"""
        trajectory = {}
        
        for year in range(self.baseline_year, self.target_year + 1):
            if year in self.yearly_targets:
                trajectory[year] = {
                    "total_emission": self.yearly_targets[year],
                    "scope1": self.scope1_targets.get(year, 0),
                    "scope2": self.scope2_targets.get(year, 0),
                    "scope3": self.scope3_targets.get(year, 0),
                    "reduction_percentage": self.yearly_percentage_targets.get(year, 0),
                    "capex": self.annual_capex.get(year, 0),
                    "opex": self.annual_opex.get(year, 0)
                }
        
        return trajectory

@dataclass
class StrategicPattern(BaseModel):
    """Enhanced strategic pattern for emission reduction"""
    pattern_id: str = ""
    name: str = ""
    description: str = ""
    pattern_type: str = "Standard"
    
    # Strategy configuration
    reduction_options: Dict[str, int] = field(default_factory=dict)  # option_name: priority
    implementation_approach: str = "Sequential"
    implementation_timeline: Dict[str, int] = field(default_factory=dict)  # phase: months
    
    # Performance estimates
    estimated_cost: float = 0.0
    estimated_capex: float = 0.0
    estimated_opex: float = 0.0
    estimated_reduction: float = 0.0
    estimated_energy_savings: float = 0.0
    
    # Risk assessment
    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    
    # Applicability
    suitable_building_types: List[BuildingType] = field(default_factory=list)
    minimum_portfolio_size: int = 1
    geographic_applicability: List[str] = field(default_factory=list)
    
    # Success metrics
    success_criteria: List[str] = field(default_factory=list)
    kpis: Dict[str, float] = field(default_factory=dict)
    benchmarks: Dict[str, float] = field(default_factory=dict)
    
    # Implementation support
    required_expertise: List[str] = field(default_factory=list)
    recommended_vendors: List[str] = field(default_factory=list)
    training_requirements: List[str] = field(default_factory=list)

@dataclass
class ConsumptionData(BaseModel):
    """Energy and fuel consumption data"""
    property_id: str = ""
    fuel_type: FuelType = FuelType.ELECTRICITY
    consumption_period: str = ""  # YYYY-MM format
    
    # Consumption metrics
    consumption_amount: float = 0.0
    consumption_unit: str = "kWh"
    normalized_consumption: float = 0.0  # per sqm
    
    # Cost data
    unit_cost: float = 0.0
    total_cost: float = 0.0
    demand_charges: float = 0.0
    
    # Emission factors
    emission_factor: float = 0.0  # tCO2e per unit
    total_emissions: float = 0.0
    emission_scope: EmissionScope = EmissionScope.SCOPE_2
    
    # Data quality
    data_source: str = ""
    data_quality_score: float = 100.0
    estimated_data: bool = False
    
    # Weather normalization
    heating_degree_days: Optional[float] = None
    cooling_degree_days: Optional[float] = None
    weather_normalized: bool = False

@dataclass
class EmissionFactor(BaseModel):
    """Emission factors for different fuel types and regions"""
    factor_id: str = ""
    fuel_type: FuelType = FuelType.ELECTRICITY
    region: str = ""
    year: int = datetime.now().year
    
    # Factor values
    emission_factor: float = 0.0  # tCO2e per unit
    unit: str = "tCO2e/MWh"
    source: str = ""
    
    # Validity
    effective_date: datetime = field(default_factory=datetime.now)
    expiry_date: Optional[datetime] = None
    
    # Additional factors
    methane_factor: float = 0.0
    nitrous_oxide_factor: float = 0.0
    
    # Metadata
    data_source: str = ""
    methodology: str = ""
    uncertainty: float = 0.0

class DatabaseManager:
    """Database manager for EcoAssist data"""
    
    def __init__(self, db_path: str = "ecoassist.db"):
        self.db_path = db_path
        self.connection = None
        self.setup_database()
    
    def setup_database(self):
        """Initialize database with all required tables"""
        self.connection = sqlite3.connect(self.db_path)
        cursor = self.connection.cursor()
        
        # Properties table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS properties (
                id TEXT PRIMARY KEY,
                property_id TEXT UNIQUE NOT NULL,
                name TEXT,
                address TEXT,
                city TEXT,
                state TEXT,
                postal_code TEXT,
                country TEXT,
                area_sqm REAL,
                gross_floor_area REAL,
                net_lettable_area REAL,
                number_of_floors INTEGER,
                year_built INTEGER,
                year_renovated INTEGER,
                occupancy_rate REAL,
                operating_hours_per_day REAL,
                operating_days_per_week INTEGER,
                peak_occupancy INTEGER,
                building_type TEXT,
                retrofit_potential TEXT,
                building_certification TEXT,
                energy_star_rating REAL,
                nabers_rating REAL,
                baseline_emission REAL,
                scope1_emission REAL,
                scope2_emission REAL,
                scope3_emission REAL,
                carbon_intensity REAL,
                annual_energy_cost REAL,
                maintenance_cost REAL,
                insurance_value REAL,
                owner TEXT,
                manager TEXT,
                tenant TEXT,
                lease_expiry TEXT,
                portfolio_id TEXT,
                business_unit TEXT,
                region TEXT,
                tags TEXT,
                notes TEXT,
                is_active BOOLEAN,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Reduction options table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reduction_options (
                id TEXT PRIMARY KEY,
                option_id TEXT UNIQUE NOT NULL,
                name TEXT,
                description TEXT,
                category TEXT,
                co2_reduction_potential REAL,
                co2_reduction_percentage REAL,
                energy_savings_kwh REAL,
                energy_savings_percentage REAL,
                capex REAL,
                opex REAL,
                maintenance_cost REAL,
                financing_cost REAL,
                payback_period_years REAL,
                net_present_value REAL,
                internal_rate_of_return REAL,
                cost_per_tonne_co2 REAL,
                priority INTEGER,
                implementation_time_months INTEGER,
                implementation_complexity TEXT,
                risk_level TEXT,
                technology_type TEXT,
                vendor TEXT,
                warranty_years INTEGER,
                expected_lifetime_years INTEGER,
                maintenance_frequency TEXT,
                minimum_building_size REAL,
                suitable_building_types TEXT,
                climate_zones TEXT,
                prerequisites TEXT,
                constraints TEXT,
                installation_date TEXT,
                commissioning_date TEXT,
                actual_performance TEXT,
                performance_variance REAL,
                regulatory_requirements TEXT,
                compliance_standards TEXT,
                incentives_available TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Milestone scenarios table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS milestone_scenarios (
                id TEXT PRIMARY KEY,
                scenario_id TEXT UNIQUE NOT NULL,
                name TEXT,
                description TEXT,
                scenario_type TEXT,
                target_year INTEGER,
                baseline_year INTEGER,
                yearly_targets TEXT,
                yearly_percentage_targets TEXT,
                scope1_targets TEXT,
                scope2_targets TEXT,
                scope3_targets TEXT,
                total_capex REAL,
                total_opex REAL,
                annual_capex TEXT,
                annual_opex TEXT,
                reduction_rate_2030 REAL,
                reduction_rate_2050 REAL,
                strategy_type TEXT,
                key_milestones TEXT,
                critical_path_items TEXT,
                risk_factors TEXT,
                monitoring_frequency TEXT,
                verification_method TEXT,
                reporting_schedule TEXT,
                approval_status TEXT,
                approved_by TEXT,
                approval_date TEXT,
                assumptions TEXT,
                dependencies TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Strategic patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategic_patterns (
                id TEXT PRIMARY KEY,
                pattern_id TEXT UNIQUE NOT NULL,
                name TEXT,
                description TEXT,
                pattern_type TEXT,
                reduction_options TEXT,
                implementation_approach TEXT,
                implementation_timeline TEXT,
                estimated_cost REAL,
                estimated_capex REAL,
                estimated_opex REAL,
                estimated_reduction REAL,
                estimated_energy_savings REAL,
                risk_level TEXT,
                risk_factors TEXT,
                mitigation_strategies TEXT,
                suitable_building_types TEXT,
                minimum_portfolio_size INTEGER,
                geographic_applicability TEXT,
                success_criteria TEXT,
                kpis TEXT,
                benchmarks TEXT,
                required_expertise TEXT,
                recommended_vendors TEXT,
                training_requirements TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Consumption data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consumption_data (
                id TEXT PRIMARY KEY,
                property_id TEXT,
                fuel_type TEXT,
                consumption_period TEXT,
                consumption_amount REAL,
                consumption_unit TEXT,
                normalized_consumption REAL,
                unit_cost REAL,
                total_cost REAL,
                demand_charges REAL,
                emission_factor REAL,
                total_emissions REAL,
                emission_scope TEXT,
                data_source TEXT,
                data_quality_score REAL,
                estimated_data BOOLEAN,
                heating_degree_days REAL,
                cooling_degree_days REAL,
                weather_normalized BOOLEAN,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (property_id) REFERENCES properties (property_id)
            )
        """)
        
        # Emission factors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emission_factors (
                id TEXT PRIMARY KEY,
                factor_id TEXT UNIQUE NOT NULL,
                fuel_type TEXT,
                region TEXT,
                year INTEGER,
                emission_factor REAL,
                unit TEXT,
                source TEXT,
                effective_date TEXT,
                expiry_date TEXT,
                methane_factor REAL,
                nitrous_oxide_factor REAL,
                data_source TEXT,
                methodology TEXT,
                uncertainty REAL,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Historical tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_tracking (
                id TEXT PRIMARY KEY,
                property_id TEXT,
                year INTEGER,
                month INTEGER,
                actual_emissions REAL,
                target_emissions REAL,
                actual_costs REAL,
                target_costs REAL,
                variance_percentage REAL,
                reoptimization_triggered BOOLEAN,
                notes TEXT,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (property_id) REFERENCES properties (property_id)
            )
        """)
        
        # AI calculation results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_calculation_results (
                id TEXT PRIMARY KEY,
                calculation_id TEXT UNIQUE NOT NULL,
                calculation_type TEXT,
                input_parameters TEXT,
                output_results TEXT,
                property_ids TEXT,
                scenario_id TEXT,
                calculation_timestamp TEXT,
                execution_time_seconds REAL,
                algorithm_version TEXT,
                confidence_score REAL,
                validation_status TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Portfolio summary table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_summary (
                id TEXT PRIMARY KEY,
                portfolio_id TEXT,
                reporting_period TEXT,
                total_properties INTEGER,
                total_area_sqm REAL,
                total_baseline_emissions REAL,
                total_current_emissions REAL,
                total_target_emissions REAL,
                reduction_achieved_percentage REAL,
                total_investment REAL,
                cost_per_tonne_reduced REAL,
                on_track_properties INTEGER,
                at_risk_properties INTEGER,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        self.connection.commit()
        logger.info("Database setup completed successfully")
    
    def save_property(self, property: Property) -> bool:
        """Save property to database"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO properties 
                (id, property_id, name, address, city, state, postal_code, country,
                 area_sqm, gross_floor_area, net_lettable_area, number_of_floors,
                 year_built, year_renovated, occupancy_rate, operating_hours_per_day,
                 operating_days_per_week, peak_occupancy, building_type, retrofit_potential,
                 building_certification, energy_star_rating, nabers_rating,
                 baseline_emission, scope1_emission, scope2_emission, scope3_emission,
                 carbon_intensity, annual_energy_cost, maintenance_cost, insurance_value,
                 owner, manager, tenant, lease_expiry, portfolio_id, business_unit,
                 region, tags, notes, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                property.id, property.property_id, property.name, property.address,
                property.city, property.state, property.postal_code, property.country,
                property.area_sqm, property.gross_floor_area, property.net_lettable_area,
                property.number_of_floors, property.year_built, property.year_renovated,
                property.occupancy_rate, property.operating_hours_per_day,
                property.operating_days_per_week, property.peak_occupancy,
                property.building_type.value, property.retrofit_potential.value,
                property.building_certification, property.energy_star_rating,
                property.nabers_rating, property.baseline_emission, property.scope1_emission,
                property.scope2_emission, property.scope3_emission, property.carbon_intensity,
                property.annual_energy_cost, property.maintenance_cost, property.insurance_value,
                property.owner, property.manager, property.tenant,
                property.lease_expiry.isoformat() if property.lease_expiry else None,
                property.portfolio_id, property.business_unit, property.region,
                json.dumps(property.tags), property.notes, property.is_active,
                property.created_at.isoformat(), property.updated_at.isoformat()
            ))
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving property: {e}")
            return False
    
    def load_property(self, property_id: str) -> Optional[Property]:
        """Load property from database"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM properties WHERE property_id = ?", (property_id,))
            row = cursor.fetchone()
            
            if row:
                # Convert row to Property object
                columns = [desc[0] for desc in cursor.description]
                data = dict(zip(columns, row))
                
                # Convert string fields back to appropriate types
                data['building_type'] = BuildingType(data['building_type'])
                data['retrofit_potential'] = RetrofitPotential(data['retrofit_potential'])
                data['tags'] = json.loads(data['tags']) if data['tags'] else []
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                
                if data['lease_expiry']:
                    data['lease_expiry'] = datetime.fromisoformat(data['lease_expiry'])
                
                return Property(**data)
        except Exception as e:
            logger.error(f"Error loading property: {e}")
        
        return None
    
    def get_all_properties(self) -> List[Property]:
        """Get all properties from database"""
        properties = []
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT property_id FROM properties WHERE is_active = 1")
            property_ids = [row[0] for row in cursor.fetchall()]
            
            for property_id in property_ids:
                property = self.load_property(property_id)
                if property:
                    properties.append(property)
        except Exception as e:
            logger.error(f"Error loading properties: {e}")
        
        return properties
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

# Factory classes for creating sample data
class DataFactory:
    """Factory for creating sample data"""
    
    @staticmethod
    def create_sample_properties(count: int = 9) -> List[Property]:
        """Create sample properties"""
        sample_properties = []
        
        property_configs = [
            {"id": "BP01", "name": "Brisbane Plaza", "type": BuildingType.OFFICE, "area": 1500, "retrofit": RetrofitPotential.HIGH, "baseline": 150000},
            {"id": "CB01", "name": "Central Building 01", "type": BuildingType.RETAIL, "area": 800, "retrofit": RetrofitPotential.MEDIUM, "baseline": 85000},
            {"id": "CB02", "name": "Central Building 02", "type": BuildingType.OFFICE, "area": 2000, "retrofit": RetrofitPotential.HIGH, "baseline": 220000},
            {"id": "CB03", "name": "Central Building 03", "type": BuildingType.RESIDENTIAL, "area": 600, "retrofit": RetrofitPotential.LOW, "baseline": 65000},
            {"id": "CB04", "name": "Central Building 04", "type": BuildingType.INDUSTRIAL, "area": 3000, "retrofit": RetrofitPotential.MEDIUM, "baseline": 380000},
            {"id": "HA1", "name": "Heritage Arcade", "type": BuildingType.MIXED_USE, "area": 1200, "retrofit": RetrofitPotential.HIGH, "baseline": 125000},
            {"id": "MP", "name": "Metro Plaza", "type": BuildingType.OFFICE, "area": 1800, "retrofit": RetrofitPotential.MEDIUM, "baseline": 195000},
            {"id": "TC01", "name": "Tech Center 01", "type": BuildingType.EDUCATIONAL, "area": 900, "retrofit": RetrofitPotential.MEDIUM, "baseline": 95000},
            {"id": "WH1", "name": "Warehouse One", "type": BuildingType.WAREHOUSE, "area": 2500, "retrofit": RetrofitPotential.LOW, "baseline": 180000}
        ]
        
        for i, config in enumerate(property_configs[:count]):
            property = Property(
                property_id=config["id"],
                name=config["name"],
                address=f"{100 + i} Example Street",
                city="Brisbane" if i < 3 else "Sydney" if i < 6 else "Melbourne",
                state="QLD" if i < 3 else "NSW" if i < 6 else "VIC",
                postal_code=f"{4000 + i}",
                area_sqm=config["area"],
                gross_floor_area=config["area"] * 1.1,
                net_lettable_area=config["area"] * 0.9,
                number_of_floors=max(1, config["area"] // 500),
                year_built=2005 + (i * 2),
                occupancy_rate=0.85 + (i * 0.02),
                operating_hours_per_day=8.0 + (i * 0.5),
                operating_days_per_week=5,
                peak_occupancy=int(config["area"] / 10),
                building_type=config["type"],
                retrofit_potential=config["retrofit"],
                baseline_emission=config["baseline"],
                scope1_emission=config["baseline"] * 0.6,
                scope2_emission=config["baseline"] * 0.3,
                scope3_emission=config["baseline"] * 0.1,
                annual_energy_cost=config["baseline"] * 0.8,
                portfolio_id="MAIN_PORTFOLIO",
                business_unit=f"BU_{i+1}",
                region="Australia"
            )
            property.calculate_carbon_intensity()
            sample_properties.append(property)
        
        return sample_properties
    
    @staticmethod
    def create_sample_reduction_options(count: int = 13) -> List[ReductionOption]:
        """Create sample reduction options"""
        option_configs = [
            {
                "id": "SOLAR001", "name": "Solar PV Installation", "category": "Renewable Energy",
                "co2_reduction": 4500, "capex": 200000, "opex": 12000, "priority": 5
            },
            {
                "id": "LED001", "name": "LED Lighting Upgrade", "category": "Energy Efficiency",
                "co2_reduction": 1200, "capex": 45000, "opex": 3000, "priority": 4
            },
            {
                "id": "HVAC001", "name": "HVAC System Upgrade", "category": "Energy Efficiency",
                "co2_reduction": 3500, "capex": 150000, "opex": 18000, "priority": 4
            },
            {
                "id": "INSUL001", "name": "Building Insulation", "category": "Building Envelope",
                "co2_reduction": 2000, "capex": 80000, "opex": 2500, "priority": 3
            },
            {
                "id": "SMART001", "name": "Smart Building Systems", "category": "Technology",
                "co2_reduction": 2200, "capex": 95000, "opex": 22000, "priority": 3
            },
            {
                "id": "CREDIT001", "name": "Carbon Credits", "category": "Offsetting",
                "co2_reduction": 800, "capex": 0, "opex": 35000, "priority": 2
            },
            {
                "id": "BIOFUEL001", "name": "Biofuel Replacement", "category": "Fuel Switching",
                "co2_reduction": 3000, "capex": 70000, "opex": 28000, "priority": 3
            },
            {
                "id": "WIND001", "name": "Wind Energy Installation", "category": "Renewable Energy",
                "co2_reduction": 3800, "capex": 180000, "opex": 15000, "priority": 3
            },
            {
                "id": "HEAT001", "name": "Heat Pump Systems", "category": "Heating/Cooling",
                "co2_reduction": 2800, "capex": 120000, "opex": 14000, "priority": 3
            },
            {
                "id": "STORAGE001", "name": "Energy Storage Systems", "category": "Energy Management",
                "co2_reduction": 1800, "capex": 140000, "opex": 8000, "priority": 2
            },
            {
                "id": "EV001", "name": "Electric Vehicle Fleet", "category": "Transportation",
                "co2_reduction": 1500, "capex": 85000, "opex": 12000, "priority": 3
            },
            {
                "id": "HYDRO001", "name": "Hydrogen Systems", "category": "Alternative Fuel",
                "co2_reduction": 2500, "capex": 160000, "opex": 20000, "priority": 2
            },
            {
                "id": "MONITOR001", "name": "Energy Monitoring Systems", "category": "Monitoring",
                "co2_reduction": 800, "capex": 35000, "opex": 8000, "priority": 4
            }
        ]
        
        sample_options = []
        for i, config in enumerate(option_configs[:count]):
            option = ReductionOption(
                option_id=config["id"],
                name=config["name"],
                description=f"Advanced {config['name'].lower()} for commercial buildings",
                category=config["category"],
                co2_reduction_potential=config["co2_reduction"],
                co2_reduction_percentage=5.0 + (i * 2),
                energy_savings_kwh=config["co2_reduction"] * 2.5,
                capex=config["capex"],
                opex=config["opex"],
                priority=PriorityLevel(config["priority"]),
                implementation_time_months=3 + (i % 6),
                implementation_complexity="Medium",
                risk_level=RiskLevel.LOW if i % 3 == 0 else RiskLevel.MEDIUM,
                technology_type=f"Type_{i+1}",
                vendor=f"Vendor_{i+1}",
                warranty_years=5 + (i % 3),
                expected_lifetime_years=15 + (i % 10),
                suitable_building_types=[BuildingType.OFFICE, BuildingType.RETAIL]
            )
            option.calculate_cost_effectiveness()
            sample_options.append(option)
        
        return sample_options
    
    @staticmethod
    def create_sample_emission_factors() -> List[EmissionFactor]:
        """Create sample emission factors for different fuel types"""
        factors = []
        
        factor_configs = [
            {"fuel": FuelType.ELECTRICITY, "factor": 0.82, "unit": "tCO2e/MWh", "region": "QLD"},
            {"fuel": FuelType.ELECTRICITY, "factor": 0.79, "unit": "tCO2e/MWh", "region": "NSW"},
            {"fuel": FuelType.ELECTRICITY, "factor": 1.02, "unit": "tCO2e/MWh", "region": "VIC"},
            {"fuel": FuelType.NATURAL_GAS, "factor": 0.0543, "unit": "tCO2e/GJ", "region": "Australia"},
            {"fuel": FuelType.DIESEL, "factor": 2.68, "unit": "tCO2e/kL", "region": "Australia"},
            {"fuel": FuelType.GASOLINE, "factor": 2.31, "unit": "tCO2e/kL", "region": "Australia"},
            {"fuel": FuelType.PROPANE, "factor": 1.51, "unit": "tCO2e/kL", "region": "Australia"}
        ]
        
        for config in factor_configs:
            factor = EmissionFactor(
                factor_id=f"{config['fuel'].value}_{config['region']}_{datetime.now().year}",
                fuel_type=config["fuel"],
                region=config["region"],
                emission_factor=config["factor"],
                unit=config["unit"],
                source="National Greenhouse Accounts",
                data_source="Australian Government",
                methodology="IPCC Guidelines"
            )
            factors.append(factor)
        
        return factors

# Validation utilities
class DataValidator:
    """Utility class for data validation"""
    
    @staticmethod
    def validate_property(property: Property) -> List[str]:
        """Validate property data and return list of errors"""
        errors = []
        
        if not property.property_id:
            errors.append("Property ID is required")
        
        if property.area_sqm <= 0:
            errors.append("Area must be greater than 0")
        
        if property.baseline_emission < 0:
            errors.append("Baseline emission cannot be negative")
        
        if property.year_built < 1900 or property.year_built > datetime.now().year:
            errors.append("Year built must be between 1900 and current year")
        
        if not (0 <= property.occupancy_rate <= 1):
            errors.append("Occupancy rate must be between 0 and 1")
        
        return errors
    
    @staticmethod
    def validate_reduction_option(option: ReductionOption) -> List[str]:
        """Validate reduction option data and return list of errors"""
        errors = []
        
        if not option.option_id:
            errors.append("Option ID is required")
        
        if option.co2_reduction_potential < 0:
            errors.append("CO2 reduction potential cannot be negative")
        
        if option.capex < 0:
            errors.append("CAPEX cannot be negative")
        
        if option.expected_lifetime_years <= 0:
            errors.append("Expected lifetime must be greater than 0")
        
        return errors
    
    @staticmethod
    def validate_milestone_scenario(scenario: MilestoneScenario) -> List[str]:
        """Validate milestone scenario data and return list of errors"""
        errors = []
        
        if not scenario.scenario_id:
            errors.append("Scenario ID is required")
        
        if scenario.target_year <= scenario.baseline_year:
            errors.append("Target year must be after baseline year")
        
        if not scenario.yearly_targets:
            errors.append("Yearly targets are required")
        
        # Validate that targets are decreasing over time
        years = sorted(scenario.yearly_targets.keys())
        for i in range(1, len(years)):
            if scenario.yearly_targets[years[i]] > scenario.yearly_targets[years[i-1]]:
                errors.append(f"Emissions should decrease over time (year {years[i]})")
        
        return errors

# Export utilities
class DataExporter:
    """Utility class for exporting data to various formats"""
    
    @staticmethod
    def export_properties_to_csv(properties: List[Property], filename: str):
        """Export properties to CSV file"""
        data = []
        for prop in properties:
            data.append({
                "Property ID": prop.property_id,
                "Name": prop.name,
                "City": prop.city,
                "Building Type": prop.building_type.value,
                "Area (sqm)": prop.area_sqm,
                "Baseline Emission (tCO2e)": prop.baseline_emission,
                "Retrofit Potential": prop.retrofit_potential.value,
                "Carbon Intensity": prop.carbon_intensity
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(properties)} properties to {filename}")
    
    @staticmethod
    def export_reduction_options_to_csv(options: List[ReductionOption], filename: str):
        """Export reduction options to CSV file"""
        data = []
        for option in options:
            data.append({
                "Option ID": option.option_id,
                "Name": option.name,
                "Category": option.category,
                "CO2 Reduction (tCO2e)": option.co2_reduction_potential,
                "CAPEX": option.capex,
                "OPEX": option.opex,
                "Priority": option.priority.value,
                "Implementation Time (months)": option.implementation_time_months,
                "Risk Level": option.risk_level.value
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(options)} reduction options to {filename}")

# Configuration management
class ConfigManager:
    """Configuration manager for system settings"""
    
    def __init__(self, config_file: str = "ecoassist_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "database": {
                "path": "ecoassist.db",
                "backup_interval_hours": 24
            },
            "calculations": {
                "default_target_year": 2050,
                "default_reduction_2030": 30,
                "default_reduction_2050": 80,
                "deviation_threshold": 0.05
            },
            "ui": {
                "default_property": "BP01",
                "charts_theme": "plotly_white",
                "table_page_size": 50
            },
            "api": {
                "rate_limit_per_hour": 1000,
                "timeout_seconds": 30
            },
            "logging": {
                "level": "INFO",
                "file": "ecoassist.log",
                "max_size_mb": 100
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    self._merge_config(default_config, loaded_config)
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
        
        return default_config
    
    def _merge_config(self, default: Dict, loaded: Dict):
        """Recursively merge loaded config with defaults"""
        for key, value in loaded.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self._merge_config(default[key], value)
                else:
                    default[key] = value
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save config file: {e}")
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value