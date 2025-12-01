#!/usr/bin/env python3
"""
EcoAssist Database Integration Layer
Replaces in-memory storage with SQL Server database connectivity
"""

import pyodbc
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from contextlib import contextmanager
import hashlib
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================================
# DATABASE CONFIGURATION
# ================================================================================

class DatabaseConfig:
    """Database configuration settings"""
    
    def __init__(self):
        # SQL Server connection settings
        self.server = "localhost"  # Change to your SQL Server instance
        self.database = "EcoAssistDB"
        self.username = "sa"  # Change to your SQL Server username
        self.password = "YourPassword123!"  # Change to your SQL Server password
        self.driver = "{ODBC Driver 17 for SQL Server}"
        
        # Connection string
        self.connection_string = (
            f"DRIVER={self.driver};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            "TrustServerCertificate=yes;"
        )
    
    def get_connection_string(self) -> str:
        """Get the database connection string"""
        return self.connection_string


class CarbonCreditPriceRepository:
    """Data access layer for carbon credit prices"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_by_year(self, year: int) -> List[Dict]:
        """Get carbon credit prices for a specific year"""
        query = """
        SELECT credit_type, year, price_per_tonne, currency, market_region,
               vintage_year, certification_standard, project_type, trading_volume,
               price_volatility, co_benefits, data_source, data_quality_score
        FROM carbon_credit_prices
        WHERE year = ?
        ORDER BY credit_type
        """
        return self.db.execute_query(query, (year,))
    
    def get_by_type_and_year_range(self, credit_type: str, start_year: int, end_year: int) -> List[Dict]:
        """Get price trends for a specific credit type"""
        query = """
        SELECT year, price_per_tonne, price_volatility, trading_volume
        FROM carbon_credit_prices
        WHERE credit_type = ? AND year BETWEEN ? AND ?
        ORDER BY year
        """
        return self.db.execute_query(query, (credit_type, start_year, end_year))

class RenewableEnergyPriceRepository:
    """Data access layer for renewable energy prices"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_by_year(self, year: int, energy_type: Optional[str] = None) -> List[Dict]:
        """Get renewable energy prices for a specific year"""
        if energy_type:
            query = """
            SELECT energy_type, year, price_per_mwh, price_per_kw_installed, currency,
                   region, technology_specification, capacity_range, capex, opex_annual,
                   capacity_factor, expected_lifetime_years, feed_in_tariff
            FROM renewable_energy_prices
            WHERE year = ? AND energy_type = ?
            ORDER BY energy_type
            """
            return self.db.execute_query(query, (year, energy_type))
        else:
            query = """
            SELECT energy_type, year, price_per_mwh, price_per_kw_installed, currency,
                   region, technology_specification, capacity_range, capex, opex_annual,
                   capacity_factor, expected_lifetime_years, feed_in_tariff
            FROM renewable_energy_prices
            WHERE year = ?
            ORDER BY energy_type
            """
            return self.db.execute_query(query, (year,))
    
    def get_lcoe_trend(self, energy_type: str, start_year: int, end_year: int) -> List[Dict]:
        """Get LCOE trend for specific technology"""
        query = """
        SELECT year, price_per_mwh as lcoe, capacity_factor
        FROM renewable_energy_prices
        WHERE energy_type = ? AND year BETWEEN ? AND ?
        ORDER BY year
        """
        return self.db.execute_query(query, (energy_type, start_year, end_year))

class RenewableFuelPriceRepository:
    """Data access layer for renewable fuel prices"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_by_year(self, year: int, fuel_type: Optional[str] = None) -> List[Dict]:
        """Get renewable fuel prices for a specific year"""
        if fuel_type:
            query = """
            SELECT fuel_type, year, price_per_unit, unit_type, currency, region,
                   feedstock_type, production_pathway, energy_content_mj, emission_factor,
                   fossil_fuel_price_equivalent, price_premium_percentage
            FROM renewable_fuel_prices
            WHERE year = ? AND fuel_type = ?
            ORDER BY fuel_type
            """
            return self.db.execute_query(query, (year, fuel_type))
        else:
            query = """
            SELECT fuel_type, year, price_per_unit, unit_type, currency, region,
                   feedstock_type, production_pathway, energy_content_mj, emission_factor,
                   fossil_fuel_price_equivalent, price_premium_percentage
            FROM renewable_fuel_prices
            WHERE year = ?
            ORDER BY fuel_type
            """
            return self.db.execute_query(query, (year,))
    
    def get_price_comparison(self, fuel_type: str, year: int) -> Dict:
        """Compare renewable fuel price with fossil fuel equivalent"""
        query = """
        SELECT fuel_type, price_per_unit, fossil_fuel_price_equivalent,
               price_premium_percentage, unit_type
        FROM renewable_fuel_prices
        WHERE fuel_type = ? AND year = ?
        """
        results = self.db.execute_query(query, (fuel_type, year))
        return results[0] if results else None

# Update EcoAssistBackend class initialization
class EcoAssistBackend:
    """Updated EcoAssist backend using SQL Server database"""
    
    def __init__(self, config: DatabaseConfig = None):
        # ... existing initialization code ...
        
        # Initialize new pricing repositories
        self.carbon_credit_repo = CarbonCreditPriceRepository(self.db_manager)
        self.renewable_energy_repo = RenewableEnergyPriceRepository(self.db_manager)
        self.renewable_fuel_repo = RenewableFuelPriceRepository(self.db_manager)
        
        # Load pricing data
        self._load_pricing_data()
    
    def _load_pricing_data(self):
        """Load pricing data from database"""
        try:
            current_year = datetime.now().year
            
            self.carbon_credit_prices = self.carbon_credit_repo.get_by_year(current_year)
            self.renewable_energy_prices = self.renewable_energy_repo.get_by_year(current_year)
            self.renewable_fuel_prices = self.renewable_fuel_repo.get_by_year(current_year)
            
            logger.info(f"Loaded {len(self.carbon_credit_prices)} carbon credit prices")
            logger.info(f"Loaded {len(self.renewable_energy_prices)} renewable energy prices")
            logger.info(f"Loaded {len(self.renewable_fuel_prices)} renewable fuel prices")
            
        except Exception as e:
            logger.error(f"Error loading pricing data: {e}")
            self.carbon_credit_prices = []
            self.renewable_energy_prices = []
            self.renewable_fuel_prices = []

# ================================================================================
# DATA MODELS (updated to match database schema)
# ================================================================================

@dataclass
class Property:
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
    carbon_intensity: Optional[float] = None
    annual_energy_cost: float = 0.0
    portfolio_id: str = "DEFAULT"
    region: Optional[str] = None
    is_active: bool = True

@dataclass
class ReductionOption:
    """Reduction option data model matching database schema"""
    option_id: str
    name: str
    description: Optional[str] = None
    category: str = "Energy Efficiency"
    co2_reduction_potential: float = 0.0
    capex: float = 0.0
    opex: float = 0.0
    priority: int = 3
    implementation_time_months: int = 6
    risk_level: str = "Medium"
    technology_type: Optional[str] = None

@dataclass
class MilestoneScenario:
    """Milestone scenario data model matching database schema"""
    scenario_id: str
    name: str
    description: Optional[str] = None
    target_year: int = 2050
    yearly_targets: Dict[str, float] = None
    total_capex: float = 0.0
    total_opex: float = 0.0
    reduction_rate_2030: float = 0.0
    reduction_rate_2050: float = 0.0
    strategy_type: str = "balanced"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.yearly_targets is None:
            self.yearly_targets = {}


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
    co_benefits: List[str] = None
    data_source: str = ""
    data_quality_score: float = 100.0
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.co_benefits is None:
            self.co_benefits = []

@dataclass
class RenewableEnergyPrice:
    """Renewable energy pricing data model"""
    energy_type: str  # solar_pv, wind, hydro, battery_storage
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
    fuel_type: str  # biodiesel, hydrogen, biogas, sustainable_aviation_fuel
    year: int
    price_per_unit: float
    unit_type: str  # liter, kg, GJ, m3
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

@dataclass
class StrategicPattern:
    """Strategic pattern data model matching database schema"""
    pattern_id: str
    name: str
    description: Optional[str] = None
    reduction_options: List[str] = None
    estimated_cost: float = 0.0
    estimated_reduction: float = 0.0
    risk_level: str = "Medium"
    
    def __post_init__(self):
        if self.reduction_options is None:
            self.reduction_options = []

# ================================================================================
# DATABASE CONNECTION AND UTILITY CLASSES
# ================================================================================

class DatabaseManager:
    """Database connection and query management"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection_string = config.get_connection_string()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = pyodbc.connect(self.connection_string)
            conn.autocommit = False
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: Tuple = None) -> List[Dict]:
        """Execute a SELECT query and return results as list of dictionaries"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Get column names
            columns = [column[0] for column in cursor.description]
            
            # Fetch results and convert to dictionaries
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
    
    def execute_non_query(self, query: str, params: Tuple = None) -> int:
        """Execute INSERT, UPDATE, DELETE queries"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            affected_rows = cursor.rowcount
            conn.commit()
            return affected_rows
    
    def execute_scalar(self, query: str, params: Tuple = None) -> Any:
        """Execute query and return single value"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            result = cursor.fetchone()
            return result[0] if result else None

# ================================================================================
# DATABASE DATA ACCESS LAYER
# ================================================================================

class PropertyRepository:
    """Data access layer for properties"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_all_active(self) -> List[Property]:
        """Get all active properties"""
        query = """
        SELECT property_id, name, address, city, state, area_sqm, building_type,
               retrofit_potential, baseline_emission, scope1_emission, scope2_emission,
               carbon_intensity, annual_energy_cost, portfolio_id, region, is_active
        FROM properties 
        WHERE is_active = 1
        ORDER BY property_id
        """
        
        results = self.db.execute_query(query)
        properties = []
        
        for row in results:
            properties.append(Property(
                property_id=row['property_id'],
                name=row['name'],
                address=row['address'],
                city=row['city'],
                state=row['state'],
                area_sqm=float(row['area_sqm']) if row['area_sqm'] else 0.0,
                building_type=row['building_type'],
                retrofit_potential=row['retrofit_potential'],
                baseline_emission=float(row['baseline_emission']) if row['baseline_emission'] else 0.0,
                scope1_emission=float(row['scope1_emission']) if row['scope1_emission'] else 0.0,
                scope2_emission=float(row['scope2_emission']) if row['scope2_emission'] else 0.0,
                carbon_intensity=float(row['carbon_intensity']) if row['carbon_intensity'] else None,
                annual_energy_cost=float(row['annual_energy_cost']) if row['annual_energy_cost'] else 0.0,
                portfolio_id=row['portfolio_id'],
                region=row['region'],
                is_active=bool(row['is_active'])
            ))
        
        return properties
    
    def get_by_id(self, property_id: str) -> Optional[Property]:
        """Get property by ID"""
        query = """
        SELECT property_id, name, address, city, state, area_sqm, building_type,
               retrofit_potential, baseline_emission, scope1_emission, scope2_emission,
               carbon_intensity, annual_energy_cost, portfolio_id, region, is_active
        FROM properties 
        WHERE property_id = ? AND is_active = 1
        """
        
        results = self.db.execute_query(query, (property_id,))
        if not results:
            return None
        
        row = results[0]
        return Property(
            property_id=row['property_id'],
            name=row['name'],
            address=row['address'],
            city=row['city'],
            state=row['state'],
            area_sqm=float(row['area_sqm']) if row['area_sqm'] else 0.0,
            building_type=row['building_type'],
            retrofit_potential=row['retrofit_potential'],
            baseline_emission=float(row['baseline_emission']) if row['baseline_emission'] else 0.0,
            scope1_emission=float(row['scope1_emission']) if row['scope1_emission'] else 0.0,
            scope2_emission=float(row['scope2_emission']) if row['scope2_emission'] else 0.0,
            carbon_intensity=float(row['carbon_intensity']) if row['carbon_intensity'] else None,
            annual_energy_cost=float(row['annual_energy_cost']) if row['annual_energy_cost'] else 0.0,
            portfolio_id=row['portfolio_id'],
            region=row['region'],
            is_active=bool(row['is_active'])
        )

class ReductionOptionRepository:
    """Data access layer for reduction options"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_all(self) -> List[ReductionOption]:
        """Get all reduction options"""
        query = """
        SELECT option_id, name, description, category, co2_reduction_potential,
               capex, opex, priority, implementation_time_months, risk_level, technology_type
        FROM reduction_options
        ORDER BY priority, name
        """
        
        results = self.db.execute_query(query)
        options = []
        
        for row in results:
            options.append(ReductionOption(
                option_id=row['option_id'],
                name=row['name'],
                description=row['description'],
                category=row['category'],
                co2_reduction_potential=float(row['co2_reduction_potential']) if row['co2_reduction_potential'] else 0.0,
                capex=float(row['capex']) if row['capex'] else 0.0,
                opex=float(row['opex']) if row['opex'] else 0.0,
                priority=int(row['priority']) if row['priority'] else 3,
                implementation_time_months=int(row['implementation_time_months']) if row['implementation_time_months'] else 6,
                risk_level=row['risk_level'],
                technology_type=row['technology_type']
            ))
        
        return options

class MilestoneScenarioRepository:
    """Data access layer for milestone scenarios"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def create(self, scenario: MilestoneScenario) -> str:
        """Create new milestone scenario"""
        query = """
        INSERT INTO milestone_scenarios (
            scenario_id, name, description, target_year, yearly_targets,
            total_capex, total_opex, reduction_rate_2030, reduction_rate_2050, strategy_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        yearly_targets_json = json.dumps(scenario.yearly_targets)
        
        self.db.execute_non_query(query, (
            scenario.scenario_id,
            scenario.name,
            scenario.description,
            scenario.target_year,
            yearly_targets_json,
            scenario.total_capex,
            scenario.total_opex,
            scenario.reduction_rate_2030,
            scenario.reduction_rate_2050,
            scenario.strategy_type
        ))
        
        return scenario.scenario_id
    
    def get_all(self) -> List[MilestoneScenario]:
        """Get all milestone scenarios"""
        query = """
        SELECT scenario_id, name, description, target_year, yearly_targets,
               total_capex, total_opex, reduction_rate_2030, reduction_rate_2050, 
               strategy_type, created_at, updated_at
        FROM milestone_scenarios
        ORDER BY created_at DESC
        """
        
        results = self.db.execute_query(query)
        scenarios = []
        
        for row in results:
            yearly_targets = json.loads(row['yearly_targets']) if row['yearly_targets'] else {}
            
            scenarios.append(MilestoneScenario(
                scenario_id=row['scenario_id'],
                name=row['name'],
                description=row['description'],
                target_year=int(row['target_year']),
                yearly_targets=yearly_targets,
                total_capex=float(row['total_capex']) if row['total_capex'] else 0.0,
                total_opex=float(row['total_opex']) if row['total_opex'] else 0.0,
                reduction_rate_2030=float(row['reduction_rate_2030']) if row['reduction_rate_2030'] else 0.0,
                reduction_rate_2050=float(row['reduction_rate_2050']) if row['reduction_rate_2050'] else 0.0,
                strategy_type=row['strategy_type'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            ))
        
        return scenarios
    
    def get_by_id(self, scenario_id: str) -> Optional[MilestoneScenario]:
        """Get milestone scenario by ID"""
        query = """
        SELECT scenario_id, name, description, target_year, yearly_targets,
               total_capex, total_opex, reduction_rate_2030, reduction_rate_2050, 
               strategy_type, created_at, updated_at
        FROM milestone_scenarios
        WHERE scenario_id = ?
        """
        
        results = self.db.execute_query(query, (scenario_id,))
        if not results:
            return None
        
        row = results[0]
        yearly_targets = json.loads(row['yearly_targets']) if row['yearly_targets'] else {}
        
        return MilestoneScenario(
            scenario_id=row['scenario_id'],
            name=row['name'],
            description=row['description'],
            target_year=int(row['target_year']),
            yearly_targets=yearly_targets,
            total_capex=float(row['total_capex']) if row['total_capex'] else 0.0,
            total_opex=float(row['total_opex']) if row['total_opex'] else 0.0,
            reduction_rate_2030=float(row['reduction_rate_2030']) if row['reduction_rate_2030'] else 0.0,
            reduction_rate_2050=float(row['reduction_rate_2050']) if row['reduction_rate_2050'] else 0.0,
            strategy_type=row['strategy_type'],
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )

class StrategicPatternRepository:
    """Data access layer for strategic patterns"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_all(self) -> List[StrategicPattern]:
        """Get all strategic patterns"""
        query = """
        SELECT pattern_id, name, description, reduction_options, estimated_cost,
               estimated_reduction, risk_level
        FROM strategic_patterns
        ORDER BY name
        """
        
        results = self.db.execute_query(query)
        patterns = []
        
        for row in results:
            reduction_options = json.loads(row['reduction_options']) if row['reduction_options'] else []
            
            patterns.append(StrategicPattern(
                pattern_id=row['pattern_id'],
                name=row['name'],
                description=row['description'],
                reduction_options=reduction_options,
                estimated_cost=float(row['estimated_cost']) if row['estimated_cost'] else 0.0,
                estimated_reduction=float(row['estimated_reduction']) if row['estimated_reduction'] else 0.0,
                risk_level=row['risk_level']
            ))
        
        return patterns

class HistoricalDataRepository:
    """Data access layer for historical data"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_consumption_data(self, property_id: str, year: int = None) -> pd.DataFrame:
        """Get historical consumption data"""
        if year:
            query = """
            SELECT property_id, fuel_type, consumption_period, consumption_amount,
                   consumption_unit, unit_cost, total_cost, emission_factor
            FROM historical_consumption
            WHERE property_id = ? AND consumption_period LIKE ?
            ORDER BY consumption_period, fuel_type
            """
            params = (property_id, f"{year}-%")
        else:
            query = """
            SELECT property_id, fuel_type, consumption_period, consumption_amount,
                   consumption_unit, unit_cost, total_cost, emission_factor
            FROM historical_consumption
            WHERE property_id = ?
            ORDER BY consumption_period, fuel_type
            """
            params = (property_id,)
        
        results = self.db.execute_query(query, params)
        return pd.DataFrame(results)
    
    def get_emissions_data(self, property_id: str, year: int = None) -> pd.DataFrame:
        """Get historical emissions data"""
        if year:
            query = """
            SELECT property_id, emission_period, fuel_type, emission_scope,
                   emissions_tco2e, emission_factor, consumption_amount
            FROM historical_emissions
            WHERE property_id = ? AND emission_period LIKE ?
            ORDER BY emission_period, fuel_type
            """
            params = (property_id, f"{year}-%")
        else:
            query = """
            SELECT property_id, emission_period, fuel_type, emission_scope,
                   emissions_tco2e, emission_factor, consumption_amount
            FROM historical_emissions
            WHERE property_id = ?
            ORDER BY emission_period, fuel_type
            """
            params = (property_id,)
        
        results = self.db.execute_query(query, params)
        return pd.DataFrame(results)

# ================================================================================
# UPDATED ECOASSIST BACKEND WITH DATABASE INTEGRATION
# ================================================================================

class EcoAssistBackend:
    """Updated EcoAssist backend using SQL Server database"""
    
    def __init__(self, config: DatabaseConfig = None):
        # Initialize database
        if config is None:
            config = DatabaseConfig()
        
        self.db_manager = DatabaseManager(config)
        
        # Initialize repositories
        self.property_repo = PropertyRepository(self.db_manager)
        self.reduction_option_repo = ReductionOptionRepository(self.db_manager)
        self.milestone_repo = MilestoneScenarioRepository(self.db_manager)
        self.pattern_repo = StrategicPatternRepository(self.db_manager)
        self.historical_repo = HistoricalDataRepository(self.db_manager)
        
        # Load data from database
        self._load_data()
    
    def _load_data(self):
        """Load data from database"""
        try:
            self.properties = self.property_repo.get_all_active()
            self.reduction_options = self.reduction_option_repo.get_all()
            self.milestone_scenarios = self.milestone_repo.get_all()
            self.strategic_patterns = self.pattern_repo.get_all()
            
            logger.info(f"Loaded {len(self.properties)} properties")
            logger.info(f"Loaded {len(self.reduction_options)} reduction options")
            logger.info(f"Loaded {len(self.milestone_scenarios)} milestone scenarios")
            logger.info(f"Loaded {len(self.strategic_patterns)} strategic patterns")
            
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            # Fallback to empty lists
            self.properties = []
            self.reduction_options = []
            self.milestone_scenarios = []
            self.strategic_patterns = []
    
    def generate_milestone_scenarios(self, target_year: int, reduction_2030: float, reduction_2050: float) -> List[MilestoneScenario]:
        """Generate milestone scenarios and save to database"""
        scenarios = []
        
        # Calculate total portfolio baseline
        total_baseline = sum(prop.baseline_emission for prop in self.properties)
        
        # Generate different scenario types
        scenario_configs = [
            {
                "name": "Conservative Approach",
                "description": "Gradual emission reduction with proven technologies and manageable investment levels",
                "strategy_type": "conservative",
                "capex_multiplier": 0.8,
                "opex_multiplier": 0.9,
                "reduction_adjustment": 0.9
            },
            {
                "name": "Aggressive Decarbonisation", 
                "description": "Front-loaded reduction with advanced technologies and higher initial investment",
                "strategy_type": "aggressive",
                "capex_multiplier": 1.3,
                "opex_multiplier": 1.2,
                "reduction_adjustment": 1.1
            },
            {
                "name": "Balanced Strategy",
                "description": "Balanced approach combining multiple technologies with moderate risk and investment",
                "strategy_type": "balanced", 
                "capex_multiplier": 1.0,
                "opex_multiplier": 1.0,
                "reduction_adjustment": 1.0
            },
            {
                "name": "Technology Focus",
                "description": "Emphasis on cutting-edge technologies with higher risk but maximum potential reduction",
                "strategy_type": "technology",
                "capex_multiplier": 1.5,
                "opex_multiplier": 1.4,
                "reduction_adjustment": 1.2
            }
        ]
        
        for config in scenario_configs:
            scenario_id = f"MS_{int(time.time())}_{hashlib.md5(config['name'].encode()).hexdigest()[:8]}"
            
            # Calculate yearly targets
            yearly_targets = self._calculate_yearly_targets(
                total_baseline, reduction_2030, reduction_2050, target_year, config['reduction_adjustment']
            )
            
            # Calculate costs
            base_capex = 2500000 * config['capex_multiplier']
            base_opex = 850000 * config['opex_multiplier']
            
            scenario = MilestoneScenario(
                scenario_id=scenario_id,
                name=config['name'],
                description=config['description'],
                target_year=target_year,
                yearly_targets=yearly_targets,
                total_capex=base_capex,
                total_opex=base_opex,
                reduction_rate_2030=reduction_2030,
                reduction_rate_2050=reduction_2050,
                strategy_type=config['strategy_type']
            )
            
            # Save to database
            try:
                self.milestone_repo.create(scenario)
                scenarios.append(scenario)
                logger.info(f"Created milestone scenario: {scenario.name}")
            except Exception as e:
                logger.error(f"Error saving scenario {scenario.name}: {e}")
        
        # Refresh milestone scenarios from database
        self.milestone_scenarios = self.milestone_repo.get_all()
        
        return scenarios
    
    def _calculate_yearly_targets(self, baseline: float, reduction_2030: float, reduction_2050: float, target_year: int, adjustment: float) -> Dict[str, float]:
        """Calculate yearly emission targets"""
        targets = {}
        
        # Apply adjustment factor
        adj_reduction_2030 = reduction_2030 * adjustment
        adj_reduction_2050 = reduction_2050 * adjustment
        
        # Calculate key year targets
        target_2030 = baseline * (1 - adj_reduction_2030 / 100)
        target_2050 = baseline * (1 - adj_reduction_2050 / 100)
        
        # Generate yearly progression
        years = [2025, 2030, 2035, 2040, 2045, 2050]
        if target_year not in years and target_year > 2050:
            years.append(target_year)
        
        for year in years:
            if year <= 2030:
                # Linear progression to 2030
                progress = (year - 2025) / (2030 - 2025)
                targets[str(year)] = baseline - (baseline - target_2030) * progress
            elif year <= 2050:
                # Linear progression from 2030 to 2050
                progress = (year - 2030) / (2050 - 2030)
                targets[str(year)] = target_2030 - (target_2030 - target_2050) * progress
            else:
                # Beyond 2050
                targets[str(year)] = target_2050
        
        return targets
    
    def calculate_property_breakdown(self, scenario_name: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Calculate property-level target breakdown"""
        try:
            # Find scenario by name
            scenario = None
            for s in self.milestone_scenarios:
                if s.name == scenario_name:
                    scenario = s
                    break
            
            if not scenario:
                return None, f"Scenario '{scenario_name}' not found"
            
            # Calculate property breakdown
            breakdown_data = []
            
            for prop in self.properties:
                # Calculate allocation weight based on carbon intensity
                total_baseline = sum(p.baseline_emission for p in self.properties)
                weight = prop.baseline_emission / total_baseline if total_baseline > 0 else 0
                
                # Get 2030 target from scenario
                target_2030 = scenario.yearly_targets.get('2030', prop.baseline_emission)
                portfolio_reduction = (total_baseline - target_2030) / total_baseline
                
                # Apply weight to property
                property_target = prop.baseline_emission * (1 - portfolio_reduction)
                reduction_amount = prop.baseline_emission - property_target
                reduction_rate = (reduction_amount / prop.baseline_emission * 100) if prop.baseline_emission > 0 else 0
                
                breakdown_data.append({
                    'Property': prop.property_id,
                    'NLA (m²)': f"{prop.area_sqm:,.0f}",
                    'Baseline (tCO₂e)': f"{prop.baseline_emission:,.0f}",
                    'Year': 2030,
                    'Target (tCO₂e)': f"{property_target:,.0f}",
                    'Reduction Rate': f"{reduction_rate:.1f}%",
                    'Carbon Intensity (tCO₂e/m²)': f"{property_target/prop.area_sqm:.2f}" if prop.area_sqm > 0 else "0.00",
                    'Building Type': prop.building_type,
                    'Retrofit Potential': prop.retrofit_potential
                })
            
            df = pd.DataFrame(breakdown_data)
            return df, "Success"
            
        except Exception as e:
            logger.error(f"Error in calculate_property_breakdown: {e}")
            return None, f"Calculation error: {str(e)}"
    
    def analyze_strategic_pattern(self, pattern_name: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], str]:
        """Analyze strategic pattern"""
        try:
            # Find pattern by name
            pattern = None
            for p in self.strategic_patterns:
                if p.name == pattern_name:
                    pattern = p
                    break
            
            if not pattern:
                return None, {}, f"Strategic pattern '{pattern_name}' not found"
            
            # Create analysis details
            analysis_data = []
            
            # Get reduction options for this pattern
            pattern_options = []
            if isinstance(pattern.reduction_options, list):
                option_ids = pattern.reduction_options
            else:
                # Parse JSON if stored as string
                option_ids = json.loads(pattern.reduction_options) if pattern.reduction_options else []
            
            for option_id in option_ids:
                for option in self.reduction_options:
                    if option.option_id == option_id:
                        pattern_options.append(option)
                        break
            
            # Build analysis data
            for option in pattern_options:
                analysis_data.append({
                    'Option': option.name,
                    'Category': option.category,
                    'CO2 Reduction (tCO₂e)': option.co2_reduction_potential,
                    'CAPEX (AUD)': option.capex,
                    'OPEX (AUD)': option.opex,
                    'Priority': option.priority,
                    'Risk Level': option.risk_level,
                    'Implementation (months)': option.implementation_time_months
                })
            
            # Summary information
            summary_info = {
                'pattern_name': pattern.name,
                'total_reduction': pattern.estimated_reduction,
                'total_cost': pattern.estimated_cost,
                'risk_level': pattern.risk_level,
                'option_count': len(pattern_options)
            }
            
            df = pd.DataFrame(analysis_data)
            return df, summary_info, "Success"
            
        except Exception as e:
            logger.error(f"Error in analyze_strategic_pattern: {e}")
            return None, {}, f"Analysis error: {str(e)}"
    
    def reoptimize_annual_plan(self, property_id: str, deviation_threshold: float) -> Tuple[Optional[Dict], Optional[pd.DataFrame], Dict[str, Any], str]:
        """Reoptimize annual plan based on actual performance"""
        try:
            # Get property
            property_obj = self.property_repo.get_by_id(property_id)
            if not property_obj:
                return None, None, {}, f"Property '{property_id}' not found"
            
            # Get historical data
            emissions_df = self.historical_repo.get_emissions_data(property_id, 2023)
            consumption_df = self.historical_repo.get_consumption_data(property_id, 2023)
            
            # Mock reoptimization calculation (simplified)
            monthly_targets = [12500, 12000, 11800, 11500, 11200, 11000, 10800, 10600, 10400, 10200, 10000, 9800]
            monthly_actuals = [13200, 12800, 12600, 12300, 11900, 11700, 11400, 11100, 10900, 10700, 10500, 10300]
            
            # Calculate variances
            variances = [(actual - target) / target * 100 for actual, target in zip(monthly_actuals, monthly_targets)]
            max_deviation = max([abs(v) for v in variances])
            
            needs_reoptimization = max_deviation > deviation_threshold
            
            plot_data = {
                'monthly_targets': monthly_targets,
                'monthly_actuals': monthly_actuals,
                'total_variance': variances,
                'needs_reoptimization': needs_reoptimization,
                'emission_refined_target': [t * 0.95 for t in monthly_targets] if needs_reoptimization else monthly_targets,
                'cost_refined_target': [17000, 16800, 16600, 16400, 16200, 16000, 15800, 15600, 15400, 15200, 15000, 14800]
            }
            
            analysis_summary = {
                'property_id': property_id,
                'building_type': property_obj.building_type,
                'ytd_emission_actual': f"{sum(monthly_actuals):,} tCO₂e",
                'ytd_emission_target': f"{sum(monthly_targets):,} tCO₂e",
                'emission_deviation': f"{(sum(monthly_actuals) - sum(monthly_targets)) / sum(monthly_targets) * 100:+.1f}%"
            }
            
            return plot_data, consumption_df, analysis_summary, "Success"
            
        except Exception as e:
            logger.error(f"Error in reoptimize_annual_plan: {e}")
            return None, None, {}, f"Reoptimization error: {str(e)}"

# ================================================================================
# UTILITY FUNCTIONS FOR DATABASE SETUP
# ================================================================================

def test_database_connection(config: DatabaseConfig = None) -> bool:
    """Test database connection"""
    if config is None:
        config = DatabaseConfig()
    
    try:
        db_manager = DatabaseManager(config)
        result = db_manager.execute_scalar("SELECT COUNT(*) FROM properties")
        logger.info(f"Database connection successful. Found {result} properties.")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

def initialize_database_backend() -> EcoAssistBackend:
    """Initialize the EcoAssist backend with database"""
    try:
        config = DatabaseConfig()
        
        # Test connection first
        if not test_database_connection(config):
            raise Exception("Database connection test failed")
        
        # Initialize backend
        backend = EcoAssistBackend(config)
        logger.info("EcoAssist backend initialized successfully with database")
        return backend
        
    except Exception as e:
        logger.error(f"Failed to initialize database backend: {e}")
        raise

# ================================================================================
# SAMPLE USAGE AND TESTING
# ================================================================================

if __name__ == "__main__":
    # Test the database integration
    print("Testing EcoAssist Database Integration...")
    
    try:
        # Initialize backend
        backend = initialize_database_backend()
        
        # Test basic functionality
        print(f"\nLoaded Data Summary:")
        print(f"  Properties: {len(backend.properties)}")
        print(f"  Reduction Options: {len(backend.reduction_options)}")
        print(f"  Milestone Scenarios: {len(backend.milestone_scenarios)}")
        print(f"  Strategic Patterns: {len(backend.strategic_patterns)}")
        
        # Test milestone generation
        print(f"\nTesting milestone generation...")
        scenarios = backend.generate_milestone_scenarios(2050, 30.0, 80.0)
        print(f"Generated {len(scenarios)} new scenarios")
        
        # Test property breakdown
        if scenarios:
            print(f"\nTesting property breakdown...")
            df, status = backend.calculate_property_breakdown(scenarios[0].name)
            if df is not None:
                print(f"Property breakdown completed: {len(df)} properties")
                print("Sample data:")
                print(df.head())
            else:
                print(f"Property breakdown failed: {status}")
        
        # Test strategic pattern analysis
        if backend.strategic_patterns:
            print(f"\nTesting strategic pattern analysis...")
            pattern_name = backend.strategic_patterns[0].name
            details_df, summary, status = backend.analyze_strategic_pattern(pattern_name)
            if details_df is not None:
                print(f"Strategic analysis completed for '{pattern_name}'")
                print(f"Summary: {summary}")
            else:
                print(f"Strategic analysis failed: {status}")
        
        print(f"\n✅ Database integration test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Database integration test failed: {e}")
        import traceback
        traceback.print_exc()
    