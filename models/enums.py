# models/enums.py - Enumeration Types for EcoAssist System
"""
Comprehensive enumeration types for the EcoAssist carbon reduction planning system.
All enums are used for data validation, database constraints, and API responses.
"""

from enum import Enum

# =============================================================================
# PROPERTY AND BUILDING ENUMERATIONS
# =============================================================================

class BuildingType(str, Enum):
    """Building type classification for properties"""
    OFFICE = "Office"
    RETAIL = "Retail"
    RESIDENTIAL = "Residential"
    INDUSTRIAL = "Industrial"
    MIXED_USE = "Mixed-Use"
    EDUCATIONAL = "Educational"
    WAREHOUSE = "Warehouse"
    HEALTHCARE = "Healthcare"
    HOSPITALITY = "Hospitality"
    DATA_CENTER = "Data Center"
    GOVERNMENT = "Government"
    RELIGIOUS = "Religious"
    OTHER = "Other"

class RetrofitPotential(str, Enum):
    """Classification of retrofit potential for properties"""
    CRITICAL = "Critical"  # Immediate action required
    HIGH = "High"         # Significant improvement potential
    MEDIUM = "Medium"     # Moderate improvement potential
    LOW = "Low"          # Limited improvement potential
    MINIMAL = "Minimal"   # Already optimized

class PropertyStatus(str, Enum):
    """Operational status of property"""
    ACTIVE = "Active"
    INACTIVE = "Inactive"
    UNDER_RENOVATION = "Under Renovation"
    PLANNED = "Planned"
    DECOMMISSIONED = "Decommissioned"

# =============================================================================
# PRIORITY AND RISK ENUMERATIONS
# =============================================================================

class PriorityLevel(str, Enum):
    """Priority level for actions and measures"""
    VERY_HIGH = "Very High"  # Priority 5
    HIGH = "High"            # Priority 4
    MEDIUM = "Medium"        # Priority 3
    LOW = "Low"              # Priority 2
    VERY_LOW = "Very Low"    # Priority 1
    
    @property
    def numeric_value(self) -> int:
        """Get numeric priority value (1-5)"""
        priority_map = {
            "Very High": 5,
            "High": 4,
            "Medium": 3,
            "Low": 2,
            "Very Low": 1
        }
        return priority_map.get(self.value, 3)

class RiskLevel(str, Enum):
    """Risk level classification"""
    CRITICAL = "Critical"  # Immediate attention required
    HIGH = "High"         # Significant risk
    MEDIUM = "Medium"     # Moderate risk
    LOW = "Low"          # Minor risk
    MINIMAL = "Minimal"   # Negligible risk

class UrgencyLevel(str, Enum):
    """Urgency level for implementation"""
    IMMEDIATE = "Immediate"  # Within 1 month
    URGENT = "Urgent"        # Within 3 months
    NORMAL = "Normal"        # Within 6 months
    DEFERRED = "Deferred"    # Within 12 months

# =============================================================================
# EMISSION SCOPE ENUMERATIONS
# =============================================================================

class EmissionScope(str, Enum):
    """GHG Protocol emission scopes"""
    SCOPE_1 = "Scope 1"  # Direct emissions
    SCOPE_2 = "Scope 2"  # Indirect emissions from purchased energy
    SCOPE_3 = "Scope 3"  # Other indirect emissions

class EmissionCategory(str, Enum):
    """Detailed emission categories"""
    # Scope 1
    STATIONARY_COMBUSTION = "Stationary Combustion"
    MOBILE_COMBUSTION = "Mobile Combustion"
    FUGITIVE_EMISSIONS = "Fugitive Emissions"
    PROCESS_EMISSIONS = "Process Emissions"
    
    # Scope 2
    PURCHASED_ELECTRICITY = "Purchased Electricity"
    PURCHASED_HEATING = "Purchased Heating"
    PURCHASED_COOLING = "Purchased Cooling"
    PURCHASED_STEAM = "Purchased Steam"
    
    # Scope 3
    BUSINESS_TRAVEL = "Business Travel"
    EMPLOYEE_COMMUTING = "Employee Commuting"
    WASTE_DISPOSAL = "Waste Disposal"
    WATER_SUPPLY = "Water Supply"
    UPSTREAM_TRANSPORT = "Upstream Transport"

class FuelType(str, Enum):
    """Fuel types for emission calculations"""
    # Fossil Fuels
    NATURAL_GAS = "Natural Gas"
    DIESEL = "Diesel"
    GASOLINE = "Gasoline"
    PROPANE = "Propane"
    HEATING_OIL = "Heating Oil"
    COAL = "Coal"
    LPG = "LPG"
    
    # Electricity
    ELECTRICITY_GRID = "Electricity (Grid)"
    ELECTRICITY_RENEWABLE = "Electricity (Renewable)"
    
    # Renewable/Alternative
    BIOFUEL = "Biofuel"
    BIODIESEL = "Biodiesel"
    BIOGAS = "Biogas"
    HYDROGEN = "Hydrogen"
    SOLAR = "Solar"
    WIND = "Wind"
    HYDRO = "Hydro"
    GEOTHERMAL = "Geothermal"

# =============================================================================
# WORKFLOW AND STATUS ENUMERATIONS
# =============================================================================

class ApprovalStatus(str, Enum):
    """Approval status for plans and milestones"""
    DRAFT = "Draft"                    # Initial creation
    PENDING = "Pending"                # Awaiting approval
    UNDER_REVIEW = "Under Review"      # Being reviewed
    APPROVED = "Approved"              # Approved and active
    REJECTED = "Rejected"              # Rejected
    REVISED = "Revised"                # Revised after feedback
    CANCELLED = "Cancelled"            # Cancelled
    EXPIRED = "Expired"                # Past validity period

class ImplementationStatus(str, Enum):
    """Implementation status for reduction measures"""
    PLANNED = "Planned"              # Scheduled for future
    IN_PROGRESS = "In Progress"      # Currently being implemented
    COMPLETED = "Completed"          # Successfully completed
    DELAYED = "Delayed"              # Behind schedule
    ON_HOLD = "On Hold"             # Temporarily paused
    CANCELLED = "Cancelled"          # Cancelled
    FAILED = "Failed"                # Implementation failed

class OnTrackStatus(str, Enum):
    """Progress tracking status"""
    AHEAD = "Ahead"          # Ahead of schedule/target
    ON_TRACK = "On Track"    # Meeting targets
    AT_RISK = "At Risk"      # Slightly behind, at risk
    OFF_TRACK = "Off Track"  # Significantly behind target
    CRITICAL = "Critical"    # Critical deviation

# =============================================================================
# SCENARIO AND STRATEGY ENUMERATIONS
# =============================================================================

class ScenarioType(str, Enum):
    """Milestone scenario types"""
    STANDARD = "Standard"        # Balanced approach
    AGGRESSIVE = "Aggressive"    # Accelerated reduction
    CONSERVATIVE = "Conservative" # Cautious approach
    CUSTOM = "Custom"            # User-defined scenario
    REGULATORY = "Regulatory"    # Compliance-driven
    COST_OPTIMIZED = "Cost-Optimized"  # Cost-focused

class AllocationMethod(str, Enum):
    """Target allocation methods for property portfolios"""
    CARBON_INTENSITY_WEIGHTED = "Carbon Intensity Weighted"
    PROPORTIONAL = "Proportional"
    RETROFIT_POTENTIAL = "Retrofit Potential"
    EQUAL_DISTRIBUTION = "Equal Distribution"
    AI_OPTIMIZED = "AI-Optimized"
    COST_BENEFIT = "Cost-Benefit Optimized"
    RISK_ADJUSTED = "Risk-Adjusted"

class StrategyType(str, Enum):
    """Reduction strategy types"""
    ENERGY_EFFICIENCY = "Energy Efficiency"
    RENEWABLE_ENERGY = "Renewable Energy"
    FUEL_SWITCHING = "Fuel Switching"
    PROCESS_OPTIMIZATION = "Process Optimization"
    BEHAVIORAL_CHANGE = "Behavioral Change"
    CARBON_OFFSET = "Carbon Offset"
    TECHNOLOGY_UPGRADE = "Technology Upgrade"
    BUILDING_RETROFIT = "Building Retrofit"
    FLEET_ELECTRIFICATION = "Fleet Electrification"
    WASTE_REDUCTION = "Waste Reduction"

# =============================================================================
# DATA TYPE ENUMERATIONS
# =============================================================================

class DataType(str, Enum):
    """Data synchronization types"""
    MILESTONES = "Milestones"
    BASELINES = "Baselines"
    COSTS = "Costs"
    ACTUAL_EMISSIONS = "Actual Emissions"
    ACTUAL_COSTS = "Actual Costs"
    PERFORMANCE_METRICS = "Performance Metrics"
    REDUCTION_OPTIONS = "Reduction Options"
    PROPERTY_DATA = "Property Data"

class DataQuality(str, Enum):
    """Data quality classification"""
    HIGH = "High"          # Verified, measured data
    MEDIUM = "Medium"      # Estimated from reliable sources
    LOW = "Low"           # Rough estimates
    UNKNOWN = "Unknown"    # Quality not assessed

class MeasurementMethod(str, Enum):
    """Method used for data measurement"""
    DIRECT_MEASUREMENT = "Direct Measurement"
    CALCULATED = "Calculated"
    ESTIMATED = "Estimated"
    MODELED = "Modeled"
    SUPPLIER_DATA = "Supplier Data"
    INDUSTRY_AVERAGE = "Industry Average"

# =============================================================================
# COST AND FINANCIAL ENUMERATIONS
# =============================================================================

class CostCategory(str, Enum):
    """Cost categories for financial analysis"""
    CAPEX = "Capital Expenditure"
    OPEX = "Operating Expenditure"
    MAINTENANCE = "Maintenance"
    ENERGY = "Energy"
    LABOR = "Labor"
    MATERIALS = "Materials"
    CONSULTING = "Consulting"
    MONITORING = "Monitoring"
    OTHER = "Other"

class Currency(str, Enum):
    """Supported currencies"""
    AUD = "AUD"  # Australian Dollar
    USD = "USD"  # US Dollar
    EUR = "EUR"  # Euro
    GBP = "GBP"  # British Pound
    JPY = "JPY"  # Japanese Yen
    CNY = "CNY"  # Chinese Yuan
    SGD = "SGD"  # Singapore Dollar

# =============================================================================
# REPORTING ENUMERATIONS
# =============================================================================

class ReportingPeriod(str, Enum):
    """Reporting period types"""
    MONTHLY = "Monthly"
    QUARTERLY = "Quarterly"
    SEMI_ANNUAL = "Semi-Annual"
    ANNUAL = "Annual"
    CUSTOM = "Custom"

class ReportFormat(str, Enum):
    """Report output formats"""
    PDF = "PDF"
    EXCEL = "Excel"
    CSV = "CSV"
    JSON = "JSON"
    HTML = "HTML"

# =============================================================================
# UNIT ENUMERATIONS
# =============================================================================

class EmissionUnit(str, Enum):
    """Emission measurement units"""
    KG_CO2E = "kg-CO2e"
    TONNES_CO2E = "tonnes-CO2e"
    MT_CO2E = "MT-CO2e"  # Metric tons

class EnergyUnit(str, Enum):
    """Energy measurement units"""
    KWH = "kWh"
    MWH = "MWh"
    GJ = "GJ"
    MMBTU = "MMBtu"
    THERMS = "Therms"

class AreaUnit(str, Enum):
    """Area measurement units"""
    SQM = "sqm"
    SQFT = "sqft"
    HECTARE = "hectare"
    ACRE = "acre"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_enum_values(enum_class) -> list:
    """Get list of all values from an enum class"""
    return [item.value for item in enum_class]

def get_enum_choices(enum_class) -> list:
    """Get list of (value, name) tuples for form choices"""
    return [(item.value, item.name) for item in enum_class]

def validate_enum_value(enum_class, value: str) -> bool:
    """Check if value is valid for given enum class"""
    try:
        enum_class(value)
        return True
    except ValueError:
        return False

# =============================================================================
# ENUM MAPPINGS
# =============================================================================

# Map priority levels to numeric values
PRIORITY_NUMERIC_MAP = {
    PriorityLevel.VERY_HIGH: 5,
    PriorityLevel.HIGH: 4,
    PriorityLevel.MEDIUM: 3,
    PriorityLevel.LOW: 2,
    PriorityLevel.VERY_LOW: 1
}

# Map risk levels to numeric scores
RISK_SCORE_MAP = {
    RiskLevel.CRITICAL: 5,
    RiskLevel.HIGH: 4,
    RiskLevel.MEDIUM: 3,
    RiskLevel.LOW: 2,
    RiskLevel.MINIMAL: 1
}

# Map retrofit potential to improvement percentage ranges
RETROFIT_POTENTIAL_MAP = {
    RetrofitPotential.CRITICAL: (50, 100),  # 50-100% improvement potential
    RetrofitPotential.HIGH: (30, 50),       # 30-50% improvement potential
    RetrofitPotential.MEDIUM: (15, 30),     # 15-30% improvement potential
    RetrofitPotential.LOW: (5, 15),         # 5-15% improvement potential
    RetrofitPotential.MINIMAL: (0, 5)       # 0-5% improvement potential
}

# Export all enums
__all__ = [
    # Property & Building
    'BuildingType',
    'RetrofitPotential',
    'PropertyStatus',
    
    # Priority & Risk
    'PriorityLevel',
    'RiskLevel',
    'UrgencyLevel',
    
    # Emissions
    'EmissionScope',
    'EmissionCategory',
    'FuelType',
    
    # Workflow & Status
    'ApprovalStatus',
    'ImplementationStatus',
    'OnTrackStatus',
    
    # Scenario & Strategy
    'ScenarioType',
    'AllocationMethod',
    'StrategyType',
    
    # Data Types
    'DataType',
    'DataQuality',
    'MeasurementMethod',
    
    # Cost & Financial
    'CostCategory',
    'Currency',
    
    # Reporting
    'ReportingPeriod',
    'ReportFormat',
    
    # Units
    'EmissionUnit',
    'EnergyUnit',
    'AreaUnit',
    
    # Helper Functions
    'get_enum_values',
    'get_enum_choices',
    'validate_enum_value',
    
    # Mappings
    'PRIORITY_NUMERIC_MAP',
    'RISK_SCORE_MAP',
    'RETROFIT_POTENTIAL_MAP'
]
