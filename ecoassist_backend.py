import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import uuid
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenAI API setup - Replace with your API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except:
    client = None

@dataclass
class Property:
    """Property data structure"""
    property_id: str
    area_sqm: float
    occupancy: float
    baseline_emission: float
    scope1_emission: float
    scope2_emission: float
    scope3_emission: float
    retrofit_potential: str
    building_type: str
    year_built: int

@dataclass
class ReductionOption:
    """CO2 reduction option data structure"""
    option_id: str
    name: str
    co2_reduction: float
    capex: float
    opex: float
    priority: str
    implementation_time: int
    risk_level: str

@dataclass
class MilestoneScenario:
    """Milestone scenario data structure"""
    scenario_id: str
    name: str
    description: str
    yearly_targets: Dict[int, float]
    total_capex: float
    total_opex: float
    strategy_type: str
    reduction_rate_2030: float
    reduction_rate_2050: float

@dataclass
class StrategicPattern:
    """Strategic pattern for emission reduction"""
    pattern_id: str
    name: str
    description: str
    reduction_options: Dict[str, int]  # option_name: priority (1-5)
    implementation_approach: str
    estimated_cost: float
    estimated_reduction: float
    risk_level: str

class EcoAssistBackend:
    """Backend logic for EcoAssist AI system"""
    
    def __init__(self):
        self.properties = []
        self.reduction_options = []
        self.strategic_patterns = []
        self.benchmark_data = self._load_benchmark_data()
        self.current_plans = {}
        self.milestone_scenarios = []
        self._initialize_sample_data()
        self._initialize_strategic_patterns()
        
    def _load_benchmark_data(self) -> Dict:
        """Load benchmark data for milestone recommendations"""
        return {
            "office": {"intensity_baseline": 85, "reduction_potential": 0.65},
            "retail": {"intensity_baseline": 120, "reduction_potential": 0.55},
            "residential": {"intensity_baseline": 45, "reduction_potential": 0.70},
            "industrial": {"intensity_baseline": 150, "reduction_potential": 0.60},
            "mixed-use": {"intensity_baseline": 95, "reduction_potential": 0.60},
            "educational": {"intensity_baseline": 70, "reduction_potential": 0.65},
            "warehouse": {"intensity_baseline": 40, "reduction_potential": 0.50}
        }
    
    def _initialize_strategic_patterns(self):
        """Initialize preset strategic patterns based on technical requirements"""
        patterns = [
            StrategicPattern(
                pattern_id="re_active",
                name="Active Installation of RE",
                description="Prioritizes renewable energy installation with medium implementation timeline",
                reduction_options={
                    "Solar PV Installation": 5,
                    "Wind Energy Installation": 4,
                    "Energy Storage Systems": 4,
                    "Smart Grid Integration": 3,
                    "Renewable Certificates": 3,
                    "LED Lighting Upgrade": 2,
                    "Carbon Credits": 1
                },
                implementation_approach="Medium",
                estimated_cost=250000,
                estimated_reduction=4200,
                risk_level="Low"
            ),
            StrategicPattern(
                pattern_id="green_fuels",
                name="Replace to Green Fuels",
                description="Focus on fuel replacement and electrification strategies",
                reduction_options={
                    "Biofuel Replacement": 5,
                    "Hydrogen Systems": 4,
                    "Electric Vehicle Fleet": 4,
                    "Heat Pump Systems": 4,
                    "Green Gas Supply": 3,
                    "Energy Efficiency": 3,
                    "Carbon Credits": 2
                },
                implementation_approach="Medium",
                estimated_cost=220000,
                estimated_reduction=3800,
                risk_level="Medium"
            ),
            StrategicPattern(
                pattern_id="carbon_credits",
                name="Carbon Credit Priority",
                description="Quick emission reduction through offset mechanisms",
                reduction_options={
                    "Carbon Credits": 5,
                    "Offset Programs": 5,
                    "Renewable Certificates": 4,
                    "Monitoring Systems": 3,
                    "Energy Efficiency": 2,
                    "Solar PV Installation": 2,
                    "LED Lighting Upgrade": 1
                },
                implementation_approach="Fast",
                estimated_cost=150000,
                estimated_reduction=3000,
                risk_level="High"
            ),
            StrategicPattern(
                pattern_id="low_energy",
                name="Active Installation of Low Energy Equipment",
                description="Comprehensive energy efficiency and equipment upgrade approach",
                reduction_options={
                    "LED Lighting Upgrade": 5,
                    "HVAC System Upgrade": 5,
                    "Smart Controls": 4,
                    "Building Insulation": 4,
                    "Energy Management Systems": 4,
                    "Smart Building Technology": 3,
                    "Heat Recovery Systems": 3
                },
                implementation_approach="Medium",
                estimated_cost=180000,
                estimated_reduction=3200,
                risk_level="Low"
            ),
            StrategicPattern(
                pattern_id="balanced",
                name="Balanced Strategy",
                description="Equally weighted approach across all emission reduction strategies",
                reduction_options={
                    "Solar PV Installation": 3,
                    "LED Lighting Upgrade": 3,
                    "HVAC System Upgrade": 3,
                    "Carbon Credits": 3,
                    "Energy Efficiency": 3,
                    "Heat Pump Systems": 3,
                    "Smart Controls": 3
                },
                implementation_approach="Balanced",
                estimated_cost=200000,
                estimated_reduction=3500,
                risk_level="Medium"
            )
        ]
        self.strategic_patterns = patterns
    
    def _initialize_sample_data(self):
        """Initialize sample data for the system"""
        # Enhanced sample properties with diverse building types and characteristics
        sample_properties = [
            Property("BP01", 1500, 0.92, 150000, 95000, 55000, 25000, "High", "Office", 2010),
            Property("CB01", 800, 0.85, 85000, 55000, 30000, 15000, "Medium", "Retail", 2015),
            Property("CB02", 2000, 0.95, 220000, 140000, 80000, 40000, "High", "Office", 2008),
            Property("CB03", 600, 0.75, 65000, 40000, 25000, 12000, "Low", "Residential", 2018),
            Property("CB04", 3000, 0.88, 380000, 240000, 140000, 70000, "Medium", "Industrial", 2012),
            Property("HA1", 1200, 0.80, 125000, 80000, 45000, 20000, "High", "Mixed-Use", 2014),
            Property("MP", 1800, 0.90, 195000, 125000, 70000, 35000, "Medium", "Office", 2016),
            Property("TC01", 900, 0.82, 95000, 60000, 35000, 18000, "Medium", "Educational", 2020),
            Property("WH1", 2500, 0.70, 180000, 120000, 60000, 30000, "Low", "Warehouse", 2013)
        ]
        
        # Enhanced reduction options with more comprehensive coverage
        sample_reduction_options = [
            ReductionOption("SOLAR001", "Solar PV Installation", 4500, 200000, 12000, "High", 4, "Low"),
            ReductionOption("LED001", "LED Lighting Upgrade", 1200, 45000, 3000, "High", 1, "Low"),
            ReductionOption("HVAC001", "HVAC System Upgrade", 3500, 150000, 18000, "High", 3, "Medium"),
            ReductionOption("INSUL001", "Building Insulation", 2000, 80000, 2500, "Medium", 2, "Low"),
            ReductionOption("SMART001", "Smart Building Systems", 2200, 95000, 22000, "Medium", 2, "Medium"),
            ReductionOption("CREDIT001", "Carbon Credits", 800, 0, 35000, "Low", 0, "High"),
            ReductionOption("BIOFUEL001", "Biofuel Replacement", 3000, 70000, 28000, "Medium", 2, "Medium"),
            ReductionOption("WIND001", "Wind Energy Installation", 3800, 180000, 15000, "Medium", 5, "Medium"),
            ReductionOption("HEAT001", "Heat Pump Systems", 2800, 120000, 14000, "Medium", 3, "Low"),
            ReductionOption("STORAGE001", "Energy Storage Systems", 1800, 140000, 8000, "Low", 3, "Medium"),
            ReductionOption("EV001", "Electric Vehicle Fleet", 1500, 85000, 12000, "Medium", 2, "Low"),
            ReductionOption("HYDRO001", "Hydrogen Systems", 2500, 160000, 20000, "Low", 4, "High"),
            ReductionOption("MONITOR001", "Energy Monitoring Systems", 800, 35000, 8000, "High", 1, "Low")
        ]
        
        self.properties = sample_properties
        self.reduction_options = sample_reduction_options
    
    def generate_milestone_scenarios(self, target_year: int, base_reduction_2030: float, 
                                   base_reduction_2050: float) -> List[MilestoneScenario]:
        """Generate multiple milestone scenarios with detailed scope breakdown"""
        scenarios = []
        current_year = datetime.now().year
        
        # Calculate total baseline emissions with scope breakdown
        total_baseline = sum(prop.baseline_emission for prop in self.properties) if self.properties else 1000000
        total_scope1 = sum(prop.scope1_emission for prop in self.properties) if self.properties else 600000
        total_scope2 = sum(prop.scope2_emission for prop in self.properties) if self.properties else 400000
        total_scope3 = sum(prop.scope3_emission for prop in self.properties) if self.properties else 300000
        
        # Calculate baseline costs (estimated)
        baseline_cost = total_baseline * 0.2  # Assume $0.2 per tCO2e baseline cost
        
        # Scenario configurations based on technical requirements
        scenario_configs = [
            {
                "name": "Slow Decarbonisation",
                "description": "Conservative linear reduction approach with steady progress",
                "reduction_2030": base_reduction_2030,
                "reduction_2050": base_reduction_2050,
                "growth_rate": 0.0,
                "front_loading": False,
                "strategy_focus": "cost-effective"
            },
            {
                "name": "Aggressive Decarbonisation", 
                "description": "Front-loaded reduction with aggressive early action and higher investment",
                "reduction_2030": min(base_reduction_2030 + 15, 65),
                "reduction_2050": min(base_reduction_2050 + 10, 95),
                "growth_rate": 0.12,
                "front_loading": True,
                "strategy_focus": "high-impact"
            },
            {
                "name": "SBTi Aligned",
                "description": "Science-based targets initiative aligned pathway following 1.5°C trajectory",
                "reduction_2030": 50.0,
                "reduction_2050": 90.0,
                "growth_rate": 0.08,
                "front_loading": True,
                "strategy_focus": "science-based"
            },
            {
                "name": "Net Zero by 2050",
                "description": "Comprehensive pathway targeting net-zero emissions by 2050",
                "reduction_2030": 45.0,
                "reduction_2050": 100.0,
                "growth_rate": 0.10,
                "front_loading": False,
                "strategy_focus": "net-zero"
            }
        ]
        
        for config in scenario_configs:
            yearly_targets = {}
            capex_estimate = 0
            opex_estimate = 0
            
            years_to_target = target_year - current_year
            
            # Calculate detailed breakdown for each milestone year
            milestone_data = {
                "base_year_2025": {
                    "cost": baseline_cost,
                    "scope1": total_scope1,
                    "scope2": total_scope2,
                    "scope1_2": total_scope1 + total_scope2
                },
                "target_year_2030": {},
                "target_year_2050": {}
            }
            
            # Calculate 2030 targets
            reduction_2030 = config["reduction_2030"] / 100
            milestone_data["target_year_2030"] = {
                "cost": baseline_cost * (1 + 0.5 * reduction_2030),  # Cost increases with reduction efforts
                "scope1": total_scope1 * (1 - reduction_2030 * 0.8),  # Scope 1 more controllable
                "scope2": total_scope2 * (1 - reduction_2030 * 0.9),  # Scope 2 easier to reduce
                "scope1_2": (total_scope1 + total_scope2) * (1 - reduction_2030 * 0.85),
                "reduction_rate": config["reduction_2030"]
            }
            
            # Calculate 2050 targets
            reduction_2050 = config["reduction_2050"] / 100
            milestone_data["target_year_2050"] = {
                "cost": baseline_cost * (1 + 0.3 * reduction_2050),  # Long-term cost efficiency
                "scope1": total_scope1 * (1 - reduction_2050 * 0.9),
                "scope2": total_scope2 * (1 - reduction_2050 * 0.95),
                "scope1_2": (total_scope1 + total_scope2) * (1 - reduction_2050 * 0.92),
                "reduction_rate": config["reduction_2050"]
            }
            
            # Generate yearly progression for visualization
            for i in range(years_to_target + 1):
                year = current_year + i
                
                if year <= 2030:
                    years_to_2030 = 2030 - current_year
                    if years_to_2030 > 0:
                        progress = i / years_to_2030
                        if config["front_loading"]:
                            progress = 1 - (1 - progress) ** 1.5
                        reduction_percent = progress * (config["reduction_2030"] / 100)
                    else:
                        reduction_percent = config["reduction_2030"] / 100
                else:
                    years_2030_to_target = target_year - 2030
                    years_from_2030 = year - 2030
                    if years_2030_to_target > 0:
                        progress_2030 = config["reduction_2030"] / 100
                        remaining_reduction = (config["reduction_2050"] - config["reduction_2030"]) / 100
                        additional_progress = (years_from_2030 / years_2030_to_target) * remaining_reduction
                        reduction_percent = progress_2030 + additional_progress
                    else:
                        reduction_percent = config["reduction_2050"] / 100
                
                target_emission = total_baseline * (1 - min(reduction_percent, config["reduction_2050"]/100))
                yearly_targets[year] = max(target_emission, 0)
                
                # Estimate costs with enhanced calculation
                annual_capex = 60000 * (1 + config["growth_rate"]) ** i
                annual_opex = 18000 * (1 + config["growth_rate"]) ** i
                capex_estimate += annual_capex
                opex_estimate += annual_opex
            
            scenario = MilestoneScenario(
                scenario_id=str(uuid.uuid4()),
                name=config["name"],
                description=config["description"],
                yearly_targets=yearly_targets,
                total_capex=capex_estimate,
                total_opex=opex_estimate,
                strategy_type=config["strategy_focus"],
                reduction_rate_2030=config["reduction_2030"],
                reduction_rate_2050=config["reduction_2050"]
            )
            
            # Add detailed milestone data for table display
            scenario.milestone_data = milestone_data
            scenarios.append(scenario)
        
        self.milestone_scenarios = scenarios
        return scenarios
    
    def get_milestone_table_data(self) -> List[Dict]:
        """Get formatted milestone data for detailed table display"""
        table_data = []
        
        for scenario in self.milestone_scenarios:
            if hasattr(scenario, 'milestone_data'):
                milestone_data = scenario.milestone_data
                
                # Create table rows for each scenario
                scenario_data = {
                    "scenario_name": scenario.name,
                    "scenario_id": scenario.scenario_id,
                    "rows": [
                        {
                            "item": "Cost",
                            "base_year_2025": f"{milestone_data['base_year_2025']['cost']:,.0f}",
                            "base_unit_2025": "kAUD",
                            "target_year_2030": f"{milestone_data['target_year_2030']['cost']:,.0f}",
                            "target_unit_2030": "kAUD",
                            "reduction_rate_2030": f"{milestone_data['target_year_2030']['reduction_rate']:.1f}",
                            "reduction_unit_2030": "%",
                            "target_year_2050": f"{milestone_data['target_year_2050']['cost']:,.0f}",
                            "target_unit_2050": "kAUD",
                            "reduction_rate_2050": f"{milestone_data['target_year_2050']['reduction_rate']:.1f}",
                            "reduction_unit_2050": "%"
                        },
                        {
                            "item": "Scope 1",
                            "base_year_2025": f"{milestone_data['base_year_2025']['scope1']:,.0f}",
                            "base_unit_2025": "tCO2",
                            "target_year_2030": f"{milestone_data['target_year_2030']['scope1']:,.0f}",
                            "target_unit_2030": "tCO2",
                            "reduction_rate_2030": f"{milestone_data['target_year_2030']['reduction_rate']:.1f}",
                            "reduction_unit_2030": "%",
                            "target_year_2050": f"{milestone_data['target_year_2050']['scope1']:,.0f}",
                            "target_unit_2050": "tCO2",
                            "reduction_rate_2050": f"{milestone_data['target_year_2050']['reduction_rate']:.1f}",
                            "reduction_unit_2050": "%"
                        },
                        {
                            "item": "Scope 2",
                            "base_year_2025": f"{milestone_data['base_year_2025']['scope2']:,.0f}",
                            "base_unit_2025": "tCO2",
                            "target_year_2030": f"{milestone_data['target_year_2030']['scope2']:,.0f}",
                            "target_unit_2030": "tCO2",
                            "reduction_rate_2030": f"{milestone_data['target_year_2030']['reduction_rate']:.1f}",
                            "reduction_unit_2030": "%",
                            "target_year_2050": f"{milestone_data['target_year_2050']['scope2']:,.0f}",
                            "target_unit_2050": "tCO2",
                            "reduction_rate_2050": f"{milestone_data['target_year_2050']['reduction_rate']:.1f}",
                            "reduction_unit_2050": "%"
                        },
                        {
                            "item": "Scope 1 + 2",
                            "base_year_2025": f"{milestone_data['base_year_2025']['scope1_2']:,.0f}",
                            "base_unit_2025": "tCO2",
                            "target_year_2030": f"{milestone_data['target_year_2030']['scope1_2']:,.0f}",
                            "target_unit_2030": "tCO2",
                            "reduction_rate_2030": f"{milestone_data['target_year_2030']['reduction_rate']:.1f}",
                            "reduction_unit_2030": "%",
                            "target_year_2050": f"{milestone_data['target_year_2050']['scope1_2']:,.0f}",
                            "target_unit_2050": "tCO2",
                            "reduction_rate_2050": f"{milestone_data['target_year_2050']['reduction_rate']:.1f}",
                            "reduction_unit_2050": "%"
                        }
                    ]
                }
                table_data.append(scenario_data)
        
        return table_data
    
    def create_custom_scenario(self, name: str, reduction_2030: float, 
                             reduction_2050: float, target_year: int) -> MilestoneScenario:
        """Create a custom milestone scenario"""
        current_year = datetime.now().year
        total_baseline = sum(prop.baseline_emission for prop in self.properties) if self.properties else 1000000
        
        yearly_targets = {}
        years_to_target = target_year - current_year
        
        for i in range(years_to_target + 1):
            year = current_year + i
            
            if year <= 2030:
                years_to_2030 = 2030 - current_year
                if years_to_2030 > 0:
                    progress = i / years_to_2030
                    reduction_percent = progress * (reduction_2030 / 100)
                else:
                    reduction_percent = reduction_2030 / 100
            else:
                years_2030_to_target = target_year - 2030
                years_from_2030 = year - 2030
                if years_2030_to_target > 0:
                    progress_2030 = reduction_2030 / 100
                    remaining_reduction = (reduction_2050 - reduction_2030) / 100
                    additional_progress = (years_from_2030 / years_2030_to_target) * remaining_reduction
                    reduction_percent = progress_2030 + additional_progress
                else:
                    reduction_percent = reduction_2050 / 100
            
            target_emission = total_baseline * (1 - min(reduction_percent, reduction_2050/100))
            yearly_targets[year] = max(target_emission, 0)
        
        # Estimate costs for custom scenario
        capex_estimate = len(yearly_targets) * 65000
        opex_estimate = len(yearly_targets) * 20000
        
        custom_scenario = MilestoneScenario(
            scenario_id=str(uuid.uuid4()),
            name=name,
            description="Custom user-defined scenario",
            yearly_targets=yearly_targets,
            total_capex=capex_estimate,
            total_opex=opex_estimate,
            strategy_type="Custom",
            reduction_rate_2030=reduction_2030,
            reduction_rate_2050=reduction_2050
        )
        
        return custom_scenario
    
    def calculate_property_breakdown(self, selected_scenario_name: str = "Aggressive Decarbonisation") -> Tuple[pd.DataFrame, str]:
        """Generate property-level breakdown for yearly targets"""
        try:
            # Find the selected scenario
            selected_scenario = None
            for scenario in self.milestone_scenarios:
                if scenario.name == selected_scenario_name:
                    selected_scenario = scenario
                    break
            
            if not selected_scenario:
                return None, "No scenario selected"
            
            # Calculate totals for analysis
            total_baseline = sum(prop.baseline_emission for prop in self.properties)
            total_nla = sum(prop.area_sqm * prop.occupancy for prop in self.properties)
            
            # Generate comprehensive breakdown data
            years = [2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032]
            breakdown_data = []
            
            for prop in self.properties:
                for year in years:
                    if year in selected_scenario.yearly_targets:
                        # Calculate proportional allocation based on carbon intensity and retrofit potential
                        prop_baseline = prop.baseline_emission
                        prop_nla = prop.area_sqm * prop.occupancy
                        carbon_intensity = prop_baseline / prop_nla if prop_nla > 0 else 0
                        
                        # Apply retrofit potential weighting
                        retrofit_multiplier = 1.3 if prop.retrofit_potential == "High" else 1.0 if prop.retrofit_potential == "Medium" else 0.7
                        
                        # Calculate weighted share for fair allocation
                        weighted_share = (carbon_intensity * retrofit_multiplier) / sum(
                            (p.baseline_emission / (p.area_sqm * p.occupancy) if (p.area_sqm * p.occupancy) > 0 else 0) * 
                            (1.3 if p.retrofit_potential == "High" else 1.0 if p.retrofit_potential == "Medium" else 0.7)
                            for p in self.properties
                        )
                        
                        total_target = selected_scenario.yearly_targets[year]
                        prop_target = total_target * weighted_share
                        
                        # Calculate metrics
                        reduction_rate = ((prop_baseline - prop_target) / prop_baseline) * 100 if prop_baseline > 0 else 0
                        target_carbon_intensity = prop_target / prop_nla if prop_nla > 0 else 0
                        
                        breakdown_data.append([
                            prop.property_id,
                            f"{prop.area_sqm:,.0f}",
                            f"{prop_baseline:,.0f}",
                            year,
                            f"{prop_target:,.0f}",
                            f"{reduction_rate:.1f}%",
                            f"{target_carbon_intensity:.2f}",
                            prop.building_type,
                            prop.retrofit_potential
                        ])
            
            breakdown_df = pd.DataFrame(breakdown_data, 
                                      columns=["Property", "NLA (m²)", "Baseline (tCO₂e)", 
                                             "Year", "Target (tCO₂e)", "Reduction Rate", 
                                             "Carbon Intensity (tCO₂e/m²)", "Building Type", "Retrofit Potential"])
            
            return breakdown_df, f"Property breakdown for {selected_scenario_name} generated successfully!"
            
        except Exception as e:
            logger.error(f"Error in property breakdown: {str(e)}")
            return None, f"Error: {str(e)}"
    
    def analyze_strategic_pattern(self, pattern_name: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict], str]:
        """Analyze strategic pattern and return detailed information"""
        try:
            selected_pattern = None
            for pattern in self.strategic_patterns:
                if pattern.name == pattern_name:
                    selected_pattern = pattern
                    break
            
            if not selected_pattern:
                return None, None, "Pattern not found"
            
            # Create enhanced pattern details display with impact analysis
            details_data = []
            for option_name, priority in selected_pattern.reduction_options.items():
                # Find matching reduction option for detailed info
                matching_option = None
                for option in self.reduction_options:
                    if option_name.lower() in option.name.lower():
                        matching_option = option
                        break
                
                # Create priority stars display
                priority_stars = "★" * priority + "☆" * (5 - priority)
                
                # Calculate estimated impact and cost efficiency
                if matching_option:
                    estimated_impact = "High" if matching_option.co2_reduction > 2000 else "Medium" if matching_option.co2_reduction > 1000 else "Low"
                    cost_per_tonne = (matching_option.capex + matching_option.opex * 5) / matching_option.co2_reduction if matching_option.co2_reduction > 0 else 0
                    cost_efficiency = "High" if cost_per_tonne < 100 else "Medium" if cost_per_tonne < 200 else "Low"
                    risk_reputation = matching_option.risk_level
                else:
                    estimated_impact = "Medium"
                    cost_efficiency = "Medium"
                    risk_reputation = "Medium"
                
                details_data.append([
                    option_name,
                    priority_stars,
                    estimated_impact,
                    cost_efficiency,
                    risk_reputation
                ])
            
            details_df = pd.DataFrame(details_data,
                                    columns=["Reduction Option", "Priority", "Estimated Impact", 
                                           "Cost Efficiency", "Risk Level"])
            
            # Calculate comprehensive impact summary
            total_potential_reduction = sum([
                next((opt.co2_reduction for opt in self.reduction_options 
                     if option_name.lower() in opt.name.lower()), 500) * priority
                for option_name, priority in selected_pattern.reduction_options.items()
            ]) / len(selected_pattern.reduction_options)
            
            # Determine target gap analysis
            target_gap_2030 = "On Track" if total_potential_reduction > 3000 else "Needs Attention"
            target_gap_2050 = "On Track" if total_potential_reduction > 2500 else "Needs Attention"
            
            summary_info = {
                "strategy_name": selected_pattern.name,
                "description": selected_pattern.description,
                "estimated_cost": f"${selected_pattern.estimated_cost:,.0f}",
                "estimated_reduction": f"{int(total_potential_reduction):,} tCO2e",
                "risk_level": selected_pattern.risk_level,
                "implementation": selected_pattern.implementation_approach,
                "target_gap_2030": target_gap_2030,
                "target_gap_2050": target_gap_2050
            }
            
            return details_df, summary_info, "Strategic pattern analysis completed successfully!"
            
        except Exception as e:
            logger.error(f"Error in strategic pattern selection: {str(e)}")
            return None, None, f"Error: {str(e)}"
    
    def generate_budget_data(self, property_id: str = "BP01") -> Tuple[Optional[Dict], Optional[pd.DataFrame], Optional[Dict], str]:
        """Generate budget analysis data for a property"""
        try:
            # Find the property for context
            target_property = None
            for prop in self.properties:
                if prop.property_id == property_id:
                    target_property = prop
                    break
            
            if not target_property:
                return None, None, None, f"Property {property_id} not found"
            
            # Generate realistic budget data based on property characteristics
            years = list(range(2019, 2036))
            capex_data = []
            opex_data = []
            
            # Base budget calculations based on property size and type
            base_capex = max(30000, target_property.area_sqm * 50)  # $50 per sqm base
            base_opex = max(15000, target_property.area_sqm * 25)   # $25 per sqm base
            
            # Retrofit potential affects investment levels
            retrofit_multiplier = 1.5 if target_property.retrofit_potential == "High" else 1.2 if target_property.retrofit_potential == "Medium" else 1.0
            
            for i, year in enumerate(years):
                # Generate budget progression with investment cycles
                annual_growth = 0.03 + (0.02 if year >= 2025 else 0)  # Higher growth post-2025
                
                year_capex = base_capex * (1 + annual_growth) ** (year - 2019) * retrofit_multiplier
                year_opex = base_opex * (1 + annual_growth) ** (year - 2019)
                
                # Add investment spikes for major upgrades
                if year in [2024, 2027, 2030, 2033]:  # Major investment years
                    year_capex *= 2.5
                elif year in [2025, 2028, 2031]:  # Medium investment years
                    year_capex *= 1.8
                
                capex_data.append(year_capex)
                opex_data.append(year_opex)
            
            # Calculate emission reduction correlation
            total_budgets = [c + o for c, o in zip(capex_data, opex_data)]
            emission_reduction = [max(0, (budget - min(total_budgets)) / 1000) for budget in total_budgets]
            
            # Create budget data structure for plotting
            budget_plot_data = {
                'years': years,
                'capex_data': capex_data,
                'opex_data': opex_data,
                'total_budgets': total_budgets,
                'emission_reduction': emission_reduction,
                'cumulative_capex': np.cumsum(capex_data).tolist(),
                'cumulative_opex': np.cumsum(opex_data).tolist(),
                'cumulative_total': (np.cumsum(capex_data) + np.cumsum(opex_data)).tolist()
            }
            
            # Create budget table data
            budget_data = []
            for i, year in enumerate(years):
                total = capex_data[i] + opex_data[i]
                roi_estimate = (emission_reduction[i] * 50) / total if total > 0 else 0  # $50 per tonne value
                budget_data.append([
                    year, 
                    f"${capex_data[i]:,.0f}", 
                    f"${opex_data[i]:,.0f}", 
                    f"${total:,.0f}",
                    f"{emission_reduction[i]:.1f}",
                    f"{roi_estimate:.2f}"
                ])
            
            budget_df = pd.DataFrame(budget_data, 
                                   columns=["Year", "CAPEX (AUD $)", "OPEX (AUD $)", 
                                          "TOTAL (AUD $)", "Est. CO₂ Reduction (tCO₂e)", "ROI Ratio"])
            
            # Calculate comprehensive totals and insights
            total_capex = sum(capex_data)
            total_opex = sum(opex_data)
            total_budget = total_capex + total_opex
            total_reduction = sum(emission_reduction)
            avg_cost_per_tonne = total_budget / total_reduction if total_reduction > 0 else 0
            
            summary_info = {
                "property_id": property_id,
                "building_type": target_property.building_type,
                "total_planning_budget": f"${total_budget:,.0f}",
                "total_capex": f"${total_capex:,.0f}",
                "total_opex": f"${total_opex:,.0f}",
                "retrofit_potential": target_property.retrofit_potential,
                "total_co2_reduction": f"{total_reduction:,.1f} tCO₂e",
                "avg_cost_per_tonne": f"${avg_cost_per_tonne:,.0f}/tCO₂e",
                "investment_efficiency": "High" if avg_cost_per_tonne < 200 else "Medium" if avg_cost_per_tonne < 400 else "Low"
            }
            
            return budget_plot_data, budget_df, summary_info, "Comprehensive budget analysis completed successfully!"
            
        except Exception as e:
            logger.error(f"Error in budget analysis: {str(e)}")
            return None, None, None, f"Error: {str(e)}"
    
    def get_properties_data(self) -> List[List]:
        """Get formatted properties data for display"""
        property_data = []
        for prop in self.properties:
            property_data.append([
                prop.property_id,
                f"{prop.area_sqm:,.0f}",
                f"{prop.occupancy:.1%}",
                f"{prop.baseline_emission:,.0f}",
                f"{prop.scope1_emission:,.0f}",
                f"{prop.scope2_emission:,.0f}",
                f"{prop.scope3_emission:,.0f}",
                prop.retrofit_potential,
                prop.building_type,
                prop.year_built
            ])
        return property_data
    
    def get_reduction_options_data(self) -> List[List]:
        """Get formatted reduction options data for display"""
        option_data = []
        for option in self.reduction_options:
            option_data.append([
                option.option_id,
                option.name,
                f"{option.co2_reduction:.0f}",
                f"${option.capex:,.0f}",
                f"${option.opex:,.0f}",
                option.priority,
                f"{option.implementation_time} years",
                option.risk_level
            ])
        return option_data
    
    def get_strategic_patterns_data(self) -> List[List]:
        """Get formatted strategic patterns data for display"""
        pattern_data = []
        for pattern in self.strategic_patterns:
            pattern_data.append([
                pattern.name,
                pattern.description,
                pattern.implementation_approach,
                f"${pattern.estimated_cost:,.0f}",
                f"{pattern.estimated_reduction:.0f} tCO₂e",
                pattern.risk_level
            ])
        return pattern_data
    
    def create_custom_strategic_pattern(self, pattern_name: str, option_priorities: Dict[str, int]) -> str:
        """Create a custom strategic pattern"""
        try:
            custom_pattern = StrategicPattern(
                pattern_id=str(uuid.uuid4()),
                name=pattern_name,
                description="Custom user-defined strategy",
                reduction_options=option_priorities,
                implementation_approach="Custom",
                estimated_cost=175000,  # Placeholder calculation
                estimated_reduction=2600,  # Placeholder calculation
                risk_level="Medium"
            )
            
            self.strategic_patterns.append(custom_pattern)
            
            return f"Custom strategic pattern '{pattern_name}' created successfully!"
            
        except Exception as e:
            logger.error(f"Error creating custom pattern: {str(e)}")
            return f"Error: {str(e)}"
    
    def reoptimize_annual_plan(self, property_id: str, deviation_threshold: float = 0.05) -> Tuple[Optional[Dict], Optional[pd.DataFrame], str]:
        """
        Function 4: Re-optimization of Annual Plan
        Analyze actual vs target emissions and costs, then reoptimize if needed
        """
        try:
            # Find the property
            target_property = None
            for prop in self.properties:
                if prop.property_id == property_id:
                    target_property = prop
                    break
            
            if not target_property:
                return None, None, f"Property {property_id} not found"
            
            # Generate sample monthly data for the property
            months = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
            current_month = 3  # Assuming we're in October (index 3)
            
            # Sample actual vs target emission data
            base_emission = target_property.baseline_emission / 12  # Monthly baseline
            emission_actual = [
                base_emission * 0.8, base_emission * 0.75, base_emission * 1.2, base_emission * 1.1,
                base_emission * 1.15, base_emission * 1.05, base_emission * 0.95, base_emission * 0.85,
                base_emission * 0.9, base_emission * 0.85, base_emission * 0.8, base_emission * 0.75
            ]
            emission_target = [base_emission * 0.9] * 12  # Target is 10% reduction
            
            # Sample cost data (in thousands AUD)
            base_cost = 15  # Base monthly cost in thousands
            cost_actual = [40, 38, 11, 15, 17, 18, 16, 20, 22, 23, 15, 16]
            cost_target = [15] * 12
            
            # Calculate total variance for emission
            total_variance = [actual - target for actual, target in zip(emission_actual, emission_target)]
            
            # Calculate YTD analysis
            ytd_actual_emission = sum(emission_actual[:current_month + 1])
            ytd_target_emission = sum(emission_target[:current_month + 1])
            ytd_actual_cost = sum(cost_actual[:current_month + 1])
            ytd_target_cost = sum(cost_target[:current_month + 1])
            
            emission_deviation = (ytd_actual_emission - ytd_target_emission) / ytd_target_emission if ytd_target_emission > 0 else 0
            cost_deviation = (ytd_actual_cost - ytd_target_cost) / ytd_target_cost if ytd_target_cost > 0 else 0
            
            needs_reoptimization = abs(emission_deviation) > deviation_threshold or abs(cost_deviation) > deviation_threshold
            
            # Generate refined targets if reoptimization is needed
            refined_emission_target = emission_target.copy()
            refined_cost_target = cost_target.copy()
            
            if needs_reoptimization:
                # Adjust remaining months
                remaining_months = 12 - (current_month + 1)
                if remaining_months > 0:
                    # For emissions: adjust to meet annual target
                    annual_emission_target = sum(emission_target)
                    remaining_emission_needed = annual_emission_target - ytd_actual_emission
                    adjusted_monthly_emission = remaining_emission_needed / remaining_months
                    
                    for i in range(current_month + 1, 12):
                        refined_emission_target[i] = max(adjusted_monthly_emission, emission_target[i] * 0.5)
                    
                    # For costs: adjust to prevent overrun
                    annual_cost_target = sum(cost_target)
                    remaining_cost_budget = annual_cost_target - ytd_actual_cost
                    adjusted_monthly_cost = remaining_cost_budget / remaining_months
                    
                    for i in range(current_month + 1, 12):
                        refined_cost_target[i] = max(adjusted_monthly_cost, cost_target[i] * 0.8)
            
            # Create plot data structure
            plot_data = {
                'months': months,
                'emission_actual': emission_actual,
                'emission_target': emission_target,
                'emission_refined_target': refined_emission_target,
                'cost_actual': cost_actual,
                'cost_target': cost_target,
                'cost_refined_target': refined_cost_target,
                'total_variance': total_variance,
                'current_month_index': current_month,
                'needs_reoptimization': needs_reoptimization
            }
            
            # Create consumptions table data
            consumptions_data = []
            consumption_types = [
                "Natural gas_BB_Cons_GJ (GJ)",
                "Other gaseous fossil fuels_BB_Cons_m3 (m3)",
                "Gasoline for use as fuel in an aircraft_BB_Cons_kL (kL)",
                "Diesel oil_BB_Cons_kL (kL)",
                "Renewable Fuel Combustion_Cons",
                "Renewable diesel_BB_Cons_GJ (GJ)"
            ]
            
            for cons_type in consumption_types:
                row = [cons_type]
                for month in months:
                    # Generate sample consumption values
                    base_value = np.random.randint(50000, 200000)
                    row.append(base_value)
                consumptions_data.append(row)
            
            # Create analysis summary
            analysis_summary = {
                'property_id': property_id,
                'building_type': target_property.building_type,
                'ytd_emission_actual': f"{ytd_actual_emission:,.0f} tCO₂e",
                'ytd_emission_target': f"{ytd_target_emission:,.0f} tCO₂e",
                'emission_deviation': f"{emission_deviation*100:+.1f}%",
                'ytd_cost_actual': f"${ytd_actual_cost:,.0f}k AUD",
                'ytd_cost_target': f"${ytd_target_cost:,.0f}k AUD",
                'cost_deviation': f"{cost_deviation*100:+.1f}%",
                'needs_reoptimization': needs_reoptimization,
                'deviation_threshold': f"{deviation_threshold*100:.0f}%",
                'recommendations': self._generate_reoptimization_recommendations(emission_deviation, cost_deviation, property_id)
            }
            
            # Create consumptions DataFrame
            headers = ['Source type'] + months
            consumptions_df = pd.DataFrame(consumptions_data, columns=headers)
            
            return plot_data, consumptions_df, analysis_summary, "Re-optimization analysis completed successfully!"
            
        except Exception as e:
            logger.error(f"Error in annual plan reoptimization: {str(e)}")
            return None, None, None, f"Error: {str(e)}"
    
    def _generate_reoptimization_recommendations(self, emission_deviation: float, cost_deviation: float, property_id: str) -> List[str]:
        """Generate specific recommendations based on deviation analysis"""
        recommendations = []
        
        if emission_deviation > 0.05:  # Over emission target
            recommendations.append(f"Property {property_id}: Implement additional emission reduction measures - exceeding target by {emission_deviation*100:.1f}%")
            recommendations.append("Consider accelerating planned energy efficiency upgrades")
            recommendations.append("Review and optimize HVAC system operations")
        elif emission_deviation < -0.05:  # Under emission target
            recommendations.append(f"Property {property_id}: Ahead of emission targets by {abs(emission_deviation)*100:.1f}%")
            recommendations.append("Consider reallocating budget to other high-impact properties")
        
        if cost_deviation > 0.05:  # Over cost target
            recommendations.append(f"Property {property_id}: Cost overrun of {cost_deviation*100:.1f}% detected")
            recommendations.append("Review and optimize planned capital expenditures")
            recommendations.append("Consider phasing implementation of non-critical upgrades")
        elif cost_deviation < -0.05:  # Under cost target
            recommendations.append(f"Property {property_id}: Under budget by {abs(cost_deviation)*100:.1f}%")
            recommendations.append("Opportunity to accelerate additional improvement projects")
        
        if not recommendations:
            recommendations.append("All targets are on track - continue current implementation plan")
        
        return recommendations
    
    def execute_replanning(self, property_id: str) -> Tuple[Optional[Dict], str]:
        """Execute the re-planning process and return updated targets"""
        try:
            # This would trigger the re-optimization algorithm
            plot_data, consumptions_df, analysis_summary, status = self.reoptimize_annual_plan(property_id, 0.05)
            
            if plot_data and analysis_summary['needs_reoptimization']:
                # Update the internal planning data (in a real system, this would persist to database)
                replanning_result = {
                    'status': 'completed',
                    'property_id': property_id,
                    'updated_emission_targets': plot_data['emission_refined_target'],
                    'updated_cost_targets': plot_data['cost_refined_target'],
                    'optimization_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'previous_deviation': analysis_summary['emission_deviation']
                }
                
                return replanning_result, "Re-planning completed successfully! New targets have been optimized."
            else:
                return None, "No re-planning needed - current plan is within acceptable deviation range."
                
        except Exception as e:
            logger.error(f"Error in re-planning execution: {str(e)}")
            return None, f"Error during re-planning: {str(e)}"