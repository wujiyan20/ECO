# services/milestone_service.py - Milestone Calculation Service
"""
Service for milestone planning and scenario generation in the EcoAssist system.

Features:
- AI-powered milestone calculation
- Multiple scenario generation (Standard, Aggressive, Conservative)
- Yearly target interpolation
- Cost projection calculation
- Scenario comparison
- Strategy breakdown analysis
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from models import ValidationError

from .base_service import (
    BaseService,
    ServiceResult,
    ServiceResultStatus,
    measure_time,
    cached,
    transaction
)

# Import models
try:
    from models import (
        MilestoneTarget,
        MilestoneScenario,
        ScenarioComparison,
        MilestoneProgress,
        MilestoneAlert,
        CostProjection,
        CapexOpex,
        ScenarioType,
        ApprovalStatus,
        OnTrackStatus,
        StrategyType,
        MilestoneRepository,
        generate_milestone_id,
        generate_scenario_id,
        validate_year,
        validate_year_range,
        validate_percentage
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    logging.warning("Models package not available - using mock mode")

# Import AI functions
try:
    from ai_functions import MilestoneOptimizer
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logging.warning("AI functions not available - using simplified calculations")

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TRANSFER OBJECTS
# =============================================================================

@dataclass
class MilestoneCalculationRequest:
    """Request for milestone calculation"""
    property_ids: List[str]
    base_year: int = 2024
    mid_term_year: int = 2030
    long_term_year: int = 2050
    baseline_emission: float = 0.0
    reduction_2030: float = 40.0  # Default SBTi-aligned
    reduction_2050: float = 95.0  # Net zero target
    strategy_preferences: Optional[Dict[str, float]] = None
    budget_constraints: Optional[Dict[str, float]] = None


@dataclass
class ScenarioConfig:
    """Configuration for scenario generation"""
    scenario_type: str
    reduction_2030: float
    reduction_2050: float
    cost_multiplier: float
    risk_factor: float
    description: str


@dataclass
class MilestoneCalculationResult:
    """Result of milestone calculation"""
    calculation_id: str
    scenarios: List[Dict[str, Any]]
    recommended_scenario_id: str
    calculation_metadata: Dict[str, Any]


# =============================================================================
# SCENARIO CONFIGURATIONS
# =============================================================================

DEFAULT_SCENARIOS = {
    "STANDARD": ScenarioConfig(
        scenario_type="STANDARD",
        reduction_2030=40.0,
        reduction_2050=90.0,
        cost_multiplier=1.0,
        risk_factor=0.3,
        description="Balanced approach aligned with SBTi targets"
    ),
    "AGGRESSIVE": ScenarioConfig(
        scenario_type="AGGRESSIVE",
        reduction_2030=50.0,
        reduction_2050=95.0,
        cost_multiplier=1.4,
        risk_factor=0.5,
        description="Accelerated decarbonization with higher investment"
    ),
    "CONSERVATIVE": ScenarioConfig(
        scenario_type="CONSERVATIVE",
        reduction_2030=30.0,
        reduction_2050=80.0,
        cost_multiplier=0.7,
        risk_factor=0.2,
        description="Lower risk approach with gradual implementation"
    )
}


# =============================================================================
# MILESTONE SERVICE
# =============================================================================

class MilestoneService(BaseService):
    """
    Service for milestone planning and scenario generation.
    
    Provides:
    - AI-powered milestone calculations
    - Multiple scenario generation
    - Yearly target interpolation
    - Cost projections
    - Scenario comparison and recommendations
    
    Usage:
        service = MilestoneService(db_manager)
        request = MilestoneCalculationRequest(
            property_ids=["PROP-001"],
            baseline_emission=10000.0
        )
        result = service.calculate_milestones(request)
    """
    
    def __init__(self, db_manager=None):
        """
        Initialize milestone service.
        
        Args:
            db_manager: Database manager instance
        """
        super().__init__(db_manager)
        self._milestone_repo: Optional[MilestoneRepository] = None
        self._ai_optimizer: Optional[MilestoneOptimizer] = None
    
    def _do_initialize(self) -> None:
        """Initialize repositories and AI"""
        if self.db_manager and MODELS_AVAILABLE:
            self._milestone_repo = MilestoneRepository(self.db_manager)
            self._logger.info("Milestone repository initialized")
        
        if AI_AVAILABLE:
            self._ai_optimizer = MilestoneOptimizer()
            self._logger.info("AI optimizer initialized")
    
    # =========================================================================
    # MILESTONE CALCULATION
    # =========================================================================
    
    @measure_time
    def calculate_milestones(self, request: MilestoneCalculationRequest) -> ServiceResult[MilestoneCalculationResult]:
        """
        Calculate milestone scenarios for given properties.
        
        Args:
            request: Calculation request with parameters
            
        Returns:
            ServiceResult containing calculation results with multiple scenarios
        """
        # Validate request
        validation_errors = self._validate_calculation_request(request)
        if validation_errors:
            return ServiceResult.validation_error(validation_errors)
        
        return self._execute(self._calculate_milestones_impl, request)
    
    def _validate_calculation_request(self, request: MilestoneCalculationRequest) -> List[str]:
        """Validate calculation request"""
        errors = []
        
        if not request.property_ids:
            errors.append("At least one property_id is required")
        
        if request.baseline_emission <= 0:
            errors.append("baseline_emission must be positive")
        
        try:
            if MODELS_AVAILABLE:
                try:
                    validate_year(int(request.base_year))
                except (ValueError, ValidationError) as e:
                    errors.append(f"Invalid base_year: {str(e)}")

                try:
                    validate_year(int(request.mid_term_year))  # Correct field
                except (ValueError, ValidationError) as e:
                    errors.append(f"Invalid mid_term_year: {str(e)}")

                try:
                    validate_year(int(request.long_term_year))  # Correct field
                except (ValueError, ValidationError) as e:
                    errors.append(f"Invalid long_term_year: {str(e)}")
                validate_year_range(request.base_year, request.long_term_year)
        except ValueError as e:
            errors.append(str(e))
        
        if request.mid_term_year <= request.base_year:
            errors.append("mid_term_year must be after base_year")
        
        if request.long_term_year <= request.mid_term_year:
            errors.append("long_term_year must be after mid_term_year")
        
        return errors
    
    def _calculate_milestones_impl(self, request: MilestoneCalculationRequest) -> ServiceResult[MilestoneCalculationResult]:
        """Implementation of calculate_milestones"""
        calculation_id = generate_milestone_id() if MODELS_AVAILABLE else f"CALC-{datetime.utcnow().timestamp()}"
        start_time = datetime.utcnow()
        
        scenarios = []
        
        # Generate each scenario type
        for scenario_type, config in DEFAULT_SCENARIOS.items():
            scenario = self._generate_scenario(
                request=request,
                config=config,
                calculation_id=calculation_id
            )
            scenarios.append(scenario)
        
        # Determine recommended scenario
        recommended = self._select_recommended_scenario(scenarios, request)
        
        result = MilestoneCalculationResult(
            calculation_id=calculation_id,
            scenarios=scenarios,
            recommended_scenario_id=recommended["scenario_id"],
            calculation_metadata={
                "calculated_at": datetime.utcnow().isoformat(),
                "algorithm_version": "2.1.0",
                "scenarios_generated": len(scenarios),
                "execution_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "ai_optimized": self._ai_optimizer is not None
            }
        )
        
        return ServiceResult.success(result)
    
    def _generate_scenario(self, request: MilestoneCalculationRequest,
                          config: ScenarioConfig,
                          calculation_id: str) -> Dict[str, Any]:
        """Generate a single scenario"""
        scenario_id = generate_scenario_id() if MODELS_AVAILABLE else f"SCN-{config.scenario_type}-{datetime.utcnow().timestamp()}"
        
        # Calculate reduction targets
        reduction_targets = self._interpolate_yearly_targets(
            baseline=request.baseline_emission,
            base_year=request.base_year,
            mid_year=request.mid_term_year,
            long_year=request.long_term_year,
            mid_reduction_pct=config.reduction_2030,
            long_reduction_pct=config.reduction_2050
        )
        
        # Calculate cost projections
        cost_projections = self._calculate_cost_projections(
            reduction_targets=reduction_targets,
            baseline=request.baseline_emission,
            cost_multiplier=config.cost_multiplier
        )
        
        # Calculate strategy breakdown
        strategy_breakdown = self._calculate_strategy_breakdown(
            config.scenario_type,
            request.strategy_preferences
        )
        
        # Calculate scores
        feasibility_score = self._calculate_feasibility_score(config, request)
        cost_efficiency = self._calculate_cost_efficiency(cost_projections, reduction_targets)
        risk_score = config.risk_factor * 100
        
        scenario = {
            "scenario_id": scenario_id,
            "scenario_type": config.scenario_type,
            "description": config.description,
            "base_year": request.base_year,
            "mid_term_year": request.mid_term_year,
            "long_term_year": request.long_term_year,
            "baseline_emission": request.baseline_emission,
            "reduction_2030": config.reduction_2030,
            "reduction_2050": config.reduction_2050,
            "reduction_targets": reduction_targets,
            "cost_projections": cost_projections,
            "strategy_breakdown": strategy_breakdown,
            "scores": {
                "feasibility_score": feasibility_score,
                "cost_efficiency_score": cost_efficiency,
                "risk_score": risk_score,
                "overall_score": (feasibility_score * 0.35 + cost_efficiency * 0.35 + (100 - risk_score) * 0.30)
            },
            "total_capex": sum(cp["breakdown"]["capex"] for cp in cost_projections),
            "total_opex": sum(cp["breakdown"]["opex"] for cp in cost_projections),
            "cumulative_cost": sum(cp["estimated_cost"] for cp in cost_projections),
            "cost_per_tonne_reduced": self._calculate_cost_per_tonne(cost_projections, reduction_targets),
            "approval_status": "PENDING",
            "created_at": datetime.utcnow().isoformat(),
            "property_ids": request.property_ids
        }
        
        return scenario
    
    def _interpolate_yearly_targets(self, baseline: float, base_year: int,
                                   mid_year: int, long_year: int,
                                   mid_reduction_pct: float,
                                   long_reduction_pct: float) -> List[Dict[str, Any]]:
        """
        Interpolate yearly reduction targets using S-curve.
        
        Uses sigmoid function for realistic adoption curve.
        """
        targets = []
        
        # Base year (no reduction)
        targets.append({
            "year": base_year,
            "target_emissions": baseline,
            "reduction_from_baseline": 0.0,
            "cumulative_reduction": 0.0,
            "unit": "kg-CO2e"
        })
        
        # Generate yearly targets
        for year in range(base_year + 1, long_year + 1):
            if year <= mid_year:
                # Phase 1: Base to mid-term
                progress = (year - base_year) / (mid_year - base_year)
                # S-curve: sigmoid function
                s_progress = 1 / (1 + math.exp(-6 * (progress - 0.5)))
                reduction_pct = s_progress * mid_reduction_pct
            else:
                # Phase 2: Mid-term to long-term
                progress = (year - mid_year) / (long_year - mid_year)
                s_progress = 1 / (1 + math.exp(-6 * (progress - 0.5)))
                reduction_pct = mid_reduction_pct + s_progress * (long_reduction_pct - mid_reduction_pct)
            
            target_emission = baseline * (1 - reduction_pct / 100)
            cumulative = baseline - target_emission
            
            targets.append({
                "year": year,
                "target_emissions": round(target_emission, 2),
                "reduction_from_baseline": round(reduction_pct, 2),
                "cumulative_reduction": round(cumulative, 2),
                "unit": "kg-CO2e"
            })
        
        return targets
    
    def _calculate_cost_projections(self, reduction_targets: List[Dict],
                                   baseline: float,
                                   cost_multiplier: float) -> List[Dict[str, Any]]:
        """Calculate yearly cost projections"""
        projections = []
        
        # Cost per tonne CO2 reduced (industry benchmark)
        base_capex_per_tonne = 150 * cost_multiplier
        base_opex_per_tonne = 30 * cost_multiplier
        
        for i, target in enumerate(reduction_targets):
            if i == 0:
                continue  # Skip base year
            
            # Calculate year's reduction
            prev_reduction = reduction_targets[i-1]["cumulative_reduction"]
            year_reduction = target["cumulative_reduction"] - prev_reduction
            
            # Calculate costs
            capex = year_reduction * base_capex_per_tonne
            opex = target["cumulative_reduction"] * base_opex_per_tonne * 0.05  # 5% of cumulative for maintenance
            
            # Apply learning curve (costs decrease over time)
            years_from_start = i
            learning_factor = max(0.7, 1 - (years_from_start * 0.01))
            
            projections.append({
                "year": target["year"],
                "estimated_cost": round((capex + opex) * learning_factor, 2),
                "breakdown": {
                    "capex": round(capex * learning_factor, 2),
                    "opex": round(opex * learning_factor, 2)
                },
                "unit": "USD"
            })
        
        return projections
    
    def _calculate_strategy_breakdown(self, scenario_type: str,
                                     preferences: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Calculate strategy allocation breakdown"""
        # Default breakdowns by scenario type
        defaults = {
            "STANDARD": {
                "energy_efficiency": 35.0,
                "renewable_energy": 30.0,
                "electrification": 20.0,
                "operational_optimization": 15.0
            },
            "AGGRESSIVE": {
                "energy_efficiency": 25.0,
                "renewable_energy": 40.0,
                "electrification": 25.0,
                "operational_optimization": 10.0
            },
            "CONSERVATIVE": {
                "energy_efficiency": 45.0,
                "renewable_energy": 20.0,
                "electrification": 15.0,
                "operational_optimization": 20.0
            }
        }
        
        base_breakdown = defaults.get(scenario_type, defaults["STANDARD"])
        
        # Apply preferences if provided
        if preferences:
            for strategy, weight in preferences.items():
                if strategy in base_breakdown:
                    base_breakdown[strategy] = weight
            
            # Normalize to 100%
            total = sum(base_breakdown.values())
            if total > 0:
                base_breakdown = {k: (v / total) * 100 for k, v in base_breakdown.items()}
        
        return base_breakdown
    
    def _calculate_feasibility_score(self, config: ScenarioConfig,
                                    request: MilestoneCalculationRequest) -> float:
        """Calculate feasibility score (0-100)"""
        score = 100.0
        
        # Reduce score for aggressive targets
        if config.reduction_2030 > 45:
            score -= (config.reduction_2030 - 45) * 2
        
        if config.reduction_2050 > 90:
            score -= (config.reduction_2050 - 90) * 1.5
        
        # Consider budget constraints
        if request.budget_constraints:
            if request.budget_constraints.get("limited", False):
                score -= 10
        
        return max(0, min(100, score))
    
    def _calculate_cost_efficiency(self, cost_projections: List[Dict],
                                  reduction_targets: List[Dict]) -> float:
        """Calculate cost efficiency score (0-100)"""
        if not cost_projections or not reduction_targets:
            return 50.0
        
        total_cost = sum(cp["estimated_cost"] for cp in cost_projections)
        total_reduction = reduction_targets[-1]["cumulative_reduction"]
        
        if total_reduction <= 0:
            return 50.0
        
        cost_per_tonne = total_cost / total_reduction
        
        # Benchmark: $150/tonne is average
        # Score of 100 at $75/tonne, 50 at $150/tonne, 0 at $300/tonne
        score = 100 - ((cost_per_tonne - 75) / 2.25)
        
        return max(0, min(100, score))
    
    def _calculate_cost_per_tonne(self, cost_projections: List[Dict],
                                 reduction_targets: List[Dict]) -> float:
        """Calculate cost per tonne of CO2 reduced"""
        total_cost = sum(cp["estimated_cost"] for cp in cost_projections)
        total_reduction = reduction_targets[-1]["cumulative_reduction"] if reduction_targets else 0
        
        if total_reduction > 0:
            return round(total_cost / total_reduction, 2)
        return 0.0
    
    def _select_recommended_scenario(self, scenarios: List[Dict],
                                    request: MilestoneCalculationRequest) -> Dict:
        """Select the recommended scenario based on scores and preferences"""
        if not scenarios:
            return None
        
        # Sort by overall score
        sorted_scenarios = sorted(
            scenarios,
            key=lambda s: s["scores"]["overall_score"],
            reverse=True
        )
        
        return sorted_scenarios[0]
    
    # =========================================================================
    # SCENARIO MANAGEMENT
    # =========================================================================
    
    @measure_time
    @transaction
    def save_milestone_scenario(self, scenario: Dict[str, Any],
                               approved_by: str = None) -> ServiceResult[str]:
        """
        Save a milestone scenario to database.
        
        Args:
            scenario: Scenario data to save
            approved_by: User approving the scenario
            
        Returns:
            ServiceResult containing saved scenario ID
        """
        return self._execute(self._save_scenario_impl, scenario, approved_by)
    
    def _save_scenario_impl(self, scenario: Dict[str, Any],
                           approved_by: str) -> ServiceResult[str]:
        """Implementation of save_milestone_scenario"""
        scenario_id = scenario.get("scenario_id")
        
        # Update approval info
        if approved_by:
            scenario["approval_status"] = "APPROVED"
            scenario["approved_by"] = approved_by
            scenario["approval_date"] = datetime.utcnow().isoformat()
        
        if self._milestone_repo and MODELS_AVAILABLE:
            # Convert to model
            milestone_scenario = MilestoneScenario(
                scenario_id=scenario_id,
                scenario_type=ScenarioType(scenario.get("scenario_type", "STANDARD")),
                description=scenario.get("description", ""),
                base_year=scenario.get("base_year", 2024),
                mid_term_year=scenario.get("mid_term_year", 2030),
                long_term_year=scenario.get("long_term_year", 2050),
                baseline_emission=scenario.get("baseline_emission", 0),
                mid_term_target=scenario.get("mid_term_target", 0),
                long_term_target=scenario.get("long_term_target", 0),
                yearly_targets=scenario.get("yearly_targets", {}),
                total_capex=scenario.get("total_capex", 0),
                total_opex=scenario.get("total_opex", 0),
                feasibility_score=scenario.get("scores", {}).get("feasibility_score", 0),
                cost_efficiency_score=scenario.get("scores", {}).get("cost_efficiency_score", 0),
                overall_score=scenario.get("scores", {}).get("overall_score", 0),
                approval_status=ApprovalStatus(scenario.get("approval_status", "PENDING"))
            )
            
            saved = self._milestone_repo.create(milestone_scenario)
            
            # Invalidate cache
            self._invalidate_cache("milestone_scenarios")
            
            return ServiceResult.success(
                data=saved.scenario_id,
                message=f"Scenario {scenario_id} saved successfully"
            )
        
        # Mock mode
        return ServiceResult.success(
            data=scenario_id,
            message="Scenario saved (mock mode)"
        )
    
    @measure_time
    @cached(ttl_seconds=300, key_prefix="milestone_scenario")
    def get_milestone_scenario(self, scenario_id: str) -> ServiceResult[Dict[str, Any]]:
        """
        Get a milestone scenario by ID.
        
        Args:
            scenario_id: Scenario ID
            
        Returns:
            ServiceResult containing scenario data
        """
        if not scenario_id:
            return ServiceResult.validation_error(["scenario_id is required"])
        
        return self._execute(self._get_scenario_impl, scenario_id)
    
    def _get_scenario_impl(self, scenario_id: str) -> ServiceResult[Dict[str, Any]]:
        """Implementation of get_milestone_scenario"""
        if self._milestone_repo:
            scenario = self._milestone_repo.get_by_id(scenario_id)
            if scenario:
                return ServiceResult.success(scenario.to_dict() if hasattr(scenario, 'to_dict') else scenario)
            return ServiceResult.not_found(f"Scenario {scenario_id} not found")
        
        # Mock mode
        return self._mock_get_scenario(scenario_id)
    
    def _mock_get_scenario(self, scenario_id: str) -> ServiceResult[Dict[str, Any]]:
        """Mock scenario for testing"""
        return ServiceResult.success({
            "scenario_id": scenario_id,
            "scenario_type": "STANDARD",
            "description": "Mock scenario",
            "base_year": 2024,
            "mid_term_year": 2030,
            "long_term_year": 2050,
            "baseline_emission": 10000.0,
            "reduction_2030": 40.0,
            "reduction_2050": 90.0,
            "approval_status": "PENDING"
        })
    
    @measure_time
    def get_all_scenarios(self, property_ids: List[str] = None,
                         status: str = None) -> ServiceResult[List[Dict[str, Any]]]:
        """
        Get all milestone scenarios with optional filtering.
        
        Args:
            property_ids: Filter by property IDs
            status: Filter by approval status
            
        Returns:
            ServiceResult containing list of scenarios
        """
        return self._execute(self._get_all_scenarios_impl, property_ids, status)
    
    def _get_all_scenarios_impl(self, property_ids: List[str],
                               status: str) -> ServiceResult[List[Dict[str, Any]]]:
        """Implementation of get_all_scenarios"""
        if self._milestone_repo:
            scenarios = self._milestone_repo.get_all_scenarios(property_ids)
            
            # Filter by status if provided
            if status:
                scenarios = [s for s in scenarios if s.approval_status.value == status]
            
            return ServiceResult.success([
                s.to_dict() if hasattr(s, 'to_dict') else s
                for s in scenarios
            ])
        
        # Mock mode
        return ServiceResult.success([])
    
    # =========================================================================
    # SCENARIO COMPARISON
    # =========================================================================
    
    @measure_time
    def compare_scenarios(self, scenario_ids: List[str]) -> ServiceResult[Dict[str, Any]]:
        """
        Compare multiple scenarios side by side.
        
        Args:
            scenario_ids: List of scenario IDs to compare
            
        Returns:
            ServiceResult containing comparison data
        """
        if not scenario_ids or len(scenario_ids) < 2:
            return ServiceResult.validation_error(["At least 2 scenario IDs required"])
        
        return self._execute(self._compare_scenarios_impl, scenario_ids)
    
    def _compare_scenarios_impl(self, scenario_ids: List[str]) -> ServiceResult[Dict[str, Any]]:
        """Implementation of compare_scenarios"""
        scenarios = []
        
        for sid in scenario_ids:
            result = self.get_milestone_scenario(sid)
            if result.is_success:
                scenarios.append(result.data)
        
        if len(scenarios) < 2:
            return ServiceResult.error("Could not retrieve enough scenarios for comparison")
        
        # Build comparison
        comparison = {
            "comparison_id": f"CMP-{datetime.utcnow().timestamp()}",
            "scenarios": scenarios,
            "comparison_matrix": self._build_comparison_matrix(scenarios),
            "recommendations": self._generate_recommendations(scenarios),
            "best_by_criteria": self._find_best_by_criteria(scenarios)
        }
        
        return ServiceResult.success(comparison)
    
    def _build_comparison_matrix(self, scenarios: List[Dict]) -> Dict[str, List]:
        """Build comparison matrix for key metrics"""
        metrics = ["reduction_2030", "reduction_2050", "total_capex", "total_opex"]
        
        matrix = {}
        for metric in metrics:
            matrix[metric] = [
                {"scenario_id": s["scenario_id"], "value": s.get(metric, 0)}
                for s in scenarios
            ]
        
        # Add scores
        score_metrics = ["feasibility_score", "cost_efficiency_score", "overall_score"]
        for metric in score_metrics:
            matrix[metric] = [
                {"scenario_id": s["scenario_id"], "value": s.get("scores", {}).get(metric, 0)}
                for s in scenarios
            ]
        
        return matrix
    
    def _generate_recommendations(self, scenarios: List[Dict]) -> List[str]:
        """Generate recommendations based on scenario comparison"""
        recommendations = []
        
        # Find best overall
        best = max(scenarios, key=lambda s: s.get("scores", {}).get("overall_score", 0))
        recommendations.append(
            f"'{best['scenario_type']}' scenario has the highest overall score of "
            f"{best.get('scores', {}).get('overall_score', 0):.1f}"
        )
        
        # Check for trade-offs
        most_ambitious = max(scenarios, key=lambda s: s.get("reduction_2050", 0))
        most_cost_efficient = max(scenarios, key=lambda s: s.get("scores", {}).get("cost_efficiency_score", 0))
        
        if most_ambitious["scenario_id"] != most_cost_efficient["scenario_id"]:
            recommendations.append(
                f"Consider '{most_cost_efficient['scenario_type']}' if budget is constrained, "
                f"or '{most_ambitious['scenario_type']}' for maximum impact"
            )
        
        return recommendations
    
    def _find_best_by_criteria(self, scenarios: List[Dict]) -> Dict[str, str]:
        """Find best scenario for each criteria"""
        return {
            "highest_reduction": max(scenarios, key=lambda s: s.get("reduction_2050", 0))["scenario_id"],
            "lowest_cost": min(scenarios, key=lambda s: s.get("total_capex", float('inf')))["scenario_id"],
            "best_feasibility": max(scenarios, key=lambda s: s.get("scores", {}).get("feasibility_score", 0))["scenario_id"],
            "best_overall": max(scenarios, key=lambda s: s.get("scores", {}).get("overall_score", 0))["scenario_id"]
        }
    
    # =========================================================================
    # PROGRESS TRACKING
    # =========================================================================
    
    @measure_time
    def calculate_progress(self, scenario_id: str, current_year: int,
                          actual_emission: float) -> ServiceResult[MilestoneProgress]:
        """
        Calculate progress against milestone targets.
        
        Args:
            scenario_id: Scenario to track
            current_year: Current year
            actual_emission: Actual emission value
            
        Returns:
            ServiceResult containing progress data
        """
        # Get scenario
        scenario_result = self.get_milestone_scenario(scenario_id)
        if not scenario_result.is_success:
            return scenario_result
        
        scenario = scenario_result.data
        
        return self._execute(
            self._calculate_progress_impl,
            scenario, current_year, actual_emission
        )
    
    def _calculate_progress_impl(self, scenario: Dict, current_year: int,
                                actual_emission: float) -> ServiceResult:
        """Implementation of calculate_progress"""
        # Find target for current year
        targets = scenario.get("reduction_targets", [])
        target_data = None
        
        for t in targets:
            if t["year"] == current_year:
                target_data = t
                break
        
        if not target_data:
            return ServiceResult.error(f"No target found for year {current_year}")
        
        target_emission = target_data["target_emissions"]
        variance = actual_emission - target_emission
        variance_pct = (variance / target_emission) * 100 if target_emission > 0 else 0
        
        # Determine status
        if variance_pct <= -5:
            status = "AHEAD"
        elif variance_pct <= 5:
            status = "ON_TRACK"
        elif variance_pct <= 15:
            status = "AT_RISK"
        elif variance_pct <= 25:
            status = "OFF_TRACK"
        else:
            status = "CRITICAL"
        
        baseline = scenario.get("baseline_emission", actual_emission)
        progress_pct = ((baseline - actual_emission) / baseline) * 100 if baseline > 0 else 0
        
        progress = {
            "scenario_id": scenario.get("scenario_id"),
            "year": current_year,
            "target_emission": target_emission,
            "actual_emission": actual_emission,
            "variance": round(variance, 2),
            "variance_percentage": round(variance_pct, 2),
            "on_track_status": status,
            "progress_percentage": round(progress_pct, 2),
            "recommended_actions": self._get_recommended_actions(status, variance_pct)
        }
        
        if MODELS_AVAILABLE:
            return ServiceResult.success(MilestoneProgress(**progress))
        return ServiceResult.success(progress)
    
    def _get_recommended_actions(self, status: str, variance_pct: float) -> List[str]:
        """Get recommended actions based on status"""
        actions = {
            "AHEAD": ["Consider accelerating future initiatives", "Review budget reallocation options"],
            "ON_TRACK": ["Continue current trajectory", "Monitor key performance indicators"],
            "AT_RISK": ["Review implementation timeline", "Identify quick-win opportunities"],
            "OFF_TRACK": ["Conduct root cause analysis", "Accelerate high-impact initiatives"],
            "CRITICAL": ["Immediate management review required", "Consider scope adjustment"]
        }
        return actions.get(status, ["Review current strategy"])
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    @measure_time
    def get_target_for_year(self, scenario_id: str, year: int) -> ServiceResult[Dict[str, Any]]:
        """Get target emission for specific year"""
        scenario_result = self.get_milestone_scenario(scenario_id)
        if not scenario_result.is_success:
            return scenario_result
        
        scenario = scenario_result.data
        targets = scenario.get("reduction_targets", [])
        
        for t in targets:
            if t["year"] == year:
                return ServiceResult.success(t)
        
        return ServiceResult.not_found(f"No target for year {year}")
    
    @measure_time
    def validate_scenario(self, scenario: Dict[str, Any]) -> ServiceResult[Dict[str, Any]]:
        """Validate scenario data"""
        errors = []
        warnings = []
        
        # Required fields
        required = ["scenario_type", "base_year", "baseline_emission"]
        for field in required:
            if field not in scenario or scenario[field] is None:
                errors.append(f"{field} is required")
        
        # Validate reduction targets
        reduction_2030 = scenario.get("reduction_2030", 0)
        reduction_2050 = scenario.get("reduction_2050", 0)
        
        if reduction_2030 > reduction_2050:
            errors.append("2050 reduction must be greater than or equal to 2030 reduction")
        
        if reduction_2050 > 100:
            errors.append("Reduction percentage cannot exceed 100%")
        
        # Warnings
        if reduction_2030 < 30:
            warnings.append("2030 target may not align with SBTi recommendations")
        
        if reduction_2050 < 90:
            warnings.append("2050 target may not achieve net-zero alignment")
        
        return ServiceResult.success({
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        })


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'MilestoneService',
    'MilestoneCalculationRequest',
    'MilestoneCalculationResult',
    'ScenarioConfig',
    'DEFAULT_SCENARIOS'
]
