# services/milestone_service.py - Milestone Calculation Service with AI
"""
Service for milestone planning and scenario generation in the EcoAssist system.

Features:
- AI-powered milestone calculation (NEW!)
- Multiple scenario generation (Standard, Aggressive, Conservative)
- Yearly target interpolation
- Cost projection calculation
- Scenario comparison
- Strategy breakdown analysis

AI Integration:
- Tries AI-based predictions first
- Falls back to rule-based calculations if AI unavailable
- Seamless integration with existing functionality
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from models import ValidationError

from database.database_manager import DatabaseManager

from .base_service import (
    BaseService,
    ServiceResult,
    ServiceResultStatus,
    measure_time,
    cached,
    transaction
)

# Import AI Inference Service (NEW!)
from .ai_inference_service import get_ai_service

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

# Import AI functions (LEGACY - keeping for compatibility)
try:
    from ai_functions import MilestoneOptimizer
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logging.warning("AI functions not available - using simplified calculations")

logger = logging.getLogger(__name__)


SCENARIO_COEFFICIENTS = {
    "Standard": 1.0,      # Baseline - use prediction as-is
    "Aggressive": 1.3,    # 30% faster reduction
    "Conservative": 0.8   # 20% slower reduction
}

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


# Default scenario configurations
DEFAULT_SCENARIOS = {
    "STANDARD": ScenarioConfig(
        scenario_type="STANDARD",
        reduction_2030=40.0,
        reduction_2050=95.0,
        cost_multiplier=1.0,
        risk_factor=0.3,
        description="Balanced approach following SBTi guidelines"
    ),
    "AGGRESSIVE": ScenarioConfig(
        scenario_type="AGGRESSIVE",
        reduction_2030=50.0,
        reduction_2050=100.0,
        cost_multiplier=1.3,
        risk_factor=0.6,
        description="Accelerated pathway with front-loaded investments"
    ),
    "CONSERVATIVE": ScenarioConfig(
        scenario_type="CONSERVATIVE",
        reduction_2030=30.0,
        reduction_2050=90.0,
        cost_multiplier=0.8,
        risk_factor=0.15,
        description="Gradual approach with proven technologies"
    )
}


# =============================================================================
# MILESTONE SERVICE
# =============================================================================

class MilestoneService(BaseService):
    """
    Service for milestone calculation and management
    
    Provides milestone-based emission reduction planning with:
    - AI-enhanced trajectory prediction (NEW!)
    - Multiple scenario generation
    - Cost-benefit analysis
    - Strategy breakdown
    - Database persistence
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None,
                 ai_optimizer: Any = None):
        """
        Initialize MilestoneService
        
        Args:
            db_manager: Database manager for persistence (optional)
            ai_optimizer: AI optimizer instance (legacy, optional)
        """
        super().__init__(db_manager)
        self._ai_optimizer = ai_optimizer
        
        # Check if we should use database persistence
        self.use_database_for_persistence = (
            db_manager is not None and 
            hasattr(db_manager, 'create_scenario')
        )
        
        if self.use_database_for_persistence:
            self._logger.info("MilestoneService will save scenarios to database")
        
        self._logger.info("MilestoneService initialized")
    
    # =========================================================================
    # PUBLIC INTERFACE
    # =========================================================================
    
    @measure_time
    def calculate_milestones(self, request: MilestoneCalculationRequest) -> ServiceResult[MilestoneCalculationResult]:
        """
        Calculate milestone scenarios with AI enhancement
        
        This method:
        1. Attempts AI-based trajectory prediction
        2. Falls back to rule-based calculation if AI unavailable
        3. Generates multiple scenarios (Standard, Aggressive, Conservative)
        4. Saves results to database
        
        Args:
            request: Milestone calculation request
        
        Returns:
            ServiceResult containing calculated scenarios
        """
        return self._execute(self._calculate_milestones_impl, request)
    
    # =========================================================================
    # CORE IMPLEMENTATION
    # =========================================================================
    
    def _calculate_milestones_impl(self, request: MilestoneCalculationRequest) -> ServiceResult[MilestoneCalculationResult]:
        """Implementation of calculate_milestones"""
        calculation_id = generate_milestone_id() if MODELS_AVAILABLE else f"CALC-{datetime.utcnow().timestamp()}"
        start_time = datetime.utcnow()
        
        # Check if AI is available
        ai_service = get_ai_service()
        ai_used = False
        
        scenarios = []
        
        # Generate each scenario type
        for scenario_type, config in DEFAULT_SCENARIOS.items():
            scenario = self._generate_scenario(
                request=request,
                config=config,
                calculation_id=calculation_id,
                ai_service=ai_service  # Pass AI service to scenario generation
            )
            
            # Check if AI was used for this scenario
            if scenario.get('ai_enhanced', False):
                ai_used = True
            
            scenarios.append(scenario)
        
        # Determine recommended scenario
        recommended = self._select_recommended_scenario(scenarios, request)
        
        result = MilestoneCalculationResult(
            calculation_id=calculation_id,
            scenarios=scenarios,
            recommended_scenario_id=recommended["scenario_id"],
            calculation_metadata={
                "calculated_at": datetime.utcnow().isoformat(),
                "algorithm_version": "2.2.0",  # Incremented for AI integration
                "scenarios_generated": len(scenarios),
                "execution_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "ai_optimized": ai_used,  # NEW: Track if AI was used
                "ai_available": ai_service is not None and ai_service.is_available()
            }
        )
        
        # Save to database
        # if self.use_database_for_persistence and request.base_year:
            # try:
                # # Save recommended scenario to database
                # recommended_scenario = next(
                    # s for s in scenarios if s['scenario_id'] == recommended["scenario_id"]
                # )
                
                # # Calculate target emission and reduction required
                # target_emission = request.baseline_emission * (1 - recommended_scenario.get('reduction_2050', 0) / 100)
                # reduction_required = request.baseline_emission * (recommended_scenario.get('reduction_2050', 0) / 100)
                
                # db_scenario_id = self.db_manager.create_scenario({
                    # 'user_id': getattr(request, 'user_id', None) or '00000000-0000-0000-0000-000000000001',
                    # 'scenario_name': f"{recommended_scenario['scenario_type']} Scenario - {request.base_year}-{request.long_term_year}",
                    # 'baseline_year': request.base_year,
                    # 'target_year': request.long_term_year,
                    # 'baseline_emission': request.baseline_emission,
                    # 'target_reduction_percentage': recommended_scenario.get('reduction_2050', 0),
                    # 'scenario_type': recommended_scenario['scenario_type'].lower(),
                    # 'description': recommended_scenario.get('description', ''),
                    # 'target_emission': target_emission,
                    # 'reduction_required': reduction_required,
                    # 'status': 'calculated'
                # })
                
                # # Save milestones
                # reduction_targets = recommended_scenario.get('reduction_targets', [])
                # if reduction_targets:
                    # milestones = []
                    # for target in reduction_targets:
                        # milestones.append({
                            # 'year': target.get('year'),
                            # 'target_emission': target.get('target_emissions'),
                            # 'reduction_from_baseline': target.get('reduction_from_baseline'),
                            # 'reduction_percentage': target.get('reduction_from_baseline'),
                            # 'cumulative_reduction': target.get('cumulative_reduction'),
                            # 'annual_reduction': 0  # Calculate if needed
                        # })
                    
                    # self.db_manager.create_scenario_milestones(db_scenario_id, milestones)
                    # ai_status = "AI-enhanced" if ai_used else "Rule-based"
                    # self._logger.info(f"✅ Saved {ai_status} scenario to database: {db_scenario_id} with {len(milestones)} milestones")
                # else:
                    # self._logger.info(f"✅ Saved scenario to database: {db_scenario_id}")
                
            # except Exception as e:
                # self._logger.warning(f"⚠️ Failed to save scenario to database: {e}")
                # import traceback
                # self._logger.warning(traceback.format_exc())
        
        if self.use_database_for_persistence and request.base_year:
            try:
                # Save ALL scenarios to database (not just recommended)
                saved_count = 0
                
                for scenario in scenarios:
                    try:
                        # Calculate target emission and reduction required
                        target_emission = request.baseline_emission * (1 - scenario.get('reduction_2050', 0) / 100)
                        reduction_required = request.baseline_emission * (scenario.get('reduction_2050', 0) / 100)
                        
                        # Use the scenario_id from the scenario (the UUID we generated)
                        scenario_id = scenario['scenario_id']
                        
                        # Create scenario with EXPLICIT scenario_id
                        db_scenario_id = self.db_manager.create_scenario({
                            'scenario_id': scenario_id,  # ← USE THE GENERATED UUID!
                            'user_id': getattr(request, 'user_id', None) or '00000000-0000-0000-0000-000000000001',
                            'scenario_name': f"{scenario['scenario_type']} Scenario - {request.base_year}-{request.long_term_year}",
                            'baseline_year': request.base_year,
                            'target_year': request.long_term_year,
                            'baseline_emission': request.baseline_emission,
                            'target_reduction_percentage': scenario.get('reduction_2050', 0),
                            'scenario_type': scenario['scenario_type'].lower(),
                            'description': scenario.get('description', ''),
                            'target_emission': target_emission,
                            'reduction_required': reduction_required,
                            'status': 'calculated'
                        })
                        
                        # Save milestones for this scenario
                        reduction_targets = scenario.get('reduction_targets', [])
                        if reduction_targets:
                            milestones = []
                            for target in reduction_targets:
                                milestones.append({
                                    'year': target.get('year'),
                                    'target_emission': target.get('target_emissions'),
                                    'reduction_from_baseline': target.get('reduction_from_baseline'),
                                    'reduction_percentage': target.get('reduction_from_baseline'),
                                    'cumulative_reduction': target.get('cumulative_reduction'),
                                    'annual_reduction': 0
                                })
                            
                            self.db_manager.create_scenario_milestones(scenario_id, milestones)
                            ai_status = "AI-enhanced" if ai_used else "Rule-based"
                            self._logger.info(f"✅ Saved {ai_status} {scenario['scenario_type']} scenario to database: {scenario_id} with {len(milestones)} milestones")
                        else:
                            self._logger.info(f"✅ Saved {scenario['scenario_type']} scenario to database: {scenario_id}")
                        
                        saved_count += 1
                        
                    except Exception as e:
                        self._logger.warning(f"⚠️ Failed to save {scenario['scenario_type']} scenario to database: {e}")
                        import traceback
                        self._logger.warning(traceback.format_exc())
                
                self._logger.info(f"✅ Saved {saved_count}/{len(scenarios)} scenarios to database")
                
            except Exception as e:
                self._logger.warning(f"⚠️ Failed to save scenarios to database: {e}")
                import traceback
                self._logger.warning(traceback.format_exc())
        
        return ServiceResult.success(result)
    
    
    # def _apply_scenario_coefficient(
        # self,
        # predictions: List[float],
        # baseline_emission: float,
        # coefficient: float
    # ) -> List[float]:
        # """
        # Apply scenario coefficient to emission predictions
        
        # Multiplies the reduction amount (not the emission itself) by the coefficient.
        # This preserves the baseline while scaling the reduction rate.
        # """
        # adjusted = []
        
        # for predicted_emission in predictions:
            # # Calculate reduction from baseline
            # reduction_amount = baseline_emission - predicted_emission
            
            # # Apply coefficient to the reduction
            # adjusted_reduction = reduction_amount * coefficient
            
            # # Calculate new emission
            # adjusted_emission = baseline_emission - adjusted_reduction
            
            # # Ensure non-negative
            # adjusted_emission = max(0, adjusted_emission)
            
            # adjusted.append(adjusted_emission)
        
        # return adjusted

    def _apply_scenario_coefficient(
        self,
        predictions: List[float],
        baseline_emission: float,
        coefficient: float
    ) -> List[float]:
        """
        Apply scenario coefficient to emission predictions
        
        Args:
            predictions: Raw emission predictions from AI
            baseline_emission: Baseline emission value
            coefficient: Scenario multiplier (0.8 for Conservative, 1.3 for Aggressive)
        
        Returns:
            Adjusted predictions
        """
        adjusted = []
        
        for i, predicted_emission in enumerate(predictions):
            # Calculate reduction from baseline
            reduction_amount = baseline_emission - predicted_emission
            
            # Apply coefficient to the reduction (not the absolute emission)
            adjusted_reduction = reduction_amount * coefficient
            
            # Calculate new emission
            adjusted_emission = baseline_emission - adjusted_reduction
            
            # Ensure non-negative
            adjusted_emission = max(0, adjusted_emission)
            
            adjusted.append(adjusted_emission)
        
        return adjusted
    
    # def _apply_scenario_coefficient_to_targets(
        # self,
        # targets: List[Dict[str, Any]],
        # baseline_emission: float,
        # coefficient: float
    # ) -> List[Dict[str, Any]]:
        # """
        # Apply scenario coefficient to existing target dictionaries
        # """
        # adjusted_targets = []
        
        # for target in targets:
            # year = target['year']
            # original_emission = target['target_emissions']
            
            # # Calculate reduction from baseline
            # reduction_amount = baseline_emission - original_emission
            
            # # Apply coefficient
            # adjusted_reduction = reduction_amount * coefficient
            
            # # Calculate new emission
            # adjusted_emission = baseline_emission - adjusted_reduction
            # adjusted_emission = max(0, adjusted_emission)
            
            # # Calculate new reduction percentage
            # reduction_pct = ((baseline_emission - adjusted_emission) / baseline_emission * 100) if baseline_emission > 0 else 0
            
            # adjusted_targets.append({
                # 'year': year,
                # 'target_emissions': round(adjusted_emission, 2),
                # 'reduction_from_baseline': round(reduction_pct, 2)
            # })
        
        # return adjusted_targets
    
    def _apply_scenario_coefficient_to_targets(
        self,
        targets: List[Dict[str, Any]],
        baseline_emission: float,
        coefficient: float
    ) -> List[Dict[str, Any]]:
        """
        Apply scenario coefficient to existing targets
        
        Args:
            targets: List of target dictionaries
            baseline_emission: Baseline emission value
            coefficient: Scenario multiplier
        
        Returns:
            Adjusted targets
        """
        adjusted_targets = []
        
        for target in targets:
            year = target['year']
            original_emission = target['target_emissions']
            
            # Calculate reduction from baseline
            reduction_amount = baseline_emission - original_emission
            
            # Apply coefficient
            adjusted_reduction = reduction_amount * coefficient
            
            # Calculate new emission
            adjusted_emission = baseline_emission - adjusted_reduction
            adjusted_emission = max(0, adjusted_emission)
            
            # Calculate new reduction percentage
            reduction_pct = ((baseline_emission - adjusted_emission) / baseline_emission * 100) if baseline_emission > 0 else 0
            
            adjusted_targets.append({
                'year': year,
                'target_emissions': round(adjusted_emission, 2),
                'reduction_from_baseline': round(reduction_pct, 2)
            })
        
        return adjusted_targets
    
    
    def _generate_scenario(self, request: MilestoneCalculationRequest,
                          config: ScenarioConfig,
                          calculation_id: str,
                          ai_service=None) -> Dict[str, Any]:
        """
        Generate a single scenario with AI enhancement
        
        Args:
            request: Calculation request
            config: Scenario configuration
            calculation_id: Unique calculation ID
            ai_service: AI inference service (optional)
        """
        # scenario_id = generate_scenario_id() if MODELS_AVAILABLE else f"SCN-{config.scenario_type}-{datetime.utcnow().timestamp()}"
        
        import uuid  # Add at top if not present
        scenario_id = str(uuid.uuid4())
        
        ai_enhanced = False
        
        # Calculate reduction targets (AI-enhanced or rule-based)
        reduction_targets = self._interpolate_yearly_targets(
            base_year=request.base_year,
            mid_term_year=request.mid_term_year,
            long_term_year=request.long_term_year,
            baseline_emission=request.baseline_emission,
            mid_term_reduction=config.reduction_2030,
            long_term_reduction=config.reduction_2050,
            ai_service=ai_service,
            scenario_type=config.scenario_type
        )
        
        # Check if AI was used
        if ai_service and ai_service.is_available():
            ai_enhanced = True
            self._logger.info(f"✅ Generated {config.scenario_type} scenario using AI predictions")
        else:
            self._logger.info(f"Generated {config.scenario_type} scenario using rule-based calculation")
        
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
            "property_ids": request.property_ids,
            "ai_enhanced": ai_enhanced  # NEW: Track if AI was used
        }
        
        return scenario
    
    # def _interpolate_yearly_targets(
                                    # self,
                                    # base_year: int,
                                    # mid_term_year: int,
                                    # long_term_year: int,
                                    # baseline_emission: float,
                                    # mid_term_reduction: float,
                                    # long_term_reduction: float,
                                    # ai_service=None,
                                    # scenario_type: str = "Standard"  # ADD THIS
                                # ) -> List[Dict[str, Any]]:
        # """
        # Interpolate yearly reduction targets using AI or S-curve
        
        # NEW: Tries AI prediction first, falls back to rule-based S-curve
        
        # Args:
            # baseline: Baseline emission
            # base_year: Starting year
            # mid_year: Mid-term target year (e.g., 2030)
            # long_year: Long-term target year (e.g., 2050)
            # mid_reduction_pct: Mid-term reduction percentage
            # long_reduction_pct: Long-term reduction percentage
            # ai_service: AI inference service (optional)
        
        # Returns:
            # List of yearly target dictionaries
        # """
        # # Get scenario coefficient
        # coefficient = SCENARIO_COEFFICIENTS.get(scenario_type, 1.0)
        # logger.info(f"Applying {scenario_type} scenario coefficient: {coefficient}")
        # # Try AI prediction first
        # if ai_service and ai_service.is_available():
            # try:
                # target_years = long_year - base_year
                
                # # Get AI predictions
                # ai_predictions = ai_service.predict_milestones(
                    # baseline_emission=baseline,
                    # target_years=target_years,
                    # building_type="Office",  # Default, could be from request
                    # building_age=10,         # Default
                    # area_sqm=5000           # Default
                # )
                
                # if ai_predictions and len(ai_predictions) > 0:
                    # # Build targets from AI predictions
                    # targets = self._build_targets_from_ai_predictions(
                        # ai_predictions,
                        # baseline,
                        # base_year,
                        # long_year
                    # )
                    
                    # self._logger.info(f"✅ Using AI-predicted trajectory ({len(targets)} milestones)")
                    # return targets
                
            # except Exception as e:
                # self._logger.warning(f"⚠️ AI prediction failed, using rule-based fallback: {e}")
        
        # # Fallback to rule-based S-curve calculation
        # return self._interpolate_yearly_targets_rule_based(
            # baseline, base_year, mid_year, long_year,
            # mid_reduction_pct, long_reduction_pct
        # )
        
        
    def _interpolate_yearly_targets(
        self,
        base_year: int,
        mid_term_year: int,
        long_term_year: int,
        baseline_emission: float,
        mid_term_reduction: float,
        long_term_reduction: float,
        ai_service=None,
        scenario_type: str = "Standard"
    ) -> List[Dict[str, Any]]:
        """
        Interpolate yearly emission targets using AI or S-curve
        
        Args:
            scenario_type: "Standard", "Aggressive", or "Conservative"
        """
        # Get scenario coefficient
        coefficient = SCENARIO_COEFFICIENTS.get(scenario_type, 1.0)
        logger.info(f"Applying {scenario_type} scenario coefficient: {coefficient}")
        
        # Try AI prediction first
        if ai_service and ai_service.is_available():
            try:
                # Calculate number of years to predict
                target_years = int(long_term_year - base_year)
                
                predictions = ai_service.predict_milestones(
                    baseline_emission=baseline_emission,
                    target_years=target_years,
                    building_type="Office",
                    building_age=10,
                    area_sqm=5000
                )
                
                if predictions and len(predictions) > 0:
                    # APPLY SCENARIO COEFFICIENT to AI predictions
                    adjusted_predictions = self._apply_scenario_coefficient(
                        predictions, 
                        baseline_emission,
                        coefficient
                    )
                    
                    # BUILD TARGETS - CORRECT PARAMETER ORDER
                    targets = self._build_targets_from_ai_predictions(
                        adjusted_predictions,      # predictions
                        baseline_emission,         # baseline
                        int(base_year),           # base_year
                        int(long_term_year)       # long_year - ADDED!
                    )
                    
                    logger.info(f"✅ Using AI-predicted milestones with {scenario_type} coefficient ({coefficient})")
                    return targets
                    
            except Exception as e:
                logger.warning(f"⚠️  AI prediction failed: {e}")
        
        # Fallback to rule-based with coefficient
        logger.info(f"⚠️  AI service not available, using rule-based calculation with {scenario_type} coefficient")
        
        # CALL RULE-BASED - CORRECT PARAMETER ORDER!
        targets = self._interpolate_yearly_targets_rule_based(
            baseline_emission,          # 1st - baseline
            int(base_year),             # 2nd - base_year
            int(mid_term_year),         # 3rd - mid_year
            int(long_term_year),        # 4th - long_year
            mid_term_reduction,         # 5th - mid_reduction_pct
            long_term_reduction         # 6th - long_reduction_pct
        )
        
        # APPLY SCENARIO COEFFICIENT to rule-based predictions
        adjusted_targets = self._apply_scenario_coefficient_to_targets(
            targets,
            baseline_emission,
            coefficient
        )
        
        return adjusted_targets
    
    def _build_targets_from_ai_predictions(self, predictions: List[float],
                                          baseline: float,
                                          base_year: int,
                                          long_year: int) -> List[Dict[str, Any]]:
        """
        Build target structure from AI predictions
        
        Args:
            predictions: List of emission values from AI
            baseline: Baseline emission
            base_year: Starting year
            long_year: End year
        
        Returns:
            List of yearly target dictionaries
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
        
        # Generate targets from AI predictions
        for i, emission in enumerate(predictions):
            year = base_year + i + 1
            
            if year > long_year:
                break
            
            reduction_pct = ((baseline - emission) / baseline) * 100 if baseline > 0 else 0
            cumulative_reduction = baseline - emission
            
            targets.append({
                "year": year,
                "target_emissions": round(emission, 2),
                "reduction_from_baseline": round(reduction_pct, 2),
                "cumulative_reduction": round(cumulative_reduction, 2),
                "unit": "kg-CO2e"
            })
        
        return targets
    
    def _interpolate_yearly_targets_rule_based(self, baseline: float, base_year: int,
                                              mid_year: int, long_year: int,
                                              mid_reduction_pct: float,
                                              long_reduction_pct: float) -> List[Dict[str, Any]]:
        """
        RULE-BASED: Interpolate yearly reduction targets using S-curve
        
        This is the original implementation, kept as fallback
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
    # ADDITIONAL METHODS (keeping existing functionality)
    # =========================================================================
    
    # ... rest of the methods remain unchanged ...
