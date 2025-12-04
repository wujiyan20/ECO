# ai_functions.py - AI/ML Functions for EcoAssist
# Version: 3.0.0 - Integrated with Models Layer
"""
AI/ML algorithms for carbon reduction planning and optimization.

This module provides:
- MilestoneOptimizer: Optimize milestone scenarios with ensemble ML models
- PropertyTargetAllocator: Intelligent property-level target allocation
- StrategicPatternAnalyzer: Analyze strategic patterns and recommend actions
- ReoptimizationEngine: Annual plan re-optimization based on actual performance

Integration:
- Uses models from models/ package (Property, MilestoneScenario, etc.)
- Used by services from services/ package (AIService, MilestoneService, etc.)
- Called by API layer (api_module1_milestones.py, etc.)

Version History:
- 1.0.0: Initial implementation
- 2.0.0: Added strategic pattern analysis
- 3.0.0: Full integration with models layer
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Union
import logging
import json
import warnings
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import pickle
import hashlib

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# =============================================================================
# MODELS INTEGRATION
# =============================================================================

# Import models - graceful fallback if not available
try:
    from models import (
        # Core Models
        Property,
        PropertyFilter,
        PropertyMetrics,
        Portfolio,
        MilestoneScenario,
        MilestoneTarget,
        MilestoneProgress,
        ScenarioComparison,
        ReductionOption,
        ReductionStrategy,
        CostProjection,
        CapexOpex,
        
        # Enums
        BuildingType,
        RetrofitPotential,
        ScenarioType,
        AllocationMethod,
        OnTrackStatus,
        RiskLevel,
        PriorityLevel,
        StrategyType,
        
        # Utilities
        generate_uuid,
        generate_scenario_id,
        validate_percentage,
        validate_positive_number,
        calculate_carbon_intensity,
        get_current_year,
        
        # Repositories
        PropertyRepository,
        MilestoneRepository
    )
    MODELS_AVAILABLE = True
    logger.info("Models package loaded successfully")
except ImportError as e:
    MODELS_AVAILABLE = False
    logger.warning(f"Models package not available: {e}. Using standalone mode.")
    
    # Define fallback enums for standalone operation
    class BuildingType(Enum):
        OFFICE = "office"
        RETAIL = "retail"
        INDUSTRIAL = "industrial"
        RESIDENTIAL = "residential"
        MIXED_USE = "mixed_use"
        WAREHOUSE = "warehouse"
    
    class RetrofitPotential(Enum):
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
    
    class ScenarioType(Enum):
        STANDARD = "standard"
        AGGRESSIVE = "aggressive"
        CONSERVATIVE = "conservative"
        AI_OPTIMIZED = "ai_optimized"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OptimizationResult:
    """Result of optimization algorithm"""
    optimal_values: Dict[str, float]
    objective_value: float
    convergence_status: str
    iterations: int
    execution_time: float
    confidence_score: float
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "optimal_values": self.optimal_values,
            "objective_value": self.objective_value,
            "convergence_status": self.convergence_status,
            "iterations": self.iterations,
            "execution_time": self.execution_time,
            "confidence_score": self.confidence_score,
            "validation_metrics": self.validation_metrics
        }


@dataclass
class AIModelMetrics:
    """Metrics for AI model performance"""
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    r2: float   # R-squared
    mape: float = 0.0  # Mean Absolute Percentage Error
    cross_val_score: float = 0.0
    training_time: float = 0.0
    prediction_time: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "mape": self.mape,
            "cross_val_score": self.cross_val_score,
            "training_time": self.training_time,
            "prediction_time": self.prediction_time
        }


@dataclass
class PropertyAllocationResult:
    """Result of property target allocation"""
    property_id: str
    property_name: str
    baseline_emission: float
    allocated_target: float
    reduction_percentage: float
    allocation_weight: float
    confidence_score: float
    recommended_strategies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "property_id": self.property_id,
            "property_name": self.property_name,
            "baseline_emission": self.baseline_emission,
            "allocated_target": self.allocated_target,
            "reduction_percentage": self.reduction_percentage,
            "allocation_weight": self.allocation_weight,
            "confidence_score": self.confidence_score,
            "recommended_strategies": self.recommended_strategies
        }


@dataclass
class ScenarioRecommendation:
    """AI-generated scenario recommendation"""
    scenario_id: str
    scenario_name: str
    scenario_type: str
    reduction_2030: float
    reduction_2050: float
    ai_confidence: float
    estimated_cost: float
    risk_level: str
    key_strategies: List[str]
    implementation_timeline: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "scenario_type": self.scenario_type,
            "reduction_2030": self.reduction_2030,
            "reduction_2050": self.reduction_2050,
            "ai_confidence": self.ai_confidence,
            "estimated_cost": self.estimated_cost,
            "risk_level": self.risk_level,
            "key_strategies": self.key_strategies,
            "implementation_timeline": self.implementation_timeline
        }


@dataclass
class DeviationAnalysisResult:
    """Result of performance deviation analysis"""
    scenario_id: str
    analysis_date: str
    deviation_metrics: Dict[str, float]
    root_causes: List[str]
    contributing_factors: List[str]
    reoptimization_needed: bool
    recommendations: List[str]
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "scenario_id": self.scenario_id,
            "analysis_date": self.analysis_date,
            "deviation_metrics": self.deviation_metrics,
            "root_causes": self.root_causes,
            "contributing_factors": self.contributing_factors,
            "reoptimization_needed": self.reoptimization_needed,
            "recommendations": self.recommendations,
            "confidence_score": self.confidence_score
        }


# =============================================================================
# BASE AI ALGORITHM
# =============================================================================

class BaseAIAlgorithm(ABC):
    """Abstract base class for AI algorithms"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_history = []
        self.performance_metrics: Optional[AIModelMetrics] = None
        self.feature_importance: Dict[str, float] = {}
        self._version = "3.0.0"
        
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> AIModelMetrics:
        """Train the algorithm"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def optimize(self, objective_function, constraints: Dict) -> OptimizationResult:
        """Optimize parameters"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get model status"""
        return {
            "name": self.name,
            "version": self._version,
            "is_trained": self.is_trained,
            "training_history_count": len(self.training_history),
            "performance_metrics": self.performance_metrics.to_dict() if self.performance_metrics else None,
            "feature_importance": self.feature_importance
        }
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file"""
        if not self.is_trained:
            logger.warning(f"Cannot save untrained model: {self.name}")
            return False
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'name': self.name,
                'version': self._version,
                'training_history': self.training_history,
                'performance_metrics': self.performance_metrics,
                'feature_importance': self.feature_importance,
                'saved_at': datetime.utcnow().isoformat()
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.training_history = model_data.get('training_history', [])
            self.performance_metrics = model_data.get('performance_metrics')
            self.feature_importance = model_data.get('feature_importance', {})
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          training_time: float = 0.0) -> AIModelMetrics:
        """Calculate standard metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Avoid division by zero for MAPE
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0
        
        return AIModelMetrics(
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            training_time=training_time
        )


# =============================================================================
# MILESTONE OPTIMIZER
# =============================================================================

class MilestoneOptimizer(BaseAIAlgorithm):
    """
    AI algorithm for optimizing milestone scenarios.
    
    Uses ensemble models (Random Forest + Gradient Boosting) to:
    - Predict optimal reduction targets
    - Generate scenario recommendations
    - Optimize milestone trajectories
    
    Integration:
    - Uses MilestoneScenario, MilestoneTarget from models
    - Called by MilestoneService from services layer
    """
    
    def __init__(self):
        super().__init__("MilestoneOptimizer")
        self.ensemble_models: Dict[str, Any] = {}
        self._feature_names = [
            "baseline_emission", "area_sqm", "carbon_intensity",
            "building_age", "retrofit_potential_score"
        ]
        
    def train(self, X: np.ndarray, y: np.ndarray,
             feature_names: List[str] = None) -> AIModelMetrics:
        """
        Train ensemble models for milestone optimization.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (reduction percentages or emissions)
            feature_names: Optional feature names for importance tracking
            
        Returns:
            AIModelMetrics with training results
        """
        start_time = datetime.now()
        
        if feature_names:
            self._feature_names = feature_names
        
        # Validate input
        if len(X) < 10:
            logger.warning("Training with limited data (<10 samples)")
        
        # Prepare data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train ensemble models
        models = {
            'rf': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        metrics_list = []
        
        for name, model in models.items():
            logger.info(f"Training {name} model...")
            
            # Fit model
            model.fit(X_scaled, y)
            self.ensemble_models[name] = model
            
            # Calculate metrics
            y_pred = model.predict(X_scaled)
            metrics = self._calculate_metrics(y, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X)), scoring='r2')
            metrics.cross_val_score = cv_scores.mean()
            
            metrics_list.append(metrics)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                for i, importance in enumerate(model.feature_importances_):
                    feat_name = self._feature_names[i] if i < len(self._feature_names) else f"feature_{i}"
                    if feat_name not in self.feature_importance:
                        self.feature_importance[feat_name] = 0
                    self.feature_importance[feat_name] += importance / len(models)
        
        self.is_trained = True
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Average metrics
        avg_metrics = AIModelMetrics(
            mae=np.mean([m.mae for m in metrics_list]),
            rmse=np.mean([m.rmse for m in metrics_list]),
            r2=np.mean([m.r2 for m in metrics_list]),
            mape=np.mean([m.mape for m in metrics_list]),
            cross_val_score=np.mean([m.cross_val_score for m in metrics_list]),
            training_time=training_time
        )
        
        self.performance_metrics = avg_metrics
        self.training_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "samples": len(X),
            "metrics": avg_metrics.to_dict()
        })
        
        logger.info(f"Training complete. R²: {avg_metrics.r2:.4f}, RMSE: {avg_metrics.rmse:.4f}")
        return avg_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (averaged from ensemble)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        start_time = datetime.now()
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        for name, model in self.ensemble_models.items():
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        # Ensemble prediction (weighted average)
        ensemble_pred = np.mean(predictions, axis=0)
        
        prediction_time = (datetime.now() - start_time).total_seconds()
        if self.performance_metrics:
            self.performance_metrics.prediction_time = prediction_time
        
        return ensemble_pred
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        for model in self.ensemble_models.values():
            predictions.append(model.predict(X_scaled))
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Confidence based on agreement between models
        confidence = np.clip(1 - (std_pred / (np.abs(mean_pred) + 1e-6)), 0, 1) * 100
        
        return mean_pred, confidence
    
    def optimize(self, objective_function, constraints: Dict) -> OptimizationResult:
        """
        Optimize milestone targets using differential evolution.
        
        Args:
            objective_function: Function to minimize
            constraints: Dict with 'bounds', 'max_iterations', etc.
            
        Returns:
            OptimizationResult with optimal values
        """
        start_time = datetime.now()
        
        bounds = constraints.get('bounds', [(-1, 1)] * 10)
        max_iter = constraints.get('max_iterations', 1000)
        
        def objective_wrapper(x):
            try:
                return objective_function(x)
            except Exception as e:
                logger.warning(f"Objective function error: {e}")
                return 1e6
        
        result = differential_evolution(
            objective_wrapper,
            bounds,
            maxiter=max_iter,
            popsize=15,
            atol=1e-6,
            seed=42
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        confidence_score = min(100.0, max(0.0, 
            100.0 * (1.0 - result.fun / (1.0 + abs(result.fun)))
        ))
        
        return OptimizationResult(
            optimal_values={f"param_{i}": float(val) for i, val in enumerate(result.x)},
            objective_value=float(result.fun),
            convergence_status="converged" if result.success else "failed",
            iterations=result.nit,
            execution_time=execution_time,
            confidence_score=confidence_score,
            validation_metrics={"function_calls": result.nfev}
        )
    
    def generate_scenario_recommendations(self, 
                                         properties: Union[pd.DataFrame, List[Dict]],
                                         target_constraints: Dict = None) -> List[ScenarioRecommendation]:
        """
        Generate AI-powered scenario recommendations.
        
        Args:
            properties: Property data (DataFrame or list of dicts)
            target_constraints: Optional constraints (budget, timeline, etc.)
            
        Returns:
            List of ScenarioRecommendation objects
        """
        if isinstance(properties, list):
            properties = pd.DataFrame(properties)
        
        target_constraints = target_constraints or {}
        
        # Analyze portfolio
        total_baseline = properties['baseline_emission'].sum()
        avg_intensity = properties.get('carbon_intensity', properties['baseline_emission'] / properties.get('area_sqm', 1)).mean()
        
        # Calculate retrofit distribution
        retrofit_col = 'retrofit_potential' if 'retrofit_potential' in properties.columns else None
        high_retrofit_ratio = 0.33  # Default
        if retrofit_col:
            retrofit_values = properties[retrofit_col].astype(str).str.lower()
            high_retrofit_ratio = (retrofit_values == 'high').sum() / len(properties)
        
        scenarios = []
        
        # Scenario configurations
        scenario_configs = [
            {
                "type": "AI_OPTIMIZED",
                "name": "AI-Optimized Aggressive",
                "reduction_2030": self._calculate_optimal_reduction(properties, 2030, high_retrofit_ratio),
                "reduction_2050": self._calculate_optimal_reduction(properties, 2050, high_retrofit_ratio),
                "risk": "MEDIUM"
            },
            {
                "type": "STANDARD",
                "name": "Portfolio-Balanced (SBTi Aligned)",
                "reduction_2030": 40.0,
                "reduction_2050": 90.0,
                "risk": "LOW"
            },
            {
                "type": "AGGRESSIVE",
                "name": "Accelerated Decarbonization",
                "reduction_2030": 50.0,
                "reduction_2050": 95.0,
                "risk": "HIGH"
            },
            {
                "type": "CONSERVATIVE",
                "name": "Risk-Minimized Gradual",
                "reduction_2030": 30.0,
                "reduction_2050": 80.0,
                "risk": "LOW"
            }
        ]
        
        for config in scenario_configs:
            scenario_id = f"AI-SCN-{config['type'][:3]}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            # Calculate estimated cost
            estimated_cost = self._estimate_scenario_cost(
                total_baseline, 
                config['reduction_2050'],
                target_constraints.get('cost_per_tonne', 150)
            )
            
            # Apply budget constraints if provided
            if 'max_budget' in target_constraints:
                if estimated_cost > target_constraints['max_budget']:
                    config['reduction_2050'] *= target_constraints['max_budget'] / estimated_cost
                    config['reduction_2030'] = min(config['reduction_2030'], config['reduction_2050'] * 0.5)
                    estimated_cost = target_constraints['max_budget']
            
            # AI confidence based on data quality and model performance
            base_confidence = 75.0
            if self.is_trained and self.performance_metrics:
                base_confidence = min(95.0, 70.0 + self.performance_metrics.r2 * 25)
            if high_retrofit_ratio > 0.5:
                base_confidence += 5.0
            
            scenario = ScenarioRecommendation(
                scenario_id=scenario_id,
                scenario_name=config['name'],
                scenario_type=config['type'],
                reduction_2030=round(config['reduction_2030'], 1),
                reduction_2050=round(config['reduction_2050'], 1),
                ai_confidence=round(base_confidence, 1),
                estimated_cost=round(estimated_cost, 2),
                risk_level=config['risk'],
                key_strategies=self._recommend_strategies(properties, config),
                implementation_timeline=self._generate_timeline(config)
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _calculate_optimal_reduction(self, properties: pd.DataFrame, 
                                    target_year: int, 
                                    high_retrofit_ratio: float) -> float:
        """Calculate AI-optimized reduction target"""
        base_reduction = 35 if target_year == 2030 else 85
        
        # Adjust based on portfolio characteristics
        adjustment = high_retrofit_ratio * 15
        
        # Use model predictions if trained
        if self.is_trained and len(properties) > 0:
            try:
                features = self._extract_features(properties)
                predictions = self.predict(features)
                predicted_reduction = np.mean(predictions)
                # Blend with heuristic
                base_reduction = 0.6 * predicted_reduction + 0.4 * base_reduction
            except Exception as e:
                logger.warning(f"Prediction failed, using heuristics: {e}")
        
        optimal = base_reduction + adjustment
        return min(95 if target_year == 2050 else 55, max(20, optimal))
    
    def _extract_features(self, properties: pd.DataFrame) -> np.ndarray:
        """Extract features from properties DataFrame"""
        features = []
        
        for _, prop in properties.iterrows():
            feat = [
                prop.get('baseline_emission', 1000),
                prop.get('area_sqm', 5000),
                prop.get('carbon_intensity', 200),
                datetime.now().year - prop.get('year_built', 2000),
                {'high': 3, 'medium': 2, 'low': 1}.get(
                    str(prop.get('retrofit_potential', 'medium')).lower(), 2
                )
            ]
            features.append(feat)
        
        return np.array(features)
    
    def _estimate_scenario_cost(self, baseline: float, reduction_pct: float,
                               cost_per_tonne: float) -> float:
        """Estimate scenario implementation cost"""
        reduction_amount = baseline * (reduction_pct / 100)
        return reduction_amount * cost_per_tonne
    
    def _recommend_strategies(self, properties: pd.DataFrame, config: Dict) -> List[str]:
        """Recommend strategies based on analysis"""
        strategies = []
        
        # Base strategies for all scenarios
        strategies.append("Energy Efficiency Improvements")
        strategies.append("LED Lighting Upgrades")
        
        # Building type specific
        if 'building_type' in properties.columns:
            building_types = properties['building_type'].value_counts()
            if building_types.get('office', 0) > len(properties) * 0.3:
                strategies.append("Smart Building Management Systems")
            if building_types.get('industrial', 0) > len(properties) * 0.2:
                strategies.append("Process Heat Recovery")
        
        # Aggressive scenarios
        if config.get('reduction_2050', 0) > 85:
            strategies.extend([
                "Solar PV Installation",
                "Heat Pump Conversion",
                "Electric Vehicle Infrastructure"
            ])
        
        return strategies[:6]
    
    def _generate_timeline(self, config: Dict) -> List[Dict[str, Any]]:
        """Generate implementation timeline"""
        return [
            {"year": 2025, "phase": "Planning & Quick Wins", "target_completion": 15},
            {"year": 2027, "phase": "Phase 1 - Efficiency", "target_completion": 30},
            {"year": 2030, "phase": f"{config['reduction_2030']:.0f}% Target", "target_completion": 50},
            {"year": 2035, "phase": "Phase 2 - Renewables", "target_completion": 70},
            {"year": 2040, "phase": "Advanced Technologies", "target_completion": 85},
            {"year": 2050, "phase": f"{config['reduction_2050']:.0f}% Target", "target_completion": 100}
        ]
    
    def create_milestone_scenario(self, recommendation: ScenarioRecommendation,
                                 property_ids: List[str],
                                 baseline_emission: float) -> Optional['MilestoneScenario']:
        """
        Convert recommendation to MilestoneScenario model.
        
        Args:
            recommendation: AI-generated recommendation
            property_ids: List of property IDs
            baseline_emission: Total baseline emission
            
        Returns:
            MilestoneScenario model instance (if models available)
        """
        if not MODELS_AVAILABLE:
            logger.warning("Models not available - cannot create MilestoneScenario")
            return None
        
        # Generate targets
        targets = []
        base_year = 2024
        
        for year in range(base_year, 2051):
            if year <= 2030:
                progress = (year - base_year) / (2030 - base_year)
                reduction = progress * recommendation.reduction_2030
            else:
                progress = (year - 2030) / (2050 - 2030)
                reduction = recommendation.reduction_2030 + progress * (
                    recommendation.reduction_2050 - recommendation.reduction_2030
                )
            
            target_emission = baseline_emission * (1 - reduction / 100)
            
            target = MilestoneTarget(
                year=year,
                target_emission=round(target_emission, 2),
                reduction_from_baseline=round(reduction, 2),
                cumulative_reduction=round(reduction, 2)
            )
            targets.append(target)
        
        scenario_type = ScenarioType(recommendation.scenario_type.lower())
        
        scenario = MilestoneScenario(
            scenario_id=recommendation.scenario_id,
            name=recommendation.scenario_name,
            scenario_type=scenario_type,
            property_ids=property_ids,
            baseline_emission=baseline_emission,
            base_year=base_year,
            mid_term_year=2030,
            long_term_year=2050,
            reduction_target_2030=recommendation.reduction_2030,
            reduction_target_2050=recommendation.reduction_2050,
            targets=targets
        )
        
        return scenario


# =============================================================================
# PROPERTY TARGET ALLOCATOR
# =============================================================================

class PropertyTargetAllocator(BaseAIAlgorithm):
    """
    AI algorithm for intelligent property-level target allocation.
    
    Uses clustering and optimization to:
    - Group similar properties
    - Allocate targets fairly and efficiently
    - Balance feasibility and impact
    
    Integration:
    - Uses Property, PropertyFilter from models
    - Called by AllocationService from services layer
    """
    
    def __init__(self):
        super().__init__("PropertyTargetAllocator")
        self.clustering_model = None
        self.allocation_weights: Dict[str, float] = {}
        self._n_clusters = 3
        
    def train(self, X: np.ndarray, y: np.ndarray) -> AIModelMetrics:
        """
        Train property allocation models.
        
        Args:
            X: Property features
            y: Historical allocation targets
            
        Returns:
            Training metrics
        """
        start_time = datetime.now()
        
        # Prepare data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal clusters
        self._n_clusters = min(3, max(2, len(X) // 5))
        
        # Train clustering model
        self.clustering_model = KMeans(
            n_clusters=self._n_clusters,
            random_state=42,
            n_init=10
        )
        clusters = self.clustering_model.fit_predict(X_scaled)
        
        # Train allocation prediction model
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        
        # Calculate metrics
        y_pred = self.model.predict(X_scaled)
        metrics = self._calculate_metrics(y, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=min(5, len(X)))
        metrics.cross_val_score = cv_scores.mean()
        metrics.training_time = (datetime.now() - start_time).total_seconds()
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_names = ['baseline', 'area', 'intensity', 'age', 'potential']
            for i, imp in enumerate(self.model.feature_importances_):
                name = feature_names[i] if i < len(feature_names) else f"f_{i}"
                self.feature_importance[name] = float(imp)
        
        self.is_trained = True
        self.performance_metrics = metrics
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict optimal allocation weights"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def optimize(self, objective_function, constraints: Dict) -> OptimizationResult:
        """Optimize property allocation using multi-objective optimization"""
        start_time = datetime.now()
        
        def multi_objective(x):
            fairness = self._calculate_fairness(x, constraints)
            efficiency = self._calculate_efficiency(x, constraints)
            feasibility = self._calculate_feasibility(x, constraints)
            
            # Weighted combination (minimize negative = maximize positive)
            weights = constraints.get('objective_weights', [0.4, 0.4, 0.2])
            score = (weights[0] * fairness + weights[1] * efficiency + weights[2] * feasibility)
            return -score
        
        n_properties = constraints.get('n_properties', 10)
        bounds = constraints.get('bounds', [(0, 1)] * n_properties)
        
        result = differential_evolution(
            multi_objective,
            bounds,
            maxiter=500,
            popsize=10,
            seed=42
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            optimal_values={f"property_{i}": float(val) for i, val in enumerate(result.x)},
            objective_value=float(result.fun),
            convergence_status="converged" if result.success else "failed",
            iterations=result.nit,
            execution_time=execution_time,
            confidence_score=85.0,
            validation_metrics={
                "fairness": self._calculate_fairness(result.x, constraints),
                "efficiency": self._calculate_efficiency(result.x, constraints)
            }
        )
    
    def _calculate_fairness(self, allocation: np.ndarray, constraints: Dict) -> float:
        """Calculate Gini-based fairness score"""
        if len(allocation) == 0:
            return 0.0
        
        sorted_alloc = np.sort(allocation)
        n = len(sorted_alloc)
        cumsum = np.cumsum(sorted_alloc)
        
        if cumsum[-1] == 0:
            return 1.0
        
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_alloc))) / (n * cumsum[-1]) - (n + 1) / n
        return max(0, 1 - gini)  # Higher = more fair
    
    def _calculate_efficiency(self, allocation: np.ndarray, constraints: Dict) -> float:
        """Calculate efficiency based on retrofit potential alignment"""
        potentials = constraints.get('retrofit_potentials', np.ones_like(allocation))
        
        if len(allocation) != len(potentials):
            return 0.5
        
        # Higher allocation to higher potential = better efficiency
        correlation = np.corrcoef(allocation, potentials)[0, 1]
        return max(0, (correlation + 1) / 2)  # Normalize to 0-1
    
    def _calculate_feasibility(self, allocation: np.ndarray, constraints: Dict) -> float:
        """Calculate budget feasibility"""
        total_budget = constraints.get('total_budget', 1e9)
        unit_costs = constraints.get('unit_costs', np.ones_like(allocation) * 100000)
        
        total_cost = np.sum(allocation * unit_costs)
        
        if total_cost <= total_budget:
            return 1.0
        else:
            return max(0, 1 - (total_cost - total_budget) / total_budget)
    
    def allocate_targets(self,
                        properties: Union[pd.DataFrame, List[Dict]],
                        total_target: float,
                        method: str = "AI_OPTIMIZED") -> List[PropertyAllocationResult]:
        """
        Allocate reduction targets across properties.
        
        Args:
            properties: Property data
            total_target: Total reduction target (tonnes CO2e)
            method: Allocation method (PROPORTIONAL, INTENSITY_WEIGHTED, AI_OPTIMIZED)
            
        Returns:
            List of PropertyAllocationResult
        """
        if isinstance(properties, list):
            properties = pd.DataFrame(properties)
        
        # Calculate weights based on method
        if method == "PROPORTIONAL":
            weights = self._proportional_weights(properties)
        elif method == "INTENSITY_WEIGHTED":
            weights = self._intensity_weights(properties)
        elif method == "AI_OPTIMIZED":
            weights = self._ai_optimized_weights(properties)
        else:
            weights = np.ones(len(properties)) / len(properties)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Create allocation results
        results = []
        for i, (_, prop) in enumerate(properties.iterrows()):
            baseline = prop.get('baseline_emission', 1000)
            allocated = total_target * weights[i]
            target_emission = baseline - allocated
            reduction_pct = (allocated / baseline) * 100 if baseline > 0 else 0
            
            # Calculate confidence
            confidence = self._calculate_allocation_confidence(prop, weights[i])
            
            result = PropertyAllocationResult(
                property_id=prop.get('property_id', f'PROP-{i+1:03d}'),
                property_name=prop.get('name', f'Property {i+1}'),
                baseline_emission=round(baseline, 2),
                allocated_target=round(target_emission, 2),
                reduction_percentage=round(reduction_pct, 2),
                allocation_weight=round(weights[i], 4),
                confidence_score=round(confidence, 1),
                recommended_strategies=self._get_property_strategies(prop)
            )
            results.append(result)
        
        return results
    
    def _proportional_weights(self, properties: pd.DataFrame) -> np.ndarray:
        """Calculate weights proportional to baseline"""
        baselines = properties.get('baseline_emission', pd.Series([1000] * len(properties)))
        return baselines.values / baselines.sum()
    
    def _intensity_weights(self, properties: pd.DataFrame) -> np.ndarray:
        """Calculate weights based on carbon intensity"""
        if 'carbon_intensity' in properties.columns:
            intensities = properties['carbon_intensity'].values
        else:
            baselines = properties.get('baseline_emission', pd.Series([1000] * len(properties)))
            areas = properties.get('area_sqm', pd.Series([5000] * len(properties)))
            intensities = baselines.values / areas.values
        
        return intensities / intensities.sum()
    
    def _ai_optimized_weights(self, properties: pd.DataFrame) -> np.ndarray:
        """Calculate AI-optimized weights"""
        n = len(properties)
        
        # Use trained model if available
        if self.is_trained:
            try:
                features = self._extract_property_features(properties)
                predictions = self.predict(features)
                return np.abs(predictions) / np.sum(np.abs(predictions))
            except Exception as e:
                logger.warning(f"AI prediction failed: {e}")
        
        # Fallback to heuristic
        weights = np.zeros(n)
        
        for i, (_, prop) in enumerate(properties.iterrows()):
            base = prop.get('baseline_emission', 1000) / 1000  # Normalize
            
            # Retrofit potential factor
            potential = str(prop.get('retrofit_potential', 'medium')).lower()
            potential_mult = {'high': 1.5, 'medium': 1.0, 'low': 0.6}.get(potential, 1.0)
            
            # Intensity factor
            intensity = prop.get('carbon_intensity', 200)
            intensity_mult = min(2.0, intensity / 150)
            
            weights[i] = base * potential_mult * intensity_mult
        
        return weights
    
    def _extract_property_features(self, properties: pd.DataFrame) -> np.ndarray:
        """Extract features for prediction"""
        features = []
        current_year = datetime.now().year
        
        for _, prop in properties.iterrows():
            feat = [
                prop.get('baseline_emission', 1000),
                prop.get('area_sqm', 5000),
                prop.get('carbon_intensity', 200),
                current_year - prop.get('year_built', 2000),
                {'high': 3, 'medium': 2, 'low': 1}.get(
                    str(prop.get('retrofit_potential', 'medium')).lower(), 2
                )
            ]
            features.append(feat)
        
        return np.array(features)
    
    def _calculate_allocation_confidence(self, prop: pd.Series, weight: float) -> float:
        """Calculate confidence for property allocation"""
        confidence = 70.0
        
        # Higher confidence for clear retrofit potential
        potential = str(prop.get('retrofit_potential', 'medium')).lower()
        if potential == 'high':
            confidence += 15
        elif potential == 'medium':
            confidence += 5
        
        # Reasonable weight range
        if 0.05 <= weight <= 0.3:
            confidence += 5
        
        # Model-based adjustment
        if self.is_trained and self.performance_metrics:
            confidence += self.performance_metrics.r2 * 10
        
        return min(95, confidence)
    
    def _get_property_strategies(self, prop: pd.Series) -> List[str]:
        """Get recommended strategies for property"""
        strategies = []
        
        building_type = str(prop.get('building_type', 'office')).lower()
        
        if building_type == 'office':
            strategies = ["HVAC Optimization", "LED Lighting", "Smart Controls"]
        elif building_type == 'retail':
            strategies = ["Refrigeration Efficiency", "Lighting Upgrades", "HVAC"]
        elif building_type == 'industrial':
            strategies = ["Process Optimization", "Heat Recovery", "Motor Upgrades"]
        else:
            strategies = ["Energy Audit", "Lighting", "HVAC Improvements"]
        
        # Add solar if high potential
        if str(prop.get('retrofit_potential', '')).lower() == 'high':
            strategies.append("Solar PV Assessment")
        
        return strategies[:4]


# =============================================================================
# STRATEGIC PATTERN ANALYZER
# =============================================================================

class StrategicPatternAnalyzer(BaseAIAlgorithm):
    """
    AI algorithm for analyzing strategic patterns and risk assessment.
    
    Capabilities:
    - Pattern recognition in reduction strategies
    - Risk assessment and mitigation
    - Best practice identification
    """
    
    def __init__(self):
        super().__init__("StrategicPatternAnalyzer")
        self.pattern_model = None
        self.risk_model = None
        self._patterns_database: List[Dict] = []
        
    def train(self, X: np.ndarray, y: np.ndarray) -> AIModelMetrics:
        """Train pattern analysis models"""
        start_time = datetime.now()
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Pattern performance model
        self.pattern_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=42
        )
        self.pattern_model.fit(X_scaled, y)
        
        # Risk classification (synthetic labels for training)
        risk_labels = np.clip(y / np.max(y) * 3, 0, 2).astype(int)
        self.risk_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.risk_model.fit(X_scaled, risk_labels)
        
        # Metrics
        y_pred = self.pattern_model.predict(X_scaled)
        metrics = self._calculate_metrics(y, y_pred)
        metrics.training_time = (datetime.now() - start_time).total_seconds()
        
        self.is_trained = True
        self.performance_metrics = metrics
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict pattern performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained")
        
        X_scaled = self.scaler.transform(X)
        return self.pattern_model.predict(X_scaled)
    
    def optimize(self, objective_function, constraints: Dict) -> OptimizationResult:
        """Optimize strategy mix"""
        start_time = datetime.now()
        
        # Define strategy bounds (allocation percentages)
        n_strategies = constraints.get('n_strategies', 4)
        bounds = [(0, 1)] * n_strategies
        
        def constrained_objective(x):
            # Ensure allocations sum to 1
            x_normalized = x / np.sum(x) if np.sum(x) > 0 else x
            return objective_function(x_normalized)
        
        result = minimize(
            constrained_objective,
            np.ones(n_strategies) / n_strategies,
            bounds=bounds,
            method='SLSQP',
            constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        )
        
        return OptimizationResult(
            optimal_values={f"strategy_{i}": float(v) for i, v in enumerate(result.x)},
            objective_value=float(result.fun),
            convergence_status="converged" if result.success else "failed",
            iterations=result.nit if hasattr(result, 'nit') else 0,
            execution_time=(datetime.now() - start_time).total_seconds(),
            confidence_score=80.0
        )
    
    def analyze_strategy_pattern(self, 
                                strategy_config: Dict,
                                historical_data: Dict = None) -> Dict[str, Any]:
        """
        Analyze a strategy pattern for effectiveness.
        
        Args:
            strategy_config: Strategy configuration
            historical_data: Optional historical performance data
            
        Returns:
            Analysis results
        """
        analysis = {
            "pattern_name": strategy_config.get('name', 'Custom Strategy'),
            "analysis_date": datetime.utcnow().isoformat(),
            "effectiveness_score": 0.0,
            "risk_assessment": {},
            "recommendations": [],
            "best_practices": [],
            "confidence": 75.0
        }
        
        # Calculate effectiveness based on strategy mix
        allocations = strategy_config.get('allocations', {})
        
        # Base scores for strategy types
        strategy_scores = {
            'energy_efficiency': 85,
            'renewable_energy': 80,
            'electrification': 75,
            'operational_optimization': 70,
            'carbon_offsets': 50
        }
        
        weighted_score = 0
        total_weight = 0
        for strategy, weight in allocations.items():
            base_score = strategy_scores.get(strategy.lower(), 60)
            weighted_score += base_score * weight
            total_weight += weight
        
        analysis['effectiveness_score'] = round(
            weighted_score / total_weight if total_weight > 0 else 60, 1
        )
        
        # Risk assessment
        analysis['risk_assessment'] = self._assess_strategy_risks(strategy_config)
        
        # Recommendations
        analysis['recommendations'] = self._generate_strategy_recommendations(
            strategy_config, analysis['risk_assessment']
        )
        
        # Best practices
        analysis['best_practices'] = self._identify_best_practices(strategy_config)
        
        # Adjust confidence based on model training
        if self.is_trained and self.performance_metrics:
            analysis['confidence'] = min(90, 70 + self.performance_metrics.r2 * 20)
        
        return analysis
    
    def _assess_strategy_risks(self, config: Dict) -> Dict[str, Any]:
        """Assess risks in strategy"""
        risks = {
            "overall_risk_level": "MEDIUM",
            "risk_factors": [],
            "mitigation_strategies": []
        }
        
        allocations = config.get('allocations', {})
        reduction_target = config.get('reduction_2050', 90)
        
        # High target risk
        if reduction_target > 90:
            risks['risk_factors'].append("Aggressive reduction target (>90%)")
            risks['mitigation_strategies'].append("Phase implementation with checkpoints")
        
        # Technology risk
        if allocations.get('renewable_energy', 0) > 0.5:
            risks['risk_factors'].append("High dependency on renewable technology")
            risks['mitigation_strategies'].append("Diversify energy sources")
        
        # Determine overall risk
        n_risks = len(risks['risk_factors'])
        if n_risks >= 3:
            risks['overall_risk_level'] = "HIGH"
        elif n_risks >= 1:
            risks['overall_risk_level'] = "MEDIUM"
        else:
            risks['overall_risk_level'] = "LOW"
        
        return risks
    
    def _generate_strategy_recommendations(self, config: Dict, 
                                          risk_assessment: Dict) -> List[str]:
        """Generate strategy recommendations"""
        recommendations = []
        
        allocations = config.get('allocations', {})
        
        # Efficiency recommendations
        if allocations.get('energy_efficiency', 0) < 0.25:
            recommendations.append("Consider increasing energy efficiency allocation - typically highest ROI")
        
        # Balance recommendation
        if len(allocations) < 3:
            recommendations.append("Diversify strategy mix for resilience")
        
        # Risk-based recommendations
        if risk_assessment.get('overall_risk_level') == 'HIGH':
            recommendations.append("Implement robust monitoring and contingency plans")
        
        recommendations.append("Conduct annual strategy review and adjustment")
        
        return recommendations[:5]
    
    def _identify_best_practices(self, config: Dict) -> List[str]:
        """Identify applicable best practices"""
        practices = [
            "Set science-based targets aligned with SBTi",
            "Implement comprehensive measurement and verification",
            "Engage stakeholders throughout implementation",
            "Document lessons learned for continuous improvement"
        ]
        
        if config.get('reduction_2030', 0) >= 40:
            practices.insert(0, "Early aggressive action aligns with 1.5°C pathway")
        
        return practices[:5]


# =============================================================================
# REOPTIMIZATION ENGINE
# =============================================================================

class ReoptimizationEngine(BaseAIAlgorithm):
    """
    AI engine for annual plan re-optimization based on actual performance.
    
    Capabilities:
    - Deviation detection and analysis
    - Root cause identification
    - Adaptive target adjustment
    - Reoptimization recommendations
    """
    
    def __init__(self):
        super().__init__("ReoptimizationEngine")
        self.deviation_predictor = None
        self.adjustment_model = None
        self._deviation_threshold = 0.10  # 10% deviation triggers reopt
        
    def train(self, X: np.ndarray, y: np.ndarray) -> AIModelMetrics:
        """Train reoptimization models"""
        start_time = datetime.now()
        
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Deviation prediction model
        self.deviation_predictor = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.deviation_predictor.fit(X_scaled, y)
        
        # Adjustment recommendation model
        self.adjustment_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.adjustment_model.fit(X_scaled, y)
        
        # Metrics
        y_pred = self.deviation_predictor.predict(X_scaled)
        metrics = self._calculate_metrics(y, y_pred)
        
        cv_scores = cross_val_score(self.deviation_predictor, X_scaled, y, cv=min(5, len(X)))
        metrics.cross_val_score = cv_scores.mean()
        metrics.training_time = (datetime.now() - start_time).total_seconds()
        
        self.is_trained = True
        self.performance_metrics = metrics
        self.model = self.deviation_predictor
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict deviations"""
        if not self.is_trained:
            raise ValueError("Model must be trained")
        
        X_scaled = self.scaler.transform(X)
        return self.deviation_predictor.predict(X_scaled)
    
    def optimize(self, objective_function, constraints: Dict) -> OptimizationResult:
        """Optimize reoptimization parameters"""
        start_time = datetime.now()
        
        def reopt_objective(x):
            current_targets = constraints.get('current_targets', x)
            deviation_penalty = np.sum(np.abs(x - current_targets))
            implementation_cost = np.sum(x * constraints.get('adjustment_costs', np.ones_like(x)))
            return deviation_penalty + 0.1 * implementation_cost
        
        bounds = constraints.get('bounds', [(0, 1)] * 12)
        initial = constraints.get('initial_guess', np.ones(len(bounds)) * 0.5)
        
        result = minimize(
            reopt_objective,
            initial,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        return OptimizationResult(
            optimal_values={f"period_{i+1}": float(v) for i, v in enumerate(result.x)},
            objective_value=float(result.fun),
            convergence_status="converged" if result.success else "failed",
            iterations=result.nit,
            execution_time=(datetime.now() - start_time).total_seconds(),
            confidence_score=85.0
        )
    
    def analyze_performance(self,
                           actual_data: Dict[str, Any],
                           target_data: Dict[str, Any],
                           scenario_id: str = None) -> DeviationAnalysisResult:
        """
        Comprehensive performance deviation analysis.
        
        Args:
            actual_data: Actual emissions/costs
            target_data: Target emissions/costs
            scenario_id: Optional scenario identifier
            
        Returns:
            DeviationAnalysisResult
        """
        # Calculate deviations
        emissions_actual = np.array(actual_data.get('emissions', []))
        emissions_target = np.array(target_data.get('emissions', []))
        
        if len(emissions_actual) == 0 or len(emissions_target) == 0:
            return DeviationAnalysisResult(
                scenario_id=scenario_id or "UNKNOWN",
                analysis_date=datetime.utcnow().isoformat(),
                deviation_metrics={},
                root_causes=["Insufficient data for analysis"],
                contributing_factors=[],
                reoptimization_needed=False,
                recommendations=["Collect more data before analysis"],
                confidence_score=0.0
            )
        
        # Align arrays
        min_len = min(len(emissions_actual), len(emissions_target))
        emissions_actual = emissions_actual[:min_len]
        emissions_target = emissions_target[:min_len]
        
        # Calculate deviation metrics
        deviations = (emissions_actual - emissions_target) / emissions_target
        
        deviation_metrics = {
            "mean_deviation": float(np.mean(deviations)),
            "max_deviation": float(np.max(np.abs(deviations))),
            "std_deviation": float(np.std(deviations)),
            "trend": "increasing" if np.mean(deviations[-3:]) > np.mean(deviations[:3]) else "decreasing"
        }
        
        # Root cause analysis
        root_causes = self._identify_root_causes(
            emissions_actual, emissions_target, actual_data
        )
        
        # Contributing factors
        contributing_factors = self._identify_contributing_factors(actual_data)
        
        # Determine if reoptimization needed
        reopt_needed = deviation_metrics['max_deviation'] > self._deviation_threshold
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            deviation_metrics, root_causes, reopt_needed
        )
        
        # Calculate confidence
        confidence = 80.0
        if self.is_trained and self.performance_metrics:
            confidence = min(95, 75 + self.performance_metrics.r2 * 20)
        
        return DeviationAnalysisResult(
            scenario_id=scenario_id or f"ANALYSIS-{datetime.utcnow().strftime('%Y%m%d')}",
            analysis_date=datetime.utcnow().isoformat(),
            deviation_metrics=deviation_metrics,
            root_causes=root_causes,
            contributing_factors=contributing_factors,
            reoptimization_needed=reopt_needed,
            recommendations=recommendations,
            confidence_score=confidence
        )
    
    def _identify_root_causes(self, actual: np.ndarray, target: np.ndarray,
                             actual_data: Dict) -> List[str]:
        """Identify root causes of deviation"""
        causes = []
        
        deviation = (actual - target) / target
        
        if np.mean(deviation) > 0.1:
            causes.append("Systematic underperformance against targets")
        
        if np.std(deviation) > 0.15:
            causes.append("High variability in performance")
        
        recent_trend = np.mean(deviation[-3:]) - np.mean(deviation[:3])
        if recent_trend > 0.05:
            causes.append("Deteriorating performance trend")
        elif recent_trend < -0.05:
            causes.append("Improving performance trend")
        
        # External factors
        if actual_data.get('weather_impact', False):
            causes.append("Weather-related impacts detected")
        
        if actual_data.get('occupancy_change', 0) > 0.1:
            causes.append("Significant occupancy changes")
        
        return causes if causes else ["No significant root causes identified"]
    
    def _identify_contributing_factors(self, actual_data: Dict) -> List[str]:
        """Identify contributing factors"""
        factors = []
        
        if 'energy_prices' in actual_data:
            factors.append("Energy price fluctuations")
        
        if 'equipment_issues' in actual_data:
            factors.append("Equipment performance degradation")
        
        factors.extend([
            "Operational efficiency variations",
            "Seasonal consumption patterns",
            "Behavioral factors"
        ])
        
        return factors[:5]
    
    def _generate_recommendations(self, metrics: Dict, 
                                 root_causes: List[str],
                                 reopt_needed: bool) -> List[str]:
        """Generate reoptimization recommendations"""
        recommendations = []
        
        max_dev = metrics.get('max_deviation', 0)
        
        if max_dev > 0.20:
            recommendations.append("URGENT: Immediate intervention required - deviation exceeds 20%")
            recommendations.append("Conduct emergency review of implementation plans")
        elif max_dev > 0.10:
            recommendations.append("Moderate adjustment needed - implement corrective measures")
            recommendations.append("Review and adjust monthly targets")
        elif max_dev > 0.05:
            recommendations.append("Minor deviation - continue monitoring closely")
        
        if reopt_needed:
            recommendations.append("Trigger formal reoptimization process")
            recommendations.append("Update scenario models with actual performance data")
        
        # Add cause-specific recommendations
        for cause in root_causes:
            if "variability" in cause.lower():
                recommendations.append("Implement enhanced monitoring and early warning systems")
            if "trend" in cause.lower() and "deteriorating" in cause.lower():
                recommendations.append("Accelerate planned efficiency improvements")
        
        return recommendations[:6]
    
    def generate_adjusted_targets(self,
                                 current_targets: List[float],
                                 actual_performance: List[float],
                                 remaining_periods: int) -> List[float]:
        """
        Generate adjusted targets to meet overall goal.
        
        Args:
            current_targets: Original target values
            actual_performance: Actual achieved values
            remaining_periods: Number of periods remaining
            
        Returns:
            Adjusted target values for remaining periods
        """
        current_targets = np.array(current_targets)
        actual_performance = np.array(actual_performance)
        
        # Calculate shortfall/surplus
        completed_periods = len(actual_performance)
        expected_progress = np.sum(current_targets[:completed_periods])
        actual_progress = np.sum(actual_performance)
        gap = expected_progress - actual_progress
        
        # Distribute gap across remaining periods
        if remaining_periods <= 0:
            return []
        
        remaining_targets = current_targets[completed_periods:completed_periods + remaining_periods].copy()
        
        if len(remaining_targets) == 0:
            return []
        
        # Adjust remaining targets proportionally
        adjustment_per_period = gap / remaining_periods
        adjusted_targets = remaining_targets + adjustment_per_period
        
        # Ensure targets are reasonable (not negative, not >2x original)
        adjusted_targets = np.clip(adjusted_targets, 0, remaining_targets * 2)
        
        return adjusted_targets.tolist()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_milestone_optimizer() -> MilestoneOptimizer:
    """Factory function to create MilestoneOptimizer"""
    return MilestoneOptimizer()


def create_property_allocator() -> PropertyTargetAllocator:
    """Factory function to create PropertyTargetAllocator"""
    return PropertyTargetAllocator()


def create_pattern_analyzer() -> StrategicPatternAnalyzer:
    """Factory function to create StrategicPatternAnalyzer"""
    return StrategicPatternAnalyzer()


def create_reoptimization_engine() -> ReoptimizationEngine:
    """Factory function to create ReoptimizationEngine"""
    return ReoptimizationEngine()


def create_all_ai_components() -> Dict[str, BaseAIAlgorithm]:
    """Create all AI components"""
    return {
        'milestone_optimizer': create_milestone_optimizer(),
        'property_allocator': create_property_allocator(),
        'pattern_analyzer': create_pattern_analyzer(),
        'reoptimization_engine': create_reoptimization_engine()
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data Classes
    'OptimizationResult',
    'AIModelMetrics',
    'PropertyAllocationResult',
    'ScenarioRecommendation',
    'DeviationAnalysisResult',
    
    # Base Class
    'BaseAIAlgorithm',
    
    # AI Algorithms
    'MilestoneOptimizer',
    'PropertyTargetAllocator',
    'StrategicPatternAnalyzer',
    'ReoptimizationEngine',
    
    # Factory Functions
    'create_milestone_optimizer',
    'create_property_allocator',
    'create_pattern_analyzer',
    'create_reoptimization_engine',
    'create_all_ai_components',
    
    # Status
    'MODELS_AVAILABLE'
]

# Module info
__version__ = "3.0.0"
__author__ = "EcoAssist Development Team"

logger.info(f"AI Functions v{__version__} loaded. Models available: {MODELS_AVAILABLE}")
