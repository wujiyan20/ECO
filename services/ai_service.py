# services/ai_service.py - AI/ML Integration Service
"""
Service for AI/ML operations in the EcoAssist system.

Features:
- AI model management
- Optimization algorithms
- Prediction services
- Training and retraining
- Model performance monitoring
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

from .base_service import (
    BaseService,
    ServiceResult,
    ServiceResultStatus,
    measure_time,
    cached,
    retry
)

# Import AI functions
try:
    from ai_functions import (
        MilestoneOptimizer,
        PropertyTargetAllocator,
        StrategicPatternAnalyzer,
        ReoptimizationEngine,
        OptimizationResult,
        AIModelMetrics
    )
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logging.warning("AI functions not available")

# Import models
try:
    from models import (
        Property,
        MilestoneScenario,
        ScenarioType
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TRANSFER OBJECTS
# =============================================================================

class ModelType(Enum):
    """Types of AI models"""
    MILESTONE_OPTIMIZER = "milestone_optimizer"
    TARGET_ALLOCATOR = "target_allocator"
    PATTERN_ANALYZER = "pattern_analyzer"
    REOPTIMIZATION_ENGINE = "reoptimization_engine"


class ModelStatus(Enum):
    """Model status"""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZED = "initialized"
    TRAINED = "trained"
    READY = "ready"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about an AI model"""
    model_type: str
    status: str
    version: str
    trained_at: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    training_samples: int = 0
    last_prediction_at: Optional[str] = None
    prediction_count: int = 0


@dataclass
class PredictionRequest:
    """Request for AI prediction"""
    model_type: str
    input_data: Dict[str, Any]
    options: Optional[Dict[str, Any]] = None


@dataclass
class PredictionResult:
    """Result of AI prediction"""
    prediction_id: str
    model_type: str
    predictions: Any
    confidence: float
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingRequest:
    """Request for model training"""
    model_type: str
    training_data: List[Dict[str, Any]]
    validation_split: float = 0.2
    hyperparameters: Optional[Dict[str, Any]] = None


@dataclass
class TrainingResult:
    """Result of model training"""
    model_type: str
    success: bool
    metrics: Dict[str, float]
    training_time_seconds: float
    samples_used: int
    validation_samples: int


# =============================================================================
# AI SERVICE
# =============================================================================

class AIService(BaseService):
    """
    Service for AI/ML operations.
    
    Provides:
    - Model initialization and management
    - Prediction services
    - Training and retraining
    - Performance monitoring
    - Optimization algorithms
    
    Usage:
        service = AIService()
        service.initialize()
        
        # Make predictions
        result = service.predict(PredictionRequest(
            model_type="milestone_optimizer",
            input_data={...}
        ))
    """
    
    def __init__(self, db_manager=None):
        """
        Initialize AI service.
        
        Args:
            db_manager: Database manager for persisting model data
        """
        super().__init__(db_manager)
        
        # Model instances
        self._models: Dict[str, Any] = {}
        self._model_status: Dict[str, ModelStatus] = {}
        self._model_metrics: Dict[str, Dict[str, float]] = {}
        self._prediction_counts: Dict[str, int] = {}
        
        # Configuration
        self._model_version = "2.1.0"
    
    def _do_initialize(self) -> None:
        """Initialize AI models"""
        if not AI_AVAILABLE:
            self._logger.warning("AI functions not available - using mock mode")
            return
        
        try:
            # Initialize models
            self._models["milestone_optimizer"] = MilestoneOptimizer()
            self._model_status["milestone_optimizer"] = ModelStatus.INITIALIZED
            
            self._models["target_allocator"] = PropertyTargetAllocator()
            self._model_status["target_allocator"] = ModelStatus.INITIALIZED
            
            # Optional models
            try:
                self._models["pattern_analyzer"] = StrategicPatternAnalyzer()
                self._model_status["pattern_analyzer"] = ModelStatus.INITIALIZED
            except Exception:
                self._logger.warning("Pattern analyzer not available")
            
            try:
                self._models["reoptimization_engine"] = ReoptimizationEngine()
                self._model_status["reoptimization_engine"] = ModelStatus.INITIALIZED
            except Exception:
                self._logger.warning("Reoptimization engine not available")
            
            self._logger.info(f"Initialized {len(self._models)} AI models")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize AI models: {e}")
    
    def _do_shutdown(self) -> None:
        """Cleanup AI resources"""
        self._models.clear()
        self._model_status.clear()
    
    # =========================================================================
    # MODEL MANAGEMENT
    # =========================================================================
    
    @measure_time
    def get_model_status(self, model_type: str = None) -> ServiceResult[Union[ModelInfo, List[ModelInfo]]]:
        """
        Get status of AI models.
        
        Args:
            model_type: Specific model type, or None for all
            
        Returns:
            ServiceResult containing model info
        """
        if model_type:
            return self._get_single_model_status(model_type)
        
        # All models
        model_infos = []
        for mt in self._models.keys():
            result = self._get_single_model_status(mt)
            if result.is_success:
                model_infos.append(result.data)
        
        return ServiceResult.success(model_infos)
    
    def _get_single_model_status(self, model_type: str) -> ServiceResult[ModelInfo]:
        """Get status of a single model"""
        model = self._models.get(model_type)
        if not model:
            return ServiceResult.not_found(f"Model {model_type} not found")
        
        status = self._model_status.get(model_type, ModelStatus.NOT_INITIALIZED)
        metrics = self._model_metrics.get(model_type, {})
        
        # Check if model is trained
        is_trained = getattr(model, 'is_trained', False)
        if is_trained:
            status = ModelStatus.TRAINED
        
        info = ModelInfo(
            model_type=model_type,
            status=status.value,
            version=self._model_version,
            trained_at=metrics.get("trained_at"),
            metrics={k: v for k, v in metrics.items() if k != "trained_at"},
            training_samples=int(metrics.get("training_samples", 0)),
            prediction_count=self._prediction_counts.get(model_type, 0)
        )
        
        return ServiceResult.success(info)
    
    @measure_time
    def check_model_health(self) -> ServiceResult[Dict[str, Any]]:
        """
        Check health of all AI models.
        
        Returns:
            ServiceResult containing health status
        """
        health = {
            "ai_available": AI_AVAILABLE,
            "models_initialized": len(self._models),
            "models_trained": sum(
                1 for m in self._models.values() 
                if getattr(m, 'is_trained', False)
            ),
            "model_details": {}
        }
        
        for model_type, model in self._models.items():
            health["model_details"][model_type] = {
                "status": self._model_status.get(model_type, ModelStatus.NOT_INITIALIZED).value,
                "trained": getattr(model, 'is_trained', False),
                "predictions": self._prediction_counts.get(model_type, 0)
            }
        
        return ServiceResult.success(health)
    
    # =========================================================================
    # PREDICTION SERVICES
    # =========================================================================
    
    @measure_time
    def predict(self, request: PredictionRequest) -> ServiceResult[PredictionResult]:
        """
        Make predictions using specified model.
        
        Args:
            request: Prediction request
            
        Returns:
            ServiceResult containing predictions
        """
        if not AI_AVAILABLE:
            return self._mock_predict(request)
        
        model = self._models.get(request.model_type)
        if not model:
            return ServiceResult.not_found(f"Model {request.model_type} not found")
        
        return self._execute(self._predict_impl, request, model)
    
    def _predict_impl(self, request: PredictionRequest, model: Any) -> ServiceResult[PredictionResult]:
        """Implementation of predict"""
        import numpy as np
        
        start_time = time.perf_counter()
        prediction_id = f"PRED-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        
        try:
            # Prepare input data
            input_array = self._prepare_input(request.input_data)
            
            # Check if model needs training
            if not getattr(model, 'is_trained', False):
                # Auto-train with synthetic data if possible
                self._auto_train_model(request.model_type, model)
            
            # Make prediction
            predictions = model.predict(input_array)
            
            # Calculate confidence
            confidence = self._calculate_confidence(model, predictions)
            
            # Update prediction count
            self._prediction_counts[request.model_type] = \
                self._prediction_counts.get(request.model_type, 0) + 1
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            result = PredictionResult(
                prediction_id=prediction_id,
                model_type=request.model_type,
                predictions=predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                confidence=confidence,
                execution_time_ms=execution_time,
                metadata={
                    "input_shape": input_array.shape if hasattr(input_array, 'shape') else len(request.input_data),
                    "model_version": self._model_version
                }
            )
            
            return ServiceResult.success(result)
            
        except Exception as e:
            self._logger.error(f"Prediction failed: {e}")
            return ServiceResult.error(f"Prediction failed: {str(e)}")
    
    def _mock_predict(self, request: PredictionRequest) -> ServiceResult[PredictionResult]:
        """Mock prediction when AI not available"""
        prediction_id = f"PRED-MOCK-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Generate mock predictions based on model type
        if request.model_type == "milestone_optimizer":
            predictions = {
                "reduction_2030": 40.0,
                "reduction_2050": 90.0,
                "optimal_strategy": "balanced"
            }
        elif request.model_type == "target_allocator":
            num_properties = len(request.input_data.get("property_ids", [1]))
            predictions = [1.0 / num_properties] * num_properties
        else:
            predictions = [0.5]
        
        result = PredictionResult(
            prediction_id=prediction_id,
            model_type=request.model_type,
            predictions=predictions,
            confidence=0.75,
            execution_time_ms=10.0,
            metadata={"mode": "mock"}
        )
        
        return ServiceResult.success(result)
    
    def _prepare_input(self, input_data: Dict[str, Any]) -> 'np.ndarray':
        """Prepare input data for model"""
        import numpy as np
        
        # Handle different input formats
        if "features" in input_data:
            return np.array(input_data["features"])
        
        if "properties" in input_data:
            # Extract features from properties
            features = []
            for prop in input_data["properties"]:
                feat = [
                    prop.get("baseline_emission", 1000),
                    prop.get("area_sqm", 5000),
                    prop.get("carbon_intensity", 200)
                ]
                features.append(feat)
            return np.array(features)
        
        # Default: use values as features
        return np.array([list(input_data.values())])
    
    def _calculate_confidence(self, model: Any, predictions: Any) -> float:
        """Calculate prediction confidence"""
        if hasattr(model, 'performance_metrics') and model.performance_metrics:
            r2 = getattr(model.performance_metrics, 'r2', 0.7)
            return min(0.99, max(0.5, r2))
        return 0.75
    
    def _auto_train_model(self, model_type: str, model: Any) -> None:
        """Auto-train model with synthetic data"""
        import numpy as np
        
        self._logger.info(f"Auto-training {model_type} with synthetic data")
        
        # Generate synthetic training data
        n_samples = 100
        n_features = 3
        
        X = np.random.randn(n_samples, n_features) * 100 + 500
        y = X[:, 0] * 0.1 + X[:, 1] * 0.05 + np.random.randn(n_samples) * 10
        
        try:
            model.train(X, y)
            self._model_status[model_type] = ModelStatus.TRAINED
            self._logger.info(f"Auto-trained {model_type}")
        except Exception as e:
            self._logger.warning(f"Auto-training failed: {e}")
    
    # =========================================================================
    # TRAINING SERVICES
    # =========================================================================
    
    @measure_time
    def train_model(self, request: TrainingRequest) -> ServiceResult[TrainingResult]:
        """
        Train or retrain a model.
        
        Args:
            request: Training request
            
        Returns:
            ServiceResult containing training results
        """
        if not AI_AVAILABLE:
            return ServiceResult.error("AI functions not available for training")
        
        model = self._models.get(request.model_type)
        if not model:
            return ServiceResult.not_found(f"Model {request.model_type} not found")
        
        return self._execute(self._train_impl, request, model)
    
    def _train_impl(self, request: TrainingRequest, model: Any) -> ServiceResult[TrainingResult]:
        """Implementation of train_model"""
        import numpy as np
        
        start_time = time.perf_counter()
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data(request.training_data)
            
            # Split into training and validation
            split_idx = int(len(X) * (1 - request.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Apply hyperparameters if provided
            if request.hyperparameters:
                for param, value in request.hyperparameters.items():
                    if hasattr(model, param):
                        setattr(model, param, value)
            
            # Train model
            metrics = model.train(X_train, y_train)
            
            training_time = time.perf_counter() - start_time
            
            # Store metrics
            self._model_metrics[request.model_type] = {
                "mae": getattr(metrics, 'mae', 0),
                "rmse": getattr(metrics, 'rmse', 0),
                "r2": getattr(metrics, 'r2', 0),
                "trained_at": datetime.utcnow().isoformat(),
                "training_samples": len(X_train)
            }
            
            self._model_status[request.model_type] = ModelStatus.TRAINED
            
            result = TrainingResult(
                model_type=request.model_type,
                success=True,
                metrics={
                    "mae": getattr(metrics, 'mae', 0),
                    "rmse": getattr(metrics, 'rmse', 0),
                    "r2": getattr(metrics, 'r2', 0),
                    "cross_val_score": getattr(metrics, 'cross_val_score', 0)
                },
                training_time_seconds=training_time,
                samples_used=len(X_train),
                validation_samples=len(X_val)
            )
            
            return ServiceResult.success(result)
            
        except Exception as e:
            self._logger.error(f"Training failed: {e}")
            return ServiceResult.error(f"Training failed: {str(e)}")
    
    def _prepare_training_data(self, data: List[Dict]) -> Tuple['np.ndarray', 'np.ndarray']:
        """Prepare training data"""
        import numpy as np
        
        features = []
        targets = []
        
        for item in data:
            if "features" in item and "target" in item:
                features.append(item["features"])
                targets.append(item["target"])
            else:
                # Auto-extract features and target
                feat = []
                target = 0
                for k, v in item.items():
                    if k == "target" or k == "y":
                        target = v
                    elif isinstance(v, (int, float)):
                        feat.append(v)
                features.append(feat)
                targets.append(target)
        
        return np.array(features), np.array(targets)
    
    # =========================================================================
    # OPTIMIZATION SERVICES
    # =========================================================================
    
    @measure_time
    def optimize(self, model_type: str, objective: str,
                constraints: Dict[str, Any]) -> ServiceResult[Dict[str, Any]]:
        """
        Run optimization using specified model.
        
        Args:
            model_type: Model to use
            objective: Objective function type
            constraints: Optimization constraints
            
        Returns:
            ServiceResult containing optimization results
        """
        if not AI_AVAILABLE:
            return self._mock_optimize(model_type, objective, constraints)
        
        model = self._models.get(model_type)
        if not model:
            return ServiceResult.not_found(f"Model {model_type} not found")
        
        return self._execute(self._optimize_impl, model, objective, constraints)
    
    def _optimize_impl(self, model: Any, objective: str,
                      constraints: Dict) -> ServiceResult[Dict[str, Any]]:
        """Implementation of optimize"""
        import numpy as np
        
        # Define objective function based on type
        if objective == "minimize_cost":
            def obj_func(x):
                return np.sum(x ** 2)  # Simple cost function
        elif objective == "maximize_reduction":
            def obj_func(x):
                return -np.sum(x)  # Maximize (negative for minimization)
        elif objective == "balance":
            def obj_func(x):
                return np.var(x) + 0.1 * np.sum(x ** 2)  # Balance + cost
        else:
            def obj_func(x):
                return np.sum(x ** 2)
        
        try:
            result = model.optimize(obj_func, constraints)
            
            optimization_result = {
                "optimal_values": result.optimal_values if hasattr(result, 'optimal_values') else {},
                "objective_value": result.objective_value if hasattr(result, 'objective_value') else 0,
                "convergence_status": result.convergence_status if hasattr(result, 'convergence_status') else "unknown",
                "iterations": result.iterations if hasattr(result, 'iterations') else 0,
                "execution_time": result.execution_time if hasattr(result, 'execution_time') else 0,
                "confidence_score": result.confidence_score if hasattr(result, 'confidence_score') else 0
            }
            
            return ServiceResult.success(optimization_result)
            
        except Exception as e:
            self._logger.error(f"Optimization failed: {e}")
            return ServiceResult.error(f"Optimization failed: {str(e)}")
    
    def _mock_optimize(self, model_type: str, objective: str,
                      constraints: Dict) -> ServiceResult[Dict[str, Any]]:
        """Mock optimization result"""
        return ServiceResult.success({
            "optimal_values": {"param_0": 0.5, "param_1": 0.3},
            "objective_value": 0.1,
            "convergence_status": "converged",
            "iterations": 100,
            "execution_time": 0.5,
            "confidence_score": 85.0,
            "mode": "mock"
        })
    
    # =========================================================================
    # SCENARIO GENERATION
    # =========================================================================
    
    @measure_time
    def generate_optimal_scenario(self, baseline_data: Dict[str, Any],
                                 target_config: Dict[str, Any]) -> ServiceResult[Dict[str, Any]]:
        """
        Generate AI-optimized scenario.
        
        Args:
            baseline_data: Baseline emission data
            target_config: Target configuration
            
        Returns:
            ServiceResult containing optimized scenario
        """
        return self._execute(
            self._generate_scenario_impl,
            baseline_data, target_config
        )
    
    def _generate_scenario_impl(self, baseline_data: Dict,
                               target_config: Dict) -> ServiceResult[Dict[str, Any]]:
        """Implementation of generate_optimal_scenario"""
        import numpy as np
        
        # Extract parameters
        baseline = baseline_data.get("total_emission", 10000)
        base_year = target_config.get("base_year", 2024)
        target_year = target_config.get("target_year", 2050)
        
        # Use AI if available
        if AI_AVAILABLE and "milestone_optimizer" in self._models:
            model = self._models["milestone_optimizer"]
            
            # Prepare features
            features = np.array([[baseline, base_year, target_year]])
            
            try:
                if not model.is_trained:
                    self._auto_train_model("milestone_optimizer", model)
                
                predictions = model.predict(features)
                optimal_reduction = float(predictions[0]) if hasattr(predictions, '__iter__') else 40.0
            except Exception:
                optimal_reduction = 40.0
        else:
            optimal_reduction = 40.0
        
        # Generate scenario
        scenario = {
            "scenario_id": f"AI-SCN-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "scenario_type": "AI_OPTIMIZED",
            "baseline_emission": baseline,
            "base_year": base_year,
            "target_year": target_year,
            "optimal_reduction_2030": min(50, max(30, optimal_reduction)),
            "optimal_reduction_2050": min(95, max(80, optimal_reduction * 2.2)),
            "strategy_allocation": {
                "energy_efficiency": 30.0,
                "renewable_energy": 35.0,
                "electrification": 20.0,
                "operational_optimization": 15.0
            },
            "confidence_score": 85.0 if AI_AVAILABLE else 70.0,
            "generated_at": datetime.utcnow().isoformat(),
            "ai_powered": AI_AVAILABLE
        }
        
        return ServiceResult.success(scenario)
    
    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================
    
    @measure_time
    def get_feature_importance(self, model_type: str) -> ServiceResult[Dict[str, float]]:
        """
        Get feature importance from a trained model.
        
        Args:
            model_type: Model type
            
        Returns:
            ServiceResult containing feature importance
        """
        if not AI_AVAILABLE:
            return ServiceResult.success({
                "baseline_emission": 0.4,
                "area_sqm": 0.3,
                "carbon_intensity": 0.2,
                "retrofit_potential": 0.1
            })
        
        model = self._models.get(model_type)
        if not model:
            return ServiceResult.not_found(f"Model {model_type} not found")
        
        if hasattr(model, 'feature_importance') and model.feature_importance:
            return ServiceResult.success(model.feature_importance)
        
        # Default feature names
        feature_names = ["baseline_emission", "area_sqm", "carbon_intensity"]
        
        # Try to get from underlying model
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
            result = {
                name: float(imp) 
                for name, imp in zip(feature_names, importances)
            }
            return ServiceResult.success(result)
        
        return ServiceResult.success({name: 1.0/len(feature_names) for name in feature_names})


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'AIService',
    'ModelType',
    'ModelStatus',
    'ModelInfo',
    'PredictionRequest',
    'PredictionResult',
    'TrainingRequest',
    'TrainingResult'
]
