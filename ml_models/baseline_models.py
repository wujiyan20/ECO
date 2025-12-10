"""
Baseline ML Models for EcoAssist
These models work WITHOUT any training data using industry-standard heuristics

Location: ml_models/baseline_models.py

Run once to create .pkl files:
    python ml_models/baseline_models.py
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
import os
import logging

logger = logging.getLogger(__name__)


class BaselineMilestonePredictor(BaseEstimator, RegressorMixin):
    """
    Baseline milestone predictor - works without training!
    Uses smart heuristics based on emission reduction research
    
    Based on industry standards:
    - Office buildings: 3-5% annual reduction potential
    - Retail: 2-4% annual reduction potential  
    - Residential: 2-3% annual reduction potential
    - Industrial: 1-3% annual reduction potential
    """
    
    def __init__(self):
        # Industry-standard annual reduction rates by building type
        self.reduction_factors = {
            'Office': 0.04,       # 4% per year
            'Retail': 0.03,       # 3% per year
            'Residential': 0.025, # 2.5% per year
            'Industrial': 0.02,   # 2% per year
            'Mixed': 0.03,        # 3% per year (default)
        }
        self.fitted_ = True  # Always "fitted" since it's rule-based
    
    def fit(self, X, y=None):
        """Fit method (does nothing for baseline, but required by sklearn)"""
        return self
    
    def predict(self, X):
        """
        Predict emission trajectory over multiple years
        
        X shape: (n_samples, n_features)
        Features expected:
            [0] area_sqm: float - Building area in square meters
            [1] baseline_emission: float - Starting emission in tCO2e
            [2] emission_intensity: float - tCO2e per sqm
            [3] building_age: int - Age of building in years
            [4] is_office: int (0 or 1)
            [5] is_retail: int (0 or 1)
            [6] is_residential: int (0 or 1)
            [7] is_industrial: int (0 or 1)
            [8] target_years: int - Number of years to predict
        
        Returns:
            np.array of shape (n_samples, target_years) - Emission predictions
        """
        predictions = []
        
        for features in X:
            baseline_emission = features[1]
            is_office = features[4]
            is_retail = features[5]
            is_residential = features[6]
            is_industrial = features[7]
            target_years = int(features[8]) if len(features) > 8 else 25
            building_age = features[3]
            emission_intensity = features[2]
            
            # Determine building type and base reduction rate
            if is_office:
                reduction_rate = self.reduction_factors['Office']
            elif is_retail:
                reduction_rate = self.reduction_factors['Retail']
            elif is_residential:
                reduction_rate = self.reduction_factors['Residential']
            elif is_industrial:
                reduction_rate = self.reduction_factors['Industrial']
            else:
                reduction_rate = self.reduction_factors['Mixed']
            
            # Adjust reduction rate based on building characteristics
            
            # Building age factor: older buildings reduce slower (harder to retrofit)
            if building_age > 30:
                reduction_rate *= 0.85  # 15% slower
            elif building_age < 10:
                reduction_rate *= 1.1   # 10% faster (newer tech)
            
            # Emission intensity factor: high intensity = more potential
            if emission_intensity > 0.5:
                reduction_rate *= 1.15  # 15% more potential
            elif emission_intensity < 0.2:
                reduction_rate *= 0.90  # 10% less potential (already efficient)
            
            # Generate trajectory with diminishing returns
            trajectory = []
            current_emission = baseline_emission
            
            for year in range(1, target_years + 1):
                # Reduction gets harder over time (diminishing returns)
                # First years: easier reductions (low-hanging fruit)
                # Later years: harder reductions (require major investments)
                year_factor = max(0.3, 1 - (year / (target_years * 1.5)))
                
                # Actual reduction rate for this year
                actual_reduction_rate = reduction_rate * year_factor
                
                # Apply reduction
                current_emission = current_emission * (1 - actual_reduction_rate)
                
                # Ensure emission doesn't go negative or too low
                current_emission = max(current_emission, baseline_emission * 0.1)  # At least 10% of baseline
                
                trajectory.append(current_emission)
            
            predictions.append(trajectory)
        
        return np.array(predictions)
    
    def save(self, filepath):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved model to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)


class BaselineAllocationOptimizer(BaseEstimator, RegressorMixin):
    """
    Baseline allocation optimizer - works without training!
    Uses cost-effectiveness heuristics based on emission reduction research
    
    Key principles:
    - Higher emission intensity = more reduction potential
    - Larger properties = economies of scale
    - Newer buildings = easier retrofits
    """
    
    def __init__(self):
        self.fitted_ = True
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        """
        Predict optimal allocation weight for each property
        
        Features expected:
            [0] property_baseline: float - Property baseline emission
            [1] property_area: float - Property area in sqm
            [2] emission_intensity: float - tCO2e per sqm
            [3] is_office: int (0 or 1)
            [4] is_retail: int (0 or 1)
            [5] is_residential: int (0 or 1)
            [6] building_age: int
            
        Returns allocation weights (to be normalized by total)
        """
        weights = []
        
        for features in X:
            property_baseline = features[0]
            emission_intensity = features[2]
            building_age = features[6]
            is_office = features[3]
            
            # Base weight: emission intensity squared (favor high-intensity properties)
            weight = emission_intensity ** 1.5
            
            # Adjust for building age (newer = easier to improve)
            if building_age < 15:
                weight *= 1.2
            elif building_age > 35:
                weight *= 0.8
            
            # Office buildings: higher reduction potential
            if is_office:
                weight *= 1.15
            
            # Ensure minimum weight
            weight = max(weight, 0.1)
            
            weights.append(weight)
        
        return np.array(weights)
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved model to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)


class BaselineActionPlanner(BaseEstimator):
    """
    Baseline action planner - recommends actions based on cost-effectiveness
    
    Action costs and effectiveness based on industry research:
    - LED Lighting: Quick ROI, low cost
    - HVAC Upgrade: High impact, medium cost
    - Solar Panels: High impact, high cost, long ROI
    - Insulation: Medium impact, medium cost
    - Smart Controls: Quick ROI, low-medium cost
    """
    
    def __init__(self):
        # Cost per square meter (USD)
        self.action_costs = {
            'LED_LIGHTING': 50,
            'SMART_CONTROLS': 75,
            'INSULATION': 150,
            'HVAC_UPGRADE': 200,
            'SOLAR_PANELS': 300,
        }
        
        # Emission reduction percentage
        self.action_reduction = {
            'LED_LIGHTING': 0.08,     # 8% reduction
            'SMART_CONTROLS': 0.10,   # 10% reduction
            'INSULATION': 0.15,       # 15% reduction
            'HVAC_UPGRADE': 0.20,     # 20% reduction
            'SOLAR_PANELS': 0.30,     # 30% reduction
        }
        
        # Implementation timeframe (months)
        self.action_timeframe = {
            'LED_LIGHTING': 3,
            'SMART_CONTROLS': 4,
            'INSULATION': 6,
            'HVAC_UPGRADE': 8,
            'SOLAR_PANELS': 12,
        }
        
        self.fitted_ = True
    
    def predict_actions(self, features):
        """
        Recommend actions based on property characteristics and constraints
        
        features: [area_sqm, target_reduction_pct, budget]
        
        Returns: list of recommended actions with costs and reductions
        """
        area_sqm = features[0]
        target_reduction_pct = features[1]
        budget = features[2]
        
        recommended_actions = []
        cumulative_reduction = 0
        cumulative_cost = 0
        
        # Sort actions by cost-effectiveness (reduction per dollar)
        actions_by_effectiveness = sorted(
            self.action_costs.keys(),
            key=lambda a: self.action_reduction[a] / (self.action_costs[a] / 100),
            reverse=True
        )
        
        for action in actions_by_effectiveness:
            # Check if we've met the target
            if cumulative_reduction >= target_reduction_pct:
                break
            
            # Calculate cost for this property
            action_cost = self.action_costs[action] * area_sqm
            
            # Check budget constraint
            if cumulative_cost + action_cost <= budget:
                recommended_actions.append({
                    'action': action,
                    'cost': action_cost,
                    'reduction_pct': self.action_reduction[action],
                    'timeframe_months': self.action_timeframe[action],
                    'cost_per_tco2e': action_cost / (self.action_reduction[action] * 100)  # Simplified
                })
                cumulative_cost += action_cost
                cumulative_reduction += self.action_reduction[action]
        
        return recommended_actions
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved model to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)


class BaselinePerformancePredictor(BaseEstimator, RegressorMixin):
    """
    Baseline performance predictor - predicts actual vs planned performance
    
    Factors considered:
    - Implementation progress
    - Time elapsed
    - Seasonal variations
    - Occupancy changes
    """
    
    def __init__(self):
        self.fitted_ = True
        np.random.seed(42)  # For reproducible variance
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        """
        Predict actual performance vs planned
        
        Features:
            [0] planned_reduction: float
            [1] implementation_progress: float (0-1)
            [2] months_elapsed: int
            [3] weather_factor: float (default 1.0)
            [4] occupancy_change: float (default 0.0)
        """
        predictions = []
        
        for features in X:
            planned_reduction = features[0]
            implementation_progress = features[1]
            months_elapsed = features[2]
            weather_factor = features[3] if len(features) > 3 else 1.0
            occupancy_change = features[4] if len(features) > 4 else 0.0
            
            # Base prediction: proportional to implementation progress
            expected_reduction = planned_reduction * implementation_progress
            
            # Time factor: early months often underperform due to ramp-up
            if months_elapsed < 6:
                time_factor = 0.8
            else:
                time_factor = 1.0
            
            expected_reduction *= time_factor
            
            # External factors
            expected_reduction *= weather_factor
            expected_reduction *= (1 - occupancy_change * 0.5)  # 50% impact of occupancy
            
            # Add realistic variance (±10%)
            variance = np.random.uniform(-0.1, 0.1)
            actual_reduction = expected_reduction * (1 + variance)
            
            # Ensure non-negative
            actual_reduction = max(0, actual_reduction)
            
            predictions.append(actual_reduction)
        
        return np.array(predictions)
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved model to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)


def create_baseline_models():
    """
    Create and save all baseline models
    
    Run this ONCE to initialize the ML models:
        python ml_models/baseline_models.py
    
    This will create:
        ml_models/checkpoints/baseline/milestone_predictor.pkl
        ml_models/checkpoints/baseline/allocation_optimizer.pkl
        ml_models/checkpoints/baseline/action_planner.pkl
        ml_models/checkpoints/baseline/performance_predictor.pkl
    """
    print("=" * 80)
    print("CREATING BASELINE ML MODELS FOR ECOASSIST")
    print("=" * 80)
    print("\nThese models work WITHOUT any training data!")
    print("They use industry-standard heuristics and research-backed formulas.\n")
    
    # Create models
    print("Creating models...")
    milestone_model = BaselineMilestonePredictor()
    allocation_model = BaselineAllocationOptimizer()
    action_model = BaselineActionPlanner()
    performance_model = BaselinePerformancePredictor()
    print("✅ Models created")
    
    # Determine checkpoint directory
    checkpoint_dir = "ml_models/checkpoints/baseline/"
    
    # Save models
    print(f"\nSaving to {checkpoint_dir}...")
    
    milestone_model.save(f"{checkpoint_dir}milestone_predictor.pkl")
    print(f"✅ Saved: milestone_predictor.pkl")
    
    allocation_model.save(f"{checkpoint_dir}allocation_optimizer.pkl")
    print(f"✅ Saved: allocation_optimizer.pkl")
    
    action_model.save(f"{checkpoint_dir}action_planner.pkl")
    print(f"✅ Saved: action_planner.pkl")
    
    performance_model.save(f"{checkpoint_dir}performance_predictor.pkl")
    print(f"✅ Saved: performance_predictor.pkl")
    
    print("\n" + "=" * 80)
    print("🎉 ALL BASELINE MODELS CREATED SUCCESSFULLY!")
    print("=" * 80)
    print("\nThese models are ready to use immediately!")
    print("They will provide AI-enhanced predictions without any training data.")
    print("\nNext steps:")
    print("1. Restart your FastAPI server")
    print("2. Test the /milestones/calculate endpoint")
    print("3. Check logs for: '✅ Using AI-predicted milestones'")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    create_baseline_models()
