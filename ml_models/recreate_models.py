"""
Recreate baseline models with proper module references
Run this from project root: python ml_models/recreate_models.py
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from the module (not __main__)
from ml_models.baseline_models import (
    BaselineMilestonePredictor,
    BaselineAllocationOptimizer,
    BaselineActionPlanner,
    BaselinePerformancePredictor
)

def recreate_models():
    """Recreate all baseline models with correct module references"""
    print("=" * 80)
    print("RECREATING BASELINE ML MODELS WITH CORRECT MODULE REFERENCES")
    print("=" * 80)
    
    checkpoint_dir = "ml_models/checkpoints/baseline/"
    
    # Create models
    print("\nCreating models...")
    milestone_model = BaselineMilestonePredictor()
    allocation_model = BaselineAllocationOptimizer()
    action_model = BaselineActionPlanner()
    performance_model = BaselinePerformancePredictor()
    print("✅ Models created")
    
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
    print("🎉 MODELS RECREATED WITH CORRECT MODULE REFERENCES!")
    print("=" * 80)
    print("\nNow restart your server and AI should load successfully!")

if __name__ == "__main__":
    recreate_models()