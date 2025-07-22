import joblib
import os
from pathlib import Path

# Get the current directory where this script is located
CURRENT_DIR = Path(__file__).resolve().parent
# Go up one level to the app directory
APP_DIR = CURRENT_DIR.parent
# Go up one more level to the project root
PROJECT_ROOT = APP_DIR.parent

# Define paths relative to the project root
MODEL_PATH = os.path.join(PROJECT_ROOT, 'app', 'models', 'power_consumption_model.pkl')
OPTIMIZED_PATH = os.path.join(PROJECT_ROOT, 'app', 'models', 'optimized_model.pkl')

def optimize_model():
    print("Loading original model...")
    print(f"Looking for model at: {MODEL_PATH}")
    
    try:
        model = joblib.load(MODEL_PATH)
        print("Original model type:", type(model).__name__)
        
        if hasattr(model, 'n_estimators'):
            print(f"Reducing n_estimators from {model.n_estimators} to 50")
            model.n_estimators = 50
            if hasattr(model, 'max_depth') and model.max_depth is not None:
                print(f"Reducing max_depth from {model.max_depth} to 10")
                model.max_depth = min(10, model.max_depth)
        
        elif hasattr(model, 'tree_'):
            print("Reducing max_depth to 10 for decision tree")
            model.max_depth = 10
        
        print(f"Saving optimized model to {OPTIMIZED_PATH}")
        joblib.dump(model, OPTIMIZED_PATH, compress=3)
        print("Optimization complete!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        raise

if __name__ == "__main__":
    optimize_model()