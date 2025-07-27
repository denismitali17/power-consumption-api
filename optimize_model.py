# optimize_model.py
import joblib
from sklearn.ensemble import RandomForestRegressor

def optimize_model():
    # Load the original model
    model = joblib.load('app/models/power_consumption_model.joblib')
    
    # Reduce the number of trees
    if hasattr(model, 'n_estimators'):
        model.n_estimators = min(50, model.n_estimators)  # Reduce to 50 trees max
    
    # Save the optimized model
    joblib.dump(model, 'app/models/optimized_model.joblib', compress=9)
    print("Optimized model saved successfully!")

if __name__ == "__main__":
    optimize_model()