from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
import os

# Create directory if it doesn't exist
os.makedirs('app/models', exist_ok=True)

# Create a very small model
model = RandomForestRegressor(
    n_estimators=5,  # Very few trees
    max_depth=3,     # Very shallow trees
    random_state=42
)

# Dummy training data (replace with your actual data loading)
X = np.random.rand(100, 8)  # 100 samples, 8 features
y = np.random.rand(100)     # Random target

# Train and save
print("Training minimal model...")
model.fit(X, y)
joblib.dump(model, 'app/models/minimal_model.joblib')
print("Minimal model saved to app/models/minimal_model.joblib")