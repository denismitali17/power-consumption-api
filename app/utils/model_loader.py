import joblib
import os
from typing import Tuple
import numpy as np

class ModelLoader:
    _instance = None
    _model = None
    _scaler = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._load_models()
        return cls._instance

    def _load_models(self):
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_path = os.path.join(base_dir, 'models', 'power_consumption_model.pkl')
        scaler_path =    os.path.join(base_dir, 'models', 'scaler.pkl')
        
        
        print(f"Looking for model at: {model_path}")
        print(f"Looking for scaler at: {scaler_path}")
        
        try:
            self._model = joblib.load(model_path)
            self._scaler = joblib.load(scaler_path)
            print("Successfully loaded model and scaler!")
        except Exception as e:
            raise RuntimeError(f"Error loading models: {str(e)}")

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._model

    @property
    def scaler(self):
        if self._scaler is None:
            raise RuntimeError("Scaler not loaded")
        return self._scaler