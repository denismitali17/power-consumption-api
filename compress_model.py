import joblib

def compress_model():
    print("Loading model...")
    model = joblib.load('app/models/power_consumption_model.pkl')
    print("Model loaded. Compressing...")
    joblib.dump(model, 'app/models/power_consumption_model_compressed.joblib', compress=3)
    print("Compression complete! New file saved as 'power_consumption_model_compressed.joblib'")

if __name__ == "__main__":
    compress_model()