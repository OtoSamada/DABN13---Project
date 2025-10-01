# Contents of /hotel-booking-cancellation/hotel-booking-cancellation/scripts/run_inference.py

import pandas as pd
import joblib
import os
from src.utils.config import load_config
from src.models.registry import ModelRegistry

def load_data(data_path):
    """Load the processed data for inference."""
    return pd.read_csv(data_path)

def run_inference(model_path, data):
    """Run inference on the input data using the trained model."""
    model = joblib.load(model_path)
    predictions = model.predict(data)
    return predictions

def main():
    # Load configuration
    config = load_config('configs/experiment.yaml')

    # Load the data
    data_path = os.path.join(config['data']['processed'], 'processed_data.csv')
    data = load_data(data_path)

    # Load the model
    model_registry = ModelRegistry()
    model_path = model_registry.get_best_model_path()

    # Run inference
    predictions = run_inference(model_path, data)

    # Save predictions
    output_path = os.path.join(config['output']['predictions'], 'predictions.csv')
    pd.DataFrame(predictions, columns=['predictions']).to_csv(output_path, index=False)

if __name__ == "__main__":
    main()