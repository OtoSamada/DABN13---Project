# Contents of /hotel-booking-cancellation/hotel-booking-cancellation/scripts/run_training.py

import os
import mlflow
from src.pipelines.training_pipeline import TrainingPipeline
from src.utils.config import load_config

def main():
    # Load configuration
    config = load_config('configs/experiment.yaml')
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.start_run()

    try:
        # Initialize and run the training pipeline
        pipeline = TrainingPipeline(config)
        pipeline.run()
        
        # Log model and parameters to MLflow
        mlflow.log_param("model_type", config['model']['type'])
        mlflow.log_param("hyperparameters", config['model']['hyperparameters'])
        mlflow.log_artifact(os.path.join('models', 'trained', 'best_model.pkl'))
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()