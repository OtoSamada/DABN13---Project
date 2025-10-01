# Hotel Booking Cancellation Prediction

This project aims to build a machine learning model to predict hotel booking cancellations using a dataset of hotel bookings. The project is structured to facilitate data processing, model training, evaluation, and reporting.

## Project Structure

- **data/**: Contains subdirectories for different stages of data processing.
  - **raw/**: Original, unprocessed data files.
  - **interim/**: Partially processed data that is not yet ready for analysis.
  - **processed/**: Final, cleaned, and processed data ready for modeling.

- **notebooks/**: Jupyter notebooks for exploration and production.
  - **exploration/**: Notebooks for exploratory data analysis.
    - **01_exploratory_analysis.ipynb**: Performs exploratory data analysis on the hotel booking dataset.
  - **production/**: Notebooks for production-level tasks.
    - **ml_py-test.ipynb**: Main code for predicting hotel booking cancellations.

- **src/**: Source code for the project, organized into modules.
  - **pipelines/**: Code related to data processing and model training pipelines.
    - **training_pipeline.py**: Implementation of the training pipeline.
  - **features/**: Code for feature engineering and preprocessing.
    - **preprocessors.py**: Functions and classes for preprocessing the data.
  - **models/**: Code related to model training and management.
    - **registry.py**: Manages model registration and versioning.
    - **trainer.py**: Implementation of model training logic.
  - **utils/**: Utility functions and classes.
    - **config.py**: Handles configuration settings for the project.
    - **logging.py**: Logging setup and utility functions.

- **configs/**: Configuration files for experiments.
  - **experiment.yaml**: Configuration settings for experiments, such as hyperparameters and model settings.

- **reports/**: Generated reports from the analysis.
  - **figures/**: Visualizations generated during the analysis.
    - **cancellations_by_month.png**: Image showing the number of cancellations by month.
  - **tables/**: Summary tables generated from the analysis.
    - **performance_summary.csv**: CSV file summarizing the performance metrics of the models.

- **models/**: Trained models and experiment tracking.
  - **trained/**: Final trained model files.
    - **best_model.pkl**: Serialized best-performing model.
  - **experiments/**: Files related to experiments, including MLflow tracking.
    - **mlflow/**: Folder containing MLflow artifacts for tracking experiments.

- **scripts/**: Scripts for running various tasks.
  - **run_training.py**: Executes the training process.
  - **run_inference.py**: Makes predictions using the trained model.
  - **evaluate.py**: Evaluates the performance of the model on test data.

- **tests/**: Unit tests for the project.
  - **test_training_pipeline.py**: Unit tests for the training pipeline functionality.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository_url>
   cd hotel-booking-cancellation
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up the environment variables if necessary (refer to `config.py` for details).

4. Run the exploratory analysis notebook to understand the dataset:
   ```
   jupyter notebook notebooks/exploration/01_exploratory_analysis.ipynb
   ```

5. Train the model using the training script:
   ```
   python scripts/run_training.py
   ```

6. Evaluate the model performance:
   ```
   python scripts/evaluate.py
   ```

## Usage Guidelines

- Use the `notebooks/` directory for exploratory analysis and prototyping.
- Implement data processing and model training in the `src/` directory.
- Store trained models and experiment tracking in the `models/` directory.
- Use the `reports/` directory for storing generated figures and tables.
- Write unit tests in the `tests/` directory to ensure code quality.

## Experiment Tracking

This project utilizes MLflow for experiment tracking. Ensure that MLflow is installed and configured properly to log experiments and track model performance.

## Conclusion

This project provides a structured approach to building a machine learning model for predicting hotel booking cancellations. Follow the setup instructions and usage guidelines to contribute to the project effectively.