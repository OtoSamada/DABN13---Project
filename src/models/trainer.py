# File: /hotel-booking-cancellation/hotel-booking-cancellation/src/models/trainer.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from src.utils.logging import setup_logging

class ModelTrainer:
    def __init__(self, model=None):
        self.model = model if model else RandomForestClassifier()
        self.logger = setup_logging()

    def load_data(self, data_path):
        self.logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        return data

    def preprocess_data(self, data):
        self.logger.info("Preprocessing data")
        # Implement preprocessing steps here
        # For example: handle missing values, encode categorical variables, etc.
        return data

    def train(self, data_path):
        data = self.load_data(data_path)
        processed_data = self.preprocess_data(data)

        X = processed_data.drop('is_cancelled', axis=1)
        y = processed_data['is_cancelled']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.logger.info("Training the model")
        self.model.fit(X_train, y_train)

        self.logger.info("Evaluating the model")
        predictions = self.model.predict(X_test)
        report = classification_report(y_test, predictions)
        self.logger.info(f"Classification Report:\n{report}")

        return report

    def save_model(self, model_path):
        self.logger.info(f"Saving model to {model_path}")
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        self.logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, X):
        self.logger.info("Making predictions")
        return self.model.predict(X)