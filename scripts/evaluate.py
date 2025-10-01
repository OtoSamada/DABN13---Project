# Contents of /hotel-booking-cancellation/hotel-booking-cancellation/scripts/evaluate.py

import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from src.utils.logging import setup_logging

def load_model(model_path):
    """Load the trained model from the specified path."""
    model = joblib.load(model_path)
    return model

def load_data(data_path):
    """Load the test data from the specified path."""
    data = pd.read_csv(data_path)
    return data

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print the classification report and confusion matrix."""
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def main():
    """Main function to evaluate the model."""
    setup_logging()
    
    model_path = '../models/trained/best_model.pkl'  # Adjust path as necessary
    test_data_path = '../data/processed/test_data.csv'  # Adjust path as necessary
    
    model = load_model(model_path)
    test_data = load_data(test_data_path)
    
    X_test = test_data.drop(columns=['is_cancelled'])  # Adjust based on your feature set
    y_test = test_data['is_cancelled']
    
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()