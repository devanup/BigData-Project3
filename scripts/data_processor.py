import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DataProcessor:
    @staticmethod
    def load_data(file_name):
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        data = pd.read_csv(file_path)
        return data

    @staticmethod
    def clean_data(data):
        # Remove rows with any NaN values
        cleaned_data = data.dropna()
        return cleaned_data

    @staticmethod
    def encode_labels(data):
        # Encode the 'diagnosis' column to numerical values
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        return data

    @staticmethod
    def split_data(data, test_size=0.2, random_state=42):
        # Assuming the label column is the 'diagnosis' column
        X = data.drop(columns=['diagnosis'])  # Features
        y = data['diagnosis']  # Label
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        # X_test is the test data, y_test is the test labels, X_train is the training data, y_train is the training labels
        return X_train, X_test, y_train, y_test
