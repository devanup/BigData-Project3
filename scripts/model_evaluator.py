import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    @staticmethod
    def train_decision_tree(X_train, y_train):
        start_time = time.time()
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        return model, training_time
    
    @staticmethod
    def train_svm_rbf(X_train, y_train):
        start_time = time.time()
        model = SVC(kernel='rbf')
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        return model, training_time
    
    @staticmethod
    def train_random_forest(X_train, y_train):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        return model, importances

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        sensitivity = recall_score(y_test, predictions, pos_label=1)  # Malignant
        specificity = recall_score(y_test, predictions, pos_label=0)  # Benign
        return accuracy, sensitivity, specificity, predictions

    @staticmethod
    def plot_decision_tree(model, feature_names):
        plt.figure(figsize=(20, 10))
        plot_tree(model, feature_names=feature_names, filled=True)
        plt.savefig('results/decision_tree.png')
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_test, predictions, model_name):
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_name} Confusion Matrix')
        plt.savefig(f'results/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.close()

    # @staticmethod
    # def plot_feature_importance(importances, feature_names):
    #     feature_importances = pd.Series(importances, index=feature_names)
    #     top_features = feature_importances.nlargest(2)
    #     plt.figure(figsize=(10, 6))
    #     sns.scatterplot(data=top_features, x=top_features.index, y=top_features.values)
    #     plt.title('Top 2 Feature Importances')
    #     plt.xlabel('Feature')
    #     plt.ylabel('Importance')
    #     plt.savefig('results/top_2_features.png')
    #     plt.close()
    @staticmethod
    def plot_feature_importance(importances, feature_names):
        feature_importances = pd.Series(importances, index=feature_names)
        plt.figure(figsize=(10, 6))
        feature_importances.nlargest(10).plot(kind='barh')
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.savefig('results/top_10_feature_importances.png')
        plt.close()
    
    @staticmethod
    def plot_top_two_features(data):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='radius_mean', y='texture_mean', hue='diagnosis', palette={0: 'blue', 1: 'red'})
        plt.title('Top 2 Features: Radius Mean vs. Texture Mean')
        plt.xlabel('Radius Mean')
        plt.ylabel('Texture Mean')
        plt.legend(title='Diagnosis', loc='upper right', labels=['Benign', 'Malignant'])
        plt.savefig('results/top_2_features_scatter.png')
        plt.close()