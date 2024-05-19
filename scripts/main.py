import pandas as pd
from data_processor import DataProcessor
from model_evaluator import ModelEvaluator

def main():
    # 1: Load the data
    csv_file = "../data/breast-cancer.csv"
    data = DataProcessor.load_data(csv_file)
    # print number of rows and columns
    print(data.shape) # (569, 32)


    #2:
    # Clean the data
    cleaned_data = DataProcessor.clean_data(data)
    print("Cleaned Data:")
    print(cleaned_data.head())
    
    # Encode the labels
    encoded_data = DataProcessor.encode_labels(cleaned_data)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = DataProcessor.split_data(encoded_data)

    # Function to train, evaluate, and plot models
    def train_evaluate_plot(model_func, model_name, X_train, X_test, y_train, y_test):
        model, training_time = model_func(X_train, y_train)
        print(f"\n{model_name} Training time: {training_time:.4f} seconds")
        accuracy, sensitivity, specificity, predictions = ModelEvaluator.evaluate_model(model, X_test, y_test)
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        print(f"{model_name} Sensitivity: {sensitivity:.4f}")
        print(f"{model_name} Specificity: {specificity:.4f}")
        ModelEvaluator.plot_confusion_matrix(y_test, predictions, model_name)
        return model

    # 3: Train, evaluate, and plot the Decision Tree
    model = train_evaluate_plot(ModelEvaluator.train_decision_tree, "Decision Tree", X_train, X_test, y_train, y_test)
    ModelEvaluator.plot_decision_tree(model, X_train.columns.tolist())

    # 4: Train, evaluate, and plot the SVM (RBF)
    train_evaluate_plot(ModelEvaluator.train_svm_rbf, "SVM (RBF)", X_train, X_test, y_train, y_test)

    # 6: Feature Importance and Iterative Feature Removal
    # Train Random Forest to get feature importances
    rf_model, importances = ModelEvaluator.train_random_forest(X_train, y_train)
    # Plot top 2 feature importances
    ModelEvaluator.plot_feature_importance(importances, X_train.columns)

    # Visualize the top two features
    top_two_data = cleaned_data[['radius_mean', 'texture_mean', 'diagnosis']]
    ModelEvaluator.plot_top_two_features(top_two_data)

    # Get feature importances sorted
    feature_importances = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

    # Remove 1, 4, and 10 least important features and retrain Decision Tree
    for num_features_to_remove in [1, 4, 10]:
        print(f"\nRemoving {num_features_to_remove} least important features and retraining...")
        features_to_remove = feature_importances.index[-num_features_to_remove:]
        X_train_reduced = X_train.drop(columns=features_to_remove)
        X_test_reduced = X_test.drop(columns=features_to_remove)
        # Retrain the Decision Tree with reduced features, Plot the Decision Tree, # Evaluate the model
        model = train_evaluate_plot(ModelEvaluator.train_decision_tree, "Decision Tree", X_train_reduced, X_test_reduced, y_train, y_test)
        ModelEvaluator.plot_decision_tree(model, X_train_reduced.columns.tolist())

        # # Retrain the Decision Tree with reduced features
        # model, training_time = ModelEvaluator.train_decision_tree(X_train_reduced, y_train)
        # print(f"Training time: {training_time:.4f} seconds")
        
        # # Plot the Decision Tree
        # feature_names_reduced = X_train_reduced.columns.tolist()
        # ModelEvaluator.plot_decision_tree(model, feature_names_reduced)
        
        # # Evaluate the model
        # accuracy, sensitivity, specificity, predictions = ModelEvaluator.evaluate_model(model, X_test_reduced, y_test)
        # print(f"Accuracy: {accuracy:.4f}")
        # print(f"Sensitivity: {sensitivity:.4f}")
        # print(f"Specificity: {specificity:.4f}")
        
        # # Visualize the Confusion Matrix
        # ModelEvaluator.plot_confusion_matrix(y_test, predictions)

if __name__ == "__main__":
    main()
