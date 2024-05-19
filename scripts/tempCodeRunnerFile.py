# Retrain the Decision Tree with reduced features
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