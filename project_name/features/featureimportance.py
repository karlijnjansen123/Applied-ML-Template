import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def KNN_shap_graphs(X_train, X_test, predict_proba, y_name, num_background=1000, num_explain=10, column_names=None):

    if column_names is None:
        column_names = [f'Feature {i}' for i in range(X_train.shape[1])]

    # Convert X_train and X_test to pandas DataFrame with column_names
    X_train_df = pd.DataFrame(X_train, columns=column_names)
    X_test_df = pd.DataFrame(X_test, columns=column_names)

    # Use a subset of background data
    background = shap.sample(X_train_df, num_background, random_state=0)

    # Use Explainer
    explainer = shap.Explainer(predict_proba, background)
    explainer.feature_names = column_names

    # Limit explain set
    X_explain = X_test_df[:num_explain]
    shap_values = explainer(X_explain)

    print("SHAP values shape:", shap_values.values.shape)  # Should be (num_explain, num_features, num_classes)
    print("X_explain shape:", X_explain.shape)

    # Plot for class 1 (index -1 means "last class")
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title(str(y_name))
    shap.waterfall_plot(shap_values[0, :, 1], show=False, max_display=20, ax=ax)
    plt.tight_layout()
    plt.show()
    return shap_values

def NN_shap_graphs(model, X_background, X_sample, column_names, class_index=0, max_display=10):
    explainer = shap.DeepExplainer(model, X_background)
    base_value = explainer.expected_value[class_index]

    shap.waterfall_plot(expected_value=base_value, feature_names=column_names, features=X_sample[0], max_display=max_display)
    plt.show()
