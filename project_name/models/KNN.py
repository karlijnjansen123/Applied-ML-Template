import shap
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def KNN_solver(X, y):
    Y_class = (y >= 2).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_class, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    predict_proba = knn.predict_proba

    return accuracy, X_train, X_test, predict_proba

def shap_graphs(X_train, X_test, predict_proba, num_background=1000, num_explain=10, column_names=None):

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
    shap.waterfall_plot(shap_values[0, :, 1])
    return shap_values






