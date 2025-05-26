import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def KNN_shap_graphs(X_train, X_test, predict_proba, y_name, num_explain=10, column_names=None):

    if column_names is None:
        column_names = [f'Feature {i}' for i in range(X_train.shape[1])]

    # Convert X_train and X_test to pandas DataFrame with column_names
    X_train_df = pd.DataFrame(X_train, columns=column_names)
    X_test_df = pd.DataFrame(X_test, columns=column_names)

    # Use a subset of background data
    background = shap.sample(X_train_df, 1000, random_state=0) #use a background of 1000 samples

    # Use Explainer
    explainer = shap.Explainer(predict_proba, background)
    explainer.feature_names = column_names

    # Limit explain set
    X_explain = X_test_df[:num_explain]
    shap_values = explainer(X_explain)

    print("SHAP values shape:", shap_values.values.shape)  # Should be (num_explain, num_features, num_classes)
    print("X_explain shape:", X_explain.shape)

    # Plot for class 1 (index -1 means "last class")
    shap.waterfall_plot(shap_values[0, :, 1], show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.title(str(y_name))
    plt.tight_layout()
    plt.show()
    return shap_values


def NN_shap_graphs(model, X_train, column_names):
    # 1. Train your model
    # 2. Create background dataset for SHAP
    background = X_train[np.random.choice(X_train.shape[0], 1000, replace=False)]

    # 3. Define a wrapper function for multi-output model
    def model_predict(x):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        preds = model(x_tensor, training=False)
        return tf.concat(preds, axis=1).numpy()

    # 4. Initialize SHAP explainer
    explainer = shap.Explainer(model_predict, background)

    # 5. Compute SHAP values
    shap_values = explainer(X_train[:100])

    # 6. Plot
    #fig, axes = plt.subplots(1, 1, figsize=(12, 18)) 
    shap.summary_plot(shap_values, X_train[:100], feature_names=column_names, show=False)
    plt.gca().legend_.remove() 

    plt.tight_layout()
    plt.show()
