import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def KNN_shap_graphs(X_train, X_test, predict_proba, y_name, num_explain=10, column_names=None, plot=True):
    """
    Function that generates SHAP plots for a KNN classifier to explain feature contributions

    :param X_train: Training input features
    :param X_test: Test input features
    :param predict_proba: Callable that returns class probabilities
    :param y_name: Label or title for the current target being explained
    :param num_explain: Number fo test samples to explain
    :param column_names: List of feature names
    :return: SHAP values for the explained samples
    """

    if column_names is None:
        column_names = [f'Feature {i}' for i in range(X_train.shape[1])]

    # Convert X_train and X_test to pandas DataFrame with column_names
    X_train_df = pd.DataFrame(X_train, columns=column_names)
    X_test_df = pd.DataFrame(X_test, columns=column_names)

    # Use a subset of background data
    background = shap.sample(X_train_df, 1000, random_state=0)  # use a background of 1000 samples

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
    if plot:
        plt.show()

    return shap_values


def NN_shap_graphs(model, X_train, column_names, sample_size=50000, no_explained=100, plot=True):
    """
    Function to compute and visualize SHAP values for a trained NN model
    using a sampled background dataset and visualizes the feature
    importance with a summary plot

    :param model: Trained multi-class neural network model
    :param X_train: Trained input features
    :param column_names: List of feature names
    :return: None
    """

    # 1. Train your model
    # 2. Create background dataset for SHAP
    np.random.seed(42)  # Set seed for reproducibility
    background = X_train[np.random.choice(X_train.shape[0], sample_size, replace=False)]

    # 3. Define a wrapper function for multi-output model
    def model_predict(x):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        preds = model(x_tensor, training=False)
        return tf.concat(preds, axis=1).numpy()

    # 4. Initialize SHAP explainer
    explainer = shap.Explainer(model_predict, background)

    # 5. Compute SHAP values
    shap_values = explainer(X_train[:1000])

    # 6. Plot
    # fig, axes = plt.subplots(1, 1, figsize=(12, 18))
    shap.summary_plot(shap_values, X_train[:no_explained], feature_names=column_names, show=False)
    plt.gca().legend_.remove()

    plt.tight_layout()
    if plot:
        plt.show()


def averaged_NN_shap_graphs(
        build_model_fn,
        X_train, X_test,
        Y1_train, Y1_test,
        Y2_train, Y2_test,
        Y3_train, Y3_test,
        size_input, column_names, n_runs=5,
        sample_size=50000,
        no_explained=1000,
        plot = True
):
    """
    Function that trains the same neural network multiple times, computes
    SHAP values for each, averages them and plots a single SHAP summary plot

    :param build_model_fn: Function to build and train a fresh model
    :param X_train: Training features
    :param X_test: Test features
    :param Y1_train, Y2_train, Y3_train: Target labels for training
    :param Y1_test, Y2_test, Y3_test: Target labels for testing
    :param size_input: Number of input features
    :param column_names: List of feature names
    :param n_runs: Number of repeated training runs to average
    :return: None
    """

    np.random.seed(42)  # For reproducibility of background selection
    background = X_train[np.random.choice(X_train.shape[0], sample_size, replace=False)]

    all_shap_values = []

    for run in range(n_runs):
        print(f"Training model {run+1}/{n_runs}...")

        # 1. Train a fresh model using your build_model_fn()
        model, scaler = build_model_fn(X_train, X_test, Y1_train, Y1_test, Y2_train, Y2_test, Y3_train, Y3_test, size_input)

        # 2. Define model predict wrapper
        def model_predict(x):
            x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            preds = model(x_tensor, training=False)
            return tf.concat(preds, axis=1).numpy()

        # 3. Explain model using SHAP
        explainer = shap.Explainer(model_predict, background)
        shap_values = explainer(X_train[:no_explained])  # Or a fixed sample

        all_shap_values.append(shap_values)

    # 4. Average SHAP values
    print("Averaging SHAP values...")
    base_values_avg = np.mean([sv.base_values for sv in all_shap_values], axis=0)
    values_avg = np.mean([sv.values for sv in all_shap_values], axis=0)

    # 5. Create an averaged SHAP object for plotting
    avg_shap = shap.Explanation(
        values=values_avg,
        base_values=base_values_avg,
        data=X_train[:1000],
        feature_names=column_names
    )

    # 6. Plot summary
    shap.summary_plot(avg_shap, X_train[:100], feature_names=column_names, show=False)
    plt.gca().legend_.remove()
    plt.tight_layout()
    if plot:
        plt.show()


def averaged_NN_shap_graphs_per_output(
        build_model_fn,
        X_train, X_test,
        Y1_train, Y1_test,
        Y2_train, Y2_test,
        Y3_train, Y3_test,
        size_input, column_names, n_runs=5,
        sample_size=50000,
        no_explained=100,
        plot=True
):
    """
    Same as averaged_NN_shap_graphs, but generates a separate SHAP summary plot
    for each model output.

    """

    np.random.seed(42)  # For reproducibility of background selection
    background = X_train[np.random.choice(X_train.shape[0], sample_size, replace=False)]

    all_shap_values = []

    for run in range(n_runs):
        print(f"Training model {run+1}/{n_runs}...")

        # Train a fresh model using your build_model_fn()
        model, X_train, X_test, scaler, val_acc1, val_acc2, val_acc3, _ = build_model_fn(
            X_train, X_test,
            Y1_train, Y1_test,
            Y2_train, Y2_test,
            Y3_train, Y3_test,
            size_input
        )

        # Define model predict wrapper
        def model_predict(x):
            x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            preds = model(x_tensor, training=False)
            return tf.concat(preds, axis=1).numpy()

        # Explain model using SHAP
        explainer = shap.Explainer(model_predict, background)
        shap_values = explainer(X_train[:no_explained])  # fixed sample size

        all_shap_values.append(shap_values)

    # Average SHAP values and base values across runs
    print("Averaging SHAP values...")

    base_values_avg = np.mean([sv.base_values for sv in all_shap_values], axis=0)  # shape: (outputs,)
    values_avg = np.mean([sv.values for sv in all_shap_values], axis=0)            # shape: (samples, features, outputs)

    output_names = ["Y1", "Y2", "Y3"]

    for i, output_name in enumerate(output_names):
        print(f"Plotting SHAP summary for output: {output_name}")

        # Build SHAP Explanation object for output i
        expl = shap.Explanation(
            values=values_avg[:, :, i],              # values for samples x features for output i
            base_values=base_values_avg[i],          # base value for output i
            data=X_train[:no_explained],
            feature_names=column_names
        )

        # Plot SHAP summary plot per output
        shap.summary_plot(expl, X_train[:no_explained], feature_names=column_names, show=False)
        if plot:
            plt.show()