import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from project_name.data.preprocessing import preprocess_hbsc_data


# Load data
base_dir = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base_dir, "data", "HBSC2018.csv")

# Get the input and output
X_base_columns = [
    "bodyweight", "bodyheight", "nervous", "irritable", "lifesat",
    "breakfastwd", "health", "fruits_2", "headache", "fight12m",
    "friendcounton", "softdrinks_2", "dizzy", "sweets_2", "friendhelp"
]
emc_cols = [f"emcsocmed{i}" for i in range(1, 10)]
y_cols = ["thinkbody", "feellow", "sleepdificulty"]

# Preprocess
data = preprocess_hbsc_data(
    filepath=data_path,
    selected_columns=X_base_columns,
    emc_cols=emc_cols,
    y=y_cols
)

X_columns = X_base_columns + ["emcsocmed_sum"]
X = data[X_columns]
y = pd.to_numeric(data["thinkbody"], errors='raise')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Build neural network model
def build_model_manual(
    learning_rate=0.001, units=64, num_layers=2, optimizer='adam'
):
    inputs = tf.keras.Input(shape=(X_train.shape[1],))
    x = inputs
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)

    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model


# The hyperparameter settings with their varied values
varied_hyperparameters = {
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "units": [8, 16, 32, 64],
    "num_layers": [2, 3, 4],
    "optimizer": ["adam", "sgd"]
}

# Run the model with each hyperparameter individually while fixing all others
fixed_parameters = {
    "learning_rate": 0.001,
    "units": 64,
    "num_layers": 2,
    "optimizer": "adam"
}

# Go through each combination of fixed hyperparameter with varied options
for param_name, values in varied_hyperparameters.items():
    print(f"\nTuning {param_name}...")
    val_accuracies = []
    for value in values:
        params = fixed_parameters.copy()
        params[param_name] = value
        print(f"Training with {param_name} = {value}")
        model = build_model_manual(
            learning_rate=params["learning_rate"],
            units=params["units"],
            num_layers=params["num_layers"],
            optimizer=params["optimizer"]
        )
        history = model.fit(
            X_train_scaled, y_train - 1,
            validation_split=0.2,
            epochs=5,
            batch_size=32,
            verbose=0
        )
        val_acc = history.history['val_sparse_categorical_accuracy'][-1]
        val_accuracies.append(val_acc)

    # Plot result for each hyperparameter
    plt.figure()
    plt.plot(values, val_accuracies, marker='o')
    if param_name == "learning_rate":
        plt.xscale('log')
    plt.xlabel(param_name.replace("_", " ").title())  # replace the _
    plt.ylabel("Validation Accuracy")
    plt.title(f"Tuning {param_name}")
    plt.grid(True)
    plt.tight_layout()

    # Save plot to "Plots manual tuning" folder
    plot_dir = os.path.join(os.path.dirname(__file__), "Plots manual tuning")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"tuning_{param_name}.png")
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
