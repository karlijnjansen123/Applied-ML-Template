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
    "bodyweight", "bodyheight", "nervous", "irritable", "lifesat", "breakfastwd",
    "health", "fruits_2", "headache", "fight12m", "friendcounton",
    "softdrinks_2", "dizzy", "sweets_2", "friendhelp"
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model with fixed values. Use to train with different learning rates.
def build_model_manual(learning_rate=0.001, units=64, num_layers=2, activation='relu'):
    inputs = tf.keras.Input(shape=(X_train.shape[1],))
    x = inputs
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(units, activation=activation)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model

# Manual learning rate tuning. Try out different values for the model with fixed other parameters
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
val_accuracies = []

for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    model = build_model_manual(learning_rate=lr)
    history = model.fit(
        X_train_scaled, y_train - 1,
        validation_split=0.2,
        epochs=5,
        batch_size=32,
        verbose=0
    )
    val_acc = history.history['val_sparse_categorical_accuracy'][-1]
    val_accuracies.append(val_acc)

# Plotting the result of learning rate versus validation accuracy
plt.plot(learning_rates, val_accuracies, marker='o')
plt.xscale('log')
plt.xlabel("Learning Rate")
plt.ylabel("Validation Accuracy")
plt.title("Learning Rate Tuning")
plt.grid(True)
plt.tight_layout()
plt.show()