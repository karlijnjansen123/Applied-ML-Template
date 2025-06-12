import os
import pandas as pd
import joblib
import tensorflow as tf
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler
from focal_loss import SparseCategoricalFocalLoss
from project_name.data.preprocessing import preprocess_hbsc_data
from project_name.models.NeuralNetwork import test_train_split
import shutil


shutil.rmtree(
    os.path.join("grid_tuning", "multi_output_nn"),
    ignore_errors=True
)

base_dir = os.path.dirname(os.path.dirname(__file__))
filepath = os.path.join(base_dir, "data", "HBSC2018.csv")

X_base_columns = [
    "bodyweight", "bodyheight",
    "nervous", "irritable", "lifesat", "breakfastwd",
    "health", "fruits_2", "headache", "fight12m", "friendcounton",
    "softdrinks_2", "dizzy", "sweets_2", "friendhelp"
]

emc_cols = [
    "emcsocmed1", "emcsocmed2", "emcsocmed3",
    "emcsocmed4", "emcsocmed5", "emcsocmed6",
    "emcsocmed7", "emcsocmed8", "emcsocmed9"
]

y_cols = ["thinkbody", "feellow", "sleepdificulty"]

clean_data = preprocess_hbsc_data(
    filepath=filepath,
    selected_columns=X_base_columns,
    emc_cols=emc_cols,
    y=y_cols
)

X_columns = X_base_columns + ["emcsocmed_sum"]

X = clean_data[X_columns]
Y1 = clean_data["thinkbody"]
Y2 = clean_data["feellow"]
Y3 = clean_data["sleepdificulty"]

Y1 = pd.to_numeric(Y1, errors='raise')
Y2 = pd.to_numeric(Y2, errors='raise')
Y3 = pd.to_numeric(Y3, errors='raise')

(
    X_train, X_test,
    Y1_train, Y1_test,
    Y2_train, Y2_test,
    Y3_train, Y3_test
) = test_train_split(X, Y1, Y2, Y3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

deployment_dir = os.path.join(base_dir, "Deployment")
os.makedirs(deployment_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(deployment_dir, "scaler.pkl"))

input_dim = X_train_scaled.shape[1]


# Neural network model
def build_model(hp):

    # Input layer
    inp = tf.keras.Input(shape=(input_dim,))

    # Hidden layers
    x = inp
    for i in range(hp.Int("num_layers", 2, 3)):
        x = tf.keras.layers.Dense(
            units=hp.Choice(f"units_{i}", [16, 32, 64]),
            activation='relu',
            kernel_initializer='glorot_uniform'  # Explicit random init
        )(x)

    # Output layers
    out1 = tf.keras.layers.Dense(
        5, activation='softmax', name='think_body'
    )(x)
    out2 = tf.keras.layers.Dense(
        5, activation='softmax', name='feeling_low'
    )(x)
    out3 = tf.keras.layers.Dense(
        5, activation='softmax', name='sleep_difficulty'
    )(x)

    # Build model
    model = tf.keras.Model(inputs=inp, outputs=[out1, out2, out3])

    # Optimizer and learning rate choice
    learning_rate = hp.Choice("learning_rate", [1e-3, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss={
            'think_body': SparseCategoricalFocalLoss(gamma=2),
            'feeling_low': SparseCategoricalFocalLoss(gamma=2),
            'sleep_difficulty': SparseCategoricalFocalLoss(gamma=2)
        },
        metrics={
            'think_body': tf.keras.metrics.SparseCategoricalAccuracy(),
            'feeling_low': tf.keras.metrics.SparseCategoricalAccuracy(),
            'sleep_difficulty': tf.keras.metrics.SparseCategoricalAccuracy()
        }
    )

    return model


# GridSearch Tuner from Keras
tuner = kt.GridSearch(
    hypermodel=build_model,
    objective=kt.Objective(
        "val_think_body_sparse_categorical_accuracy", direction="max"
    ),
    max_trials=None,  # Full grid
    executions_per_trial=1,
    directory="grid_tuning",
    project_name="multi_output_nn"
)

# Run search
tuner.search(
    X_train_scaled,
    {
        "think_body": Y1_train - 1,
        "feeling_low": Y2_train - 1,
        "sleep_difficulty": Y3_train - 1,
    },
    validation_split=0.2,
    epochs=2,
    batch_size=32,
    verbose=1
)

# Report the top hyperparameters
top_n = 5
print(f"\nTop {top_n} hyperparameter configurations:")
for i, hp in enumerate(tuner.get_best_hyperparameters(top_n)):
    print(f"\nRank {i + 1}")
    for k, v in hp.values.items():
        print(f"  {k}: {v}")
