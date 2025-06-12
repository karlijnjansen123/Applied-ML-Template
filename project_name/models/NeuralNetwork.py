import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from focal_loss import SparseCategoricalFocalLoss
import joblib
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import os
from datetime import datetime


def test_train_split(X, Y1, Y2, Y3):
    """
    Function to split feature and target data into training and testing sets

    :param X: Feature dataframe
    :param Y1: Output variable for body image
    :param Y2: Output variable for feeling low
    :param Y3: Output variable for sleep difficulty
    :return: Train-test splits for X, Y1, Y2 and Y3
    """
    # Splitting in test and train data
    (
        X_train, X_test,
        Y1_train, Y1_test,
        Y2_train, Y2_test,
        Y3_train, Y3_test
    ) = train_test_split(X, Y1, Y2, Y3, test_size=0.2)

    return (X_train, X_test,
            Y1_train, Y1_test,
            Y2_train, Y2_test,
            Y3_train, Y3_test)


def compute_class_weights(y_train):
    """
    Function that calculates class weights for the output labels
    :param y_train: train data for the y value (pandas)
    :return: weight vector
    """
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=classes, y=y_train
    )
    return weights


def build_neural_network(X_train, X_test, Y1_train, Y1_test,
                         Y2_train, Y2_test, Y3_train, Y3_test,
                         size_input):
    """

    :param X_train: train data set for X
    :param X_test: test data set for X
    :param Y1_train: train data set for Body image
    :param Y1_test: test data set for Body image
    :param Y2_train: train data set for Feeling Low
    :param Y2_test: test data set for Feeling Low
    :param Y3_train: train data set for Sleep Difficulties
    :param Y3_test: test data set for Sleep Difficulties
    :param size_input: amount of x-features
    :return:
    """
    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Save the scaler
    joblib.dump(scaler, "./project_name/Deployment/scaler.pkl")
    # Tensorboard implementation
    log_directory = os.path.join("logs", datetime.now().strftime(
        "%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_directory, histogram_freq=1)

    # Calculating class weights
    y1_weights = compute_class_weights(Y1_train)
    print(y1_weights, 'Weights for Body Image')
    y2_weights = compute_class_weights(Y2_train)
    print(y2_weights, 'Weights for Feeling Low')
    y3_weights = compute_class_weights(Y3_train)
    print(y3_weights, 'Weights for Sleep difficulties')

    # Model Architecture
    inp = tf.keras.Input(shape=(size_input,))
    hidden1 = tf.keras.layers.Dense(32, activation='relu')(inp)
    hidden2 = tf.keras.layers.Dense(32, activation='relu')(hidden1)
    hidden3 = tf.keras.layers.Dense(64, activation='relu')(hidden2)

    out1 = (tf.keras.layers.Dense(
        5, activation='softmax', name='think_body'
    )(hidden3))
    out2 = tf.keras.layers.Dense(
        5, activation='softmax', name='feeling_low'
    )(hidden3)
    out3 = tf.keras.layers.Dense(
        5, activation='softmax', name='sleep_difficulty'
    )(hidden3)

    model = tf.keras.Model(inputs=inp, outputs=[out1, out2, out3])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss={
            'think_body':
                SparseCategoricalFocalLoss(
                    gamma=1,
                    class_weight=y1_weights
                ),
            'feeling_low':
                SparseCategoricalFocalLoss(
                    gamma=1,
                    class_weight=y2_weights
                ),
            'sleep_difficulty':
                SparseCategoricalFocalLoss(
                    gamma=1,
                    class_weight=y3_weights
                )
        },
        metrics={
            'think_body': tf.keras.metrics.SparseCategoricalAccuracy(),
            'feeling_low': tf.keras.metrics.SparseCategoricalAccuracy(),
            'sleep_difficulty': tf.keras.metrics.SparseCategoricalAccuracy()
        }
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
    )

    # Train
    history = model.fit(
        X_train, {
            'think_body': Y1_train - 1,
            'feeling_low': Y2_train - 1,
            'sleep_difficulty': Y3_train - 1
        },
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, tensorboard_callback]
    )

    # Validation Metrics from History
    val_accuracy_thinkbody = history.history[
         'val_think_body_sparse_categorical_accuracy'][-1]
    val_accuracy_feelinglow = history.history[
        'val_feeling_low_sparse_categorical_accuracy'][-1]
    val_accuracy_sleepdiff = history.history[
        'val_sleep_difficulty_sparse_categorical_accuracy'][-1]

    # Evaluate F1 and AUC on Test Data
    Y_preds = model.predict(X_test)
    metrics_dict = {}

    for name, Y_true, Y_pred in zip(
            ['think_body', 'feeling_low', 'sleep_difficulty'],
            [Y1_test, Y2_test, Y3_test],
            Y_preds):

        Y_true_adjusted = Y_true - 1  # adjust to 0-indexed labels
        Y_pred_classes = np.argmax(Y_pred, axis=1)

        # F1 (macro)
        f1 = f1_score(Y_true_adjusted, Y_pred_classes, average='macro')
        # AUC (macro)
        try:
            auc = roc_auc_score(
                tf.keras.utils.to_categorical(Y_true_adjusted, num_classes=5),
                Y_pred,
                multi_class='ovr',
                average='macro'
            )
        except ValueError:
            auc = np.nan  # handle cases when a class is missing in test set

        metrics_dict[name] = {'f1_score': f1, 'auc_score': auc}

    # Save model
    model.save('project_name/Deployment/neural_network_model.keras')

    # Return model, scaler, and all metrics
    return (
        model, X_train, X_test, scaler,
        val_accuracy_thinkbody, val_accuracy_feelinglow,
        val_accuracy_sleepdiff,

        metrics_dict
    )
