from datetime import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from focal_loss import SparseCategoricalFocalLoss
import joblib
import os


def test_train_split(X, Y1, Y2, Y3):
    """
    Function to split feature and target data into training and testing sets

    :param X: Feature dataframe
    :param Y1: Output variable for body image
    :param Y2: Output variable for feeling low
    :param Y3: Output variable for sleep difficulty
    :return: Train-test splits for X, Y1, Y2 and Y3
    """

    # Print statements to check imbalanced classes
    print(Y1.value_counts(normalize=True).round(3))
    print(Y2.value_counts(normalize=True).round(3))
    print(Y3.value_counts(normalize=True).round(3))

    # Splitting in test and train data
    (
        X_train, X_test,
        Y1_train, Y1_test,
        Y2_train, Y2_test,
        Y3_train, Y3_test
    ) = train_test_split(X, Y1, Y2, Y3, test_size=0.2)

    # Print statement for checking with and without using stratify
    print('Class distribution for the train set of Y3')
    print(Y1_train.value_counts(normalize=True).round(3))
    print('Class distribution for the test set of Y3')
    print(Y1_test.value_counts(normalize=True).round(3))

    return (X_train, X_test,
            Y1_train, Y1_test,
            Y2_train, Y2_test,
            Y3_train, Y3_test)


def build_neural_network(X_train, X_test, Y1_train, Y1_test,
                         Y2_train, Y2_test, Y3_train, Y3_test,
                         size_input):
    """
    Function to build, compile, train and save a multi-output neural network
    for predicting the 3 output variables.

    :param X_train: Scaled training input features
    :param X_test: Scaled test input features
    :param Y1_train, Y2_train, Y3_train: Training labels for body image, feeling low, and sleep difficulty
    :param Y1_test, Y2_test, Y3_test: Testing labels for body image, feeling low, and sleep difficulty
    :param size_input: Number of input features
    :return: Tuple containing the trained model, scaled train and test X,
    the scalar, and final validation accuracy for each output
    """

    # Create Tensorboard callback, create a log directory for every training, pass the tensor_board callback throught fit()
    log_directory = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_directory, histogram_freq = 1)

    # Normalisation of the x-features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the scaler in the Deployment folder to use for API
    joblib.dump(scaler, "./project_name/Deployment/scaler.pkl")

    # Input Layer; when model is loaded it expects a numpy array of same size
    inp = tf.keras.Input(shape=(size_input,))

    # Hidden Layers
    hidden1 = tf.keras.layers.Dense(128, activation='relu')(inp)
    hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)
    hidden3 = tf.keras.layers.Dense(32, activation='relu')(hidden2)
    hidden4 = tf.keras.layers.Dense(16, activation='relu')(hidden3)

    # Output Layers
    out1 = tf.keras.layers.Dense(
        5, activation='softmax', name='think_body')(hidden4)
    out2 = tf.keras.layers.Dense(
        5, activation='softmax', name='feeling_low')(hidden4)
    out3 = tf.keras.layers.Dense(
        5, activation='softmax', name='sleep_difficulty')(hidden4)

    # Model
    model = tf.keras.Model(inp, [out1, out2, out3])
    model.compile(
        optimizer='adam',
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

    history = model.fit(X_train, {
        'think_body': Y1_train - 1,  # Make it a 0-based index
        'feeling_low': Y2_train - 1,
        'sleep_difficulty': Y3_train - 1
    }, epochs=3, batch_size=32, validation_split=0.2,callbacks = [tensorboard_callback])

    # Extract the latest validation accuracies from history
    val_accuracy_thinkbody = history.history[
        'val_think_body_sparse_categorical_accuracy'][-1]
    val_accuracy_feelinglow = history.history[
        'val_feeling_low_sparse_categorical_accuracy'][-1]
    val_accuracy_sleepdiff = history.history[
        'val_sleep_difficulty_sparse_categorical_accuracy'][-1]
    # Saving the model in the Deployment directory
    model.save('project_name/Deployment/neural_network_model.keras')

    return (model, X_train, X_test,
            scaler, val_accuracy_thinkbody,
            val_accuracy_feelinglow,
            val_accuracy_sleepdiff)
