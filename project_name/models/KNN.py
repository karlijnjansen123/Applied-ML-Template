from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def KNN_solver(X, y):
    """
    Function that trains K-Nearest Neighbors model and evaluates performance


    :param X: Input features
    :param y: Target label
    :return: Accuracy, scaled training/test features and predict_proba function
    """

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Standardize the feature values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the KNN model and evaluate its accuracy on the test set
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Store probability prediction
    predict_proba = knn.predict_proba

    return accuracy, X_train, X_test, predict_proba
