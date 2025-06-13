from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

def KNN_solver(X, y, scoring='accuracy', plot=True):
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
    y_proba = knn.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Get the f1_score to compare
    f1_score_knn = f1_score(y_test, y_pred, average='macro')

    # Store probability prediction
    predict_proba = knn.predict_proba

    # AUC
    classes = np.unique(y)
    y_test_bin = label_binarize(
        y_test,
        classes=classes)
    try:
        auc = roc_auc_score(
            y_test_bin,
            y_proba,
            average='macro',
            multi_class='ovr')
    except ValueError:
        auc = np.nan

    return accuracy, X_train, X_test, predict_proba, f1_score_knn, auc