from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

    # Manually tune n_neighbors
    best_k = None
    best_score = 0
    scores = []
    n_neighbors_max = min(100, len(X_train))  # 100, or less than training samples
    neighbors = list(range(1, n_neighbors_max, 10))

    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_k = k
            best_knn = knn  # Save the best model

    print(f"Best n_neighbors for '{y.name}': {best_k}")

    # Plot the test accuracy vs. n_neighbors
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(neighbors, scores, marker='o')
        plt.title("KNN Test Accuracy by n_neighbors")
        plt.xlabel("n_neighbors")
        plt.ylabel("Test Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Train the KNN model and evaluate its accuracy on the test set
    # BEWARE the n_neighbors is set to 3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Store probability prediction
    predict_proba = knn.predict_proba

    return accuracy, X_train, X_test, predict_proba
