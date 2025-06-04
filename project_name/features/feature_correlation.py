import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def feature_correlation_plot(input):
    """
    Function that generates and displays a heatmap showing the feature
    correlation between input features.

    :param input: An array representing input feature values
    :return: None
    """

    X_df = pd.DataFrame(input)
    corr_matrix = X_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm',
                annot=True, fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()
