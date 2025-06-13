import pandas as pd
import numpy as np


def preprocess_hbsc_data(filepath, selected_columns, emc_cols, y):
    """
    Function that preprocesses the HBSC dataset for modeling.

    :param filepath: Path to the raw CSV data file
    :param selected_columns: List of selected features
    :param emc_cols: List of columns to be aggregated into 'emcsocmed_sum'
    :param y: Target label column(s) to be cleaned
    :return: Preprocessed pandas DataFrame
    """

    # Load data
    print(f"Loading data from: {filepath}")
    data = pd.read_csv(filepath, sep=";", low_memory=False)

    # Data shape check
    # print(f"Original data shape: {data.shape}")

    # Clean column names
    data.columns = data.columns.str.strip()

    # Replace placeholder values with NaN
    data[selected_columns] = data[selected_columns].replace(
        [
            99, -99, 999, -999,
            '99', '-99', '999', '-999',
            99.0, -99.0, 999.0, -999.0
        ],
        np.nan
    )
    data[selected_columns] = data[selected_columns].replace(
        r'^\s*$', np.nan, regex=True
    )
    data[y] = data[y].replace(
        [
            99, -99, 999, -999,
            '99', '-99', '999', '-999',
            99.0, -99.0, 999.0, -999.0
        ],
        np.nan
    )
    data[y] = data[y].replace(
        r'^\s*$', np.nan, regex=True
    )

    # Convert values to numeric, coercing errors to NaN
    data[selected_columns] = data[selected_columns].apply(
        pd.to_numeric,
        errors='coerce'
    )
    data[emc_cols] = data[emc_cols].apply(
        pd.to_numeric,
        errors='coerce'
    )
    data[y] = data[y].apply(
        pd.to_numeric,
        errors='coerce'
    )

    # Aggregate emcsocsum columns into a sum column
    data['emcsocmed_sum'] = data[emc_cols].sum(axis=1)

    # Impute NaNs in cols_of_interest with median
    data[selected_columns] = data[selected_columns].fillna(
        data[selected_columns].median()
    )

    # Drop rows where target column(s) contain NaN
    data = data.dropna(subset=y)

    # Impute NaNs in the aggregated emcsocmed_sum column with its median
    data['emcsocmed_sum'] = data['emcsocmed_sum'].fillna(
        data['emcsocmed_sum'].median()
    )

    print("Preprocessing finished.")

    return data
