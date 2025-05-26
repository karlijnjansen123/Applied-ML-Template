import pandas as pd
import numpy as np
import os


def preprocess_hbsc_data(filepath, selected_columns, emc_cols, y):
    # Load data
    print(f"Loading data from: {filepath}")
    data = pd.read_csv(filepath, sep=";")

    #for column in data.columns:
        #print(column)

    # Small check
    print(f"Original data shape: {data.shape}")
    #print("First few rows:")
    #print(data.head(3))

    # Clean column names
    data.columns = data.columns.str.strip()

    # Replace placeholder values with NaN
    data[selected_columns] = data[selected_columns].replace([99, -99, 999, -999, '99', '-99', '999', '-999', 99.0, -99.0, 999.0, -999.0], np.nan)
    data[selected_columns] = data[selected_columns].replace(r'^\s*$', np.nan, regex=True)

    data[y] = data[y].replace([99, -99, 999, -999, '99', '-99', '999', '-999', 99.0, -99.0, 999.0, -999.0], np.nan)
    data[y] = data[y].replace(r'^\s*$', np.nan, regex=True)

    data[selected_columns] = data[selected_columns].apply(pd.to_numeric, errors='coerce')
    data[emc_cols] = data[emc_cols].apply(pd.to_numeric, errors='coerce')
    data[y] = data[y].apply(pd.to_numeric, errors='coerce')

    # Aggregate emcsocsum columns into a sum column
    data['emcsocmed_sum'] = data[emc_cols].sum(axis=1)

    # Impute NaNs in cols_of_interest with median
    data[selected_columns] = data[selected_columns].fillna(data[selected_columns].median())

    # Calculate percentage NaNs in X
    #for column in data[selected_columns]:
    #    num_nans = data[column].isna().sum()
    #    percent_nans = (num_nans / len(data)) * 100
    #    print(f"{column} has {num_nans} NaNs which is {percent_nans:.2f}%")

    # If there is NaN in y, drop the row
    data = data.dropna(subset=y)

    # Impute NaNs in the aggregated emcsocmed_median column
    data['emcsocmed_sum'] = data['emcsocmed_sum'].fillna(data['emcsocmed_sum'].median())

    print("Preprocessing finished.")
    return data
