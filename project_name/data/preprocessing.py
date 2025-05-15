import pandas as pd
import numpy as np
import os


def preprocess_hbsc_data(filepath, cols_of_interest, emc_cols):
    # Load data
    print(f"Loading data from: {filepath}")
    data = pd.read_csv(filepath, sep=";")

    # Small check
    print(f"Data shape: {data.shape}")
    print("First few rows:")
    print(data.head(3))

    # Clean column names
    data.columns = data.columns.str.strip()

    # Replace placeholder values with NaN
    data[cols_of_interest] = data[cols_of_interest].replace([99, -99, 999, -999, '99', '-99', '999', '-999', 99.0, -99.0, 999.0, -999.0], np.nan)
    data[cols_of_interest] = data[cols_of_interest].replace(r'^\s*$', np.nan, regex=True)

    data[cols_of_interest] = data[cols_of_interest].apply(pd.to_numeric, errors='coerce')
    data[emc_cols] = data[emc_cols].apply(pd.to_numeric, errors='coerce')

    # Aggregate emcsocsum columns into a sum column
    data['emcsocmed_sum'] = data[emc_cols].sum(axis=1)

    # Impute NaNs in cols_of_interest with median
    data[cols_of_interest] = data[cols_of_interest].fillna(data[cols_of_interest].median())

    # Impute NaNs in the aggregated emcsocmed_median column
    data['emcsocmed_sum'] = data['emcsocmed_sum'].fillna(data['emcsocmed_sum'].median())

    print("Preprocessing finished.")
    return data
