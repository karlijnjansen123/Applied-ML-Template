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

    # Aggregate emcsocmed columns into a median column
    data['emcsocmed_median'] = data[emc_cols].median(axis=1)

    # Impute NaNs in cols_of_interest with median
    data[cols_of_interest] = data[cols_of_interest].fillna(data[cols_of_interest].median())

    # Impute NaNs in the aggregated emcsocmed_median column
    data['emcsocmed_median'] = data['emcsocmed_median'].fillna(data['emcsocmed_median'].median())

    print("Preprocessing finished.")
    return data
