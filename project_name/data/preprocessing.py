import pandas as pd
import os
import numpy as np
from IPython.display import display

# check cwd
print("Current working directory: ", os.getcwd())
filepath = os.path.join(os.getcwd(), "HBSC2018.csv")
print(filepath)

data = pd.read_csv(filepath, sep=";")
# this will throw a warning: columns 11, 14, 17, 18 have mixed types.

# a little check to see if the dataset is what we expect, so display the 3 first data rows
display(data.head(3))
print("Shape of the CSV file: ", data.shape)
print("\ncolumn names: \n")
for i in data.columns:
    print(i)

# Define columns you care about
cols_of_interest = ["sex", "health", "timeexe", "sweets_2", "sleepdificulty", "emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9", "thinkbody", "feellow", "beenbullied"]

# Convert only those columns to numeric
data[cols_of_interest] = data[cols_of_interest].apply(pd.to_numeric, errors='coerce')

# Replace 99, 999, -99, -999 and empty with NaN
data = data.replace(99, np.nan)
data = data.replace(-99, np.nan)
data = data.replace(999, np.nan)
data = data.replace(-999, np.nan)
data = data.replace(" ", np.nan)

# Columns emcsocmed1, 2, etc are correlated, and we want it to be 1 target variable.
# Therefore make an aggregated feature based on mean (if not skewed) or median (if skewed)
emc_cols = ['emcsocmed1', 'emcsocmed2', 'emcsocmed3', 'emcsocmed4', 'emcsocmed5', 'emcsocmed6', 'emcsocmed7', 'emcsocmed8', 'emcsocmed9']
data['emcsocmed_median'] = data[emc_cols].median(axis=1)

print(data.loc[:50, ["emcsocmed_median"]])

# Handle categorical values with one-hot-encoder
# Not Yet Implemented

# Normalization
# Not Yet Implemented