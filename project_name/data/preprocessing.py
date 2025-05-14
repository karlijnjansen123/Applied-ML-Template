import pandas as pd
import os
import numpy as np
from IPython.display import display

# check cwd
print("Current working directory: ", os.getcwd())

# data
data = pd.read_csv("HBSC2018OAed1.1.csv", sep=";")

# a little check to see if the dataset is what we expect, so display the 3 first data rows
display(data.head(3))
print("Shape of the CSV file: ", data.shape)
print("\ncolumn names: \n")
for i in data.columns:
    print(i)

# Define columns you care about
cols_of_interest = ["sex", "health", "timeexe", "sweets_2", "sleepdificulty", "emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9", "thinkbody", "feellow", "beenbullied"]

# Columns emcsocmed1, 2, etc are correlated, and we want it to be 1 target variable.
# Therefore make an aggregated feature based on mean (if not skewed) or median (if skewed)
# Not Yet Implemented

# Convert only those columns to numeric
data[cols_of_interest] = data[cols_of_interest].apply(pd.to_numeric, errors='coerce')

# Handle categorical values with one-hot-encoder

# Not Yet Implemented

# Replace 99, 999, -99, -999 and empty with NaN
# Not Yet Implemented

# Handle NaNs
# Not Yet Implemented

# Normalization
# Not Yet Implemented