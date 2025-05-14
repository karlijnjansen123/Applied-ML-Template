import pandas as pd
import os
import numpy as np
from IPython.display import display
from project_name.data import preprocessing
from project_name.models.KNN import *

# check cwd
print("Current working directory: ", os.getcwd())
filepath = os.path.join(os.getcwd(), "project_name", "data", "HBSC2018.csv")
print(filepath)

# Define which columns you care about
cols_of_interest = ["sex", "health", "timeexe", "sweets_2", "sleepdificulty", "emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9", "thinkbody", "feellow", "beenbullied"]

# Columns to aggregate
emc_cols = ["emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9"]

clean_data = preprocessing.preprocess_hbsc_data(filepath, cols_of_interest, emc_cols)

print(clean_data.head())

X = clean_data[["sex", "health", "timeexe", "sweets_2", "beenbullied"]]
Y1 = clean_data["emcsocmed_median"]
Y2 = clean_data["thinkbody"]
Y3 = clean_data["feellow"]
Y4 = clean_data["sleepdificulty"]
print("X shape:", X.shape)
print("Y1 shape:", Y1.shape)
print("Y1 NaNs:", Y1.isna().sum())

# KNN Accuracy
print(KNN_solver(X, Y1))
print(KNN_solver(X, Y2))
print(KNN_solver(X, Y3))
print(KNN_solver(X, Y4))


#if __name__ == '__main__':
 #   hello_world()
