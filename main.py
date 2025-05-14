import pandas as pd
import os
import numpy as np
from IPython.display import display
from project_name.data.preprocessing import preprocess_hbsc_data

# check cwd
print("Current working directory: ", os.getcwd())
filepath = os.path.join(os.getcwd(), "project_name", "data", "HBSC2018.csv")
print(filepath)

# Define which columns you care about
cols_of_interest = ["sex", "health", "timeexe", "sweets_2", "sleepdificulty", "emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9", "thinkbody", "feellow", "beenbullied"]

# Columns to aggregate
emc_cols = ["emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9"]

clean_data = preprocess_hbsc_data(filepath, cols_of_interest, emc_cols)

print(clean_data.head())

#if __name__ == '__main__':
 #   hello_world()
