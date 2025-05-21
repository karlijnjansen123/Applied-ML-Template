import pandas as pd
import os
import numpy as np
from IPython.display import display
from project_name.data import preprocessing
from project_name.models.KNN import *
from project_name.models.NeuralNetwork import build_neural_network

# check cwd
print("Current working directory: ", os.getcwd())
filepath = os.path.join(os.getcwd(), "project_name", "data", "HBSC2018.csv")
print(filepath)

# the columns for input
cols_of_interest = ["sex", "health", "timeexe", "sweets_2", "sleepdificulty", "emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9", "thinkbody", "feellow", "beenbullied"]

# social media columns to aggregate
emc_cols = ["emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9"]

# preprocess
clean_data = preprocessing.preprocess_hbsc_data(filepath, cols_of_interest, emc_cols)
print(clean_data.head())

<<<<<<< HEAD
# input and output
X = clean_data[["sex", "health", "timeexe", "sweets_2", "beenbullied", "emcsocmed_sum"]]
Y1 = clean_data["thinkbody"]
Y2 = clean_data["feellow"]
Y3 = clean_data["sleepdificulty"]

size_input = X.shape[1]
print(size_input)


print("X shape:", X.shape)

# run KNN model
print(KNN_solver(X, Y1))
print(KNN_solver(X, Y2))
print(KNN_solver(X, Y3))

#run the neural network
neural_network = build_neural_network(X,Y1,Y2,Y3,size_input)
=======
#column_names = [i for i in clean_data.columns]
#print(column_names)
column_names = ["sex", "health", "timeexe", "sweets_2", "sleepdificulty"]
X = clean_data[["sex", "health", "timeexe", "sweets_2", "sleepdificulty"]]
Y1 = clean_data["emcsocmed_median"]
Y2 = clean_data["thinkbody"]
Y3 = clean_data["feellow"]
Y4 = clean_data["sleepdificulty"]
print("X shape:", X.shape)
print("Y1 shape:", Y1.shape)
print("Y1 NaNs:", Y1.isna().sum())


acc, X_tr, X_te, predict_proba = KNN_solver(X, Y1)
print(acc)
shap_graphs(X_tr, X_te, predict_proba, column_names=column_names)

acc, X_tr, X_te, predict_proba = KNN_solver(X, Y2)
print(acc)
shap_graphs(X_tr, X_te, predict_proba, column_names=column_names)

acc, X_tr, X_te, predict_proba = KNN_solver(X, Y3)
print(acc)
shap_graphs(X_tr, X_te, predict_proba, column_names=column_names)

acc, X_tr, X_te, predict_proba = KNN_solver(X, Y4)
print(acc)
shap_graphs(X_tr, X_te, predict_proba, column_names=column_names)



#if __name__ == '__main__':
 #   hello_world()
>>>>>>> 2e5b088 (added functionality for KNN and shap as two separate funtions)
