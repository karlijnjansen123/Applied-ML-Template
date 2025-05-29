import pandas as pd
import os
import numpy as np
from IPython.display import display
from project_name.data import preprocessing
from project_name.models.KNN import *
from project_name.models.NeuralNetwork import build_neural_network
from project_name.features.featureimportance import *
from project_name.features.feature_correlation import *

# check cwd
print("Current working directory: ", os.getcwd())
filepath = os.path.join(os.getcwd(), "project_name", "data", "HBSC2018.csv")
print(filepath)

# the columns for input
cols_of_interest = ["sex", "health", "timeexe", "sweets_2", "emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9", "thinkbody", "feellow", "beenbullied"]

# alternatively: all columns minus separate social media columns
all_columns_minus_socmed = ["fasfamcar", "fasbedroom", "fascomputers", "fasbathroom", "fasdishwash", "fasholidays", "health", "lifesat", "headache", "stomachache", "backache", "irritable", "nervous", "dizzy", "physact60", "breakfastwd", "breakfastwe", "fruits_2", "vegetables_2", "sweets_2", "softdrinks_2", "fmeal", "toothbr", "timeexe", "smokltm", "smok30d_2", "alcltm", "alc30d_2", "drunkltm", "drunk30d", "cannabisltm_2", "cannabis30d_2", "bodyweight", "bodyheight", "likeschool", "schoolpressure", "studtogether", "studhelpful", "studaccept", "teacheraccept", "teachercare", "teachertrust", "bulliedothers", "beenbullied", "cbulliedothers", "cbeenbullied", "fight12m", "injured12m", "friendhelp", "friendcounton", "friendshare", "friendtalk", "hadsex", "agesex", "contraceptcondom", "contraceptpill", "motherhome1", "fatherhome1", "stepmohome1", "stepfahome1", "fosterhome1", "elsehome1_2", "employfa", "employmo", "employnotfa", "employnotmo", "talkfather", "talkmother", "talkstepmo", "famhelp", "famsup", "famtalk", "famdec", "MBMI", "IRFAS", "IRRELFAS_LMH", "IOTF4", "oweight_who"]

# social media columns to aggregate
emc_cols = ["emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9"]

y = ["thinkbody", "feellow", "sleepdificulty"]

# preprocess
clean_data = preprocessing.preprocess_hbsc_data(filepath, all_columns_minus_socmed, emc_cols, y)
print(clean_data.head())

# input and output
#X = clean_data[["sex", "health", "timeexe", "sweets_2", "beenbullied", "emcsocmed_sum"]]
X = clean_data[[ "health","oweight_who", "famtalk", "backache", "schoolpressure", "breakfastwe", "studaccept", "lifesat", "headache", "stomachache",  "irritable", "nervous", "dizzy", "bodyweight", "bodyheight",  "beenbullied"]]
Y1 = clean_data["thinkbody"]
Y2 = clean_data["feellow"]
Y3 = clean_data["sleepdificulty"]

Y1 = pd.to_numeric(Y1, errors='raise')
Y2 = pd.to_numeric(Y2, errors='raise')
Y3 = pd.to_numeric(Y3, errors='raise')

#column_names = ["sex", "subj. health", "vig. exercise", "eating sweets", "being bullied", "social media"]
column_names = ["emcsocmed_sum", "fasfamcar", "fasbedroom", "fascomputers", "fasbathroom", "fasdishwash", "fasholidays", "health", "lifesat", "headache", "stomachache", "backache", "irritable", "nervous", "dizzy", "physact60", "breakfastwd", "breakfastwe", "fruits_2", "vegetables_2", "sweets_2", "softdrinks_2", "fmeal", "toothbr", "timeexe", "smokltm", "smok30d_2", "alcltm", "alc30d_2", "drunkltm", "drunk30d", "cannabisltm_2", "cannabis30d_2", "bodyweight", "bodyheight", "likeschool", "schoolpressure", "studtogether", "studhelpful", "studaccept", "teacheraccept", "teachercare", "teachertrust", "bulliedothers", "beenbullied", "cbulliedothers", "cbeenbullied", "fight12m", "injured12m", "friendhelp", "friendcounton", "friendshare", "friendtalk", "hadsex", "agesex", "contraceptcondom", "contraceptpill", "motherhome1", "fatherhome1", "stepmohome1", "stepfahome1", "fosterhome1", "elsehome1_2", "employfa", "employmo", "employnotfa", "employnotmo", "talkfather", "talkmother", "talkstepmo", "famhelp", "famsup", "famtalk", "famdec", "MBMI", "IRFAS", "IRRELFAS_LMH", "IOTF4", "oweight_who"]

size_input = X.shape[1]
print(size_input)

print("X shape:", X.shape)
print("X NaNs", X.isna().sum().sum())
print("Y1 shape:", Y1.shape)
print("Y1 NaNs:", Y1.isna().sum())
print("preview of Y values", Y1.head)

# run KNN model
acc, X_tr, X_te, predict_proba = KNN_solver(X, Y1)
print("Accuracy for Body Image:", acc)
#KNN_shap_graphs(X_tr, X_te, predict_proba, "Body Image", column_names=column_names)

acc, X_tr, X_te, predict_proba = KNN_solver(X, Y2)
print("Accuracy for Feeling Low:", acc)
#KNN_shap_graphs(X_tr, X_te, predict_proba, "Feeling Low", column_names=column_names)

acc, X_tr, X_te, predict_proba = KNN_solver(X, Y3)
print("Accuracy for Sleep Difficulty:", acc)
#KNN_shap_graphs(X_tr, X_te, predict_proba, "Sleep Difficulty", column_names=column_names)

#run the neural network
neural_network, X_train, X_test, scaler = build_neural_network(X,Y1,Y2,Y3,size_input)
#print(type(neural_network))

averaged_NN_shap_graphs(build_neural_network, X, Y1, Y2, Y3, size_input, column_names)

#NN_shap_graphs(
#    neural_network,
#    X_train,
#    column_names
#)

#feature_correlation_plot(X_train)
