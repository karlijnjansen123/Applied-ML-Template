import os
from distutils.command.build import build
import pandas as pd
from tabulate import tabulate
from project_name.data import preprocessing
from project_name.models.KNN import KNN_solver
from project_name.models.NeuralNetwork import (
    build_neural_network,
    test_train_split
)
from project_name.features.featureimportance import (
    averaged_NN_shap_graphs_per_output
)
from collections import Counter

# Check current working directory
print("Current working directory: ", os.getcwd())
filepath = os.path.join(
    os.getcwd(), "project_name", "data", "HBSC2018.csv"
)
print(filepath)

# The columns for input
cols_of_interest = [
    "sex", "health", "timeexe", "sweets_2",
    "emcsocmed1", "emcsocmed2", "emcsocmed3",
    "emcsocmed4", "emcsocmed5", "emcsocmed6",
    "emcsocmed7", "emcsocmed8", "emcsocmed9"
    ]
# alternatively: all columns minus separate social media columns
all_columns_minus_socmed = [
    "fasfamcar", "fasbedroom", "fascomputers", "fasbathroom",
    "fasdishwash", "fasholidays", "health", "lifesat", "headache",
    "stomachache", "backache", "irritable", "nervous", "dizzy",
    "physact60", "breakfastwd", "breakfastwe", "fruits_2",
    "vegetables_2", "sweets_2", "softdrinks_2", "fmeal", "toothbr",
    "timeexe", "smokltm", "smok30d_2", "alcltm", "alc30d_2",
    "drunkltm", "drunk30d", "cannabisltm_2", "cannabis30d_2",
    "bodyweight", "bodyheight", "likeschool", "schoolpressure",
    "studtogether", "studhelpful", "studaccept", "teacheraccept",
    "teachercare", "teachertrust", "bulliedothers", "beenbullied",
    "cbulliedothers", "cbeenbullied", "fight12m", "injured12m",
    "friendhelp", "friendcounton", "friendshare", "friendtalk",
    "hadsex", "agesex", "contraceptcondom", "contraceptpill",
    "motherhome1", "fatherhome1", "stepmohome1", "stepfahome1",
    "fosterhome1", "elsehome1_2", "employfa", "employmo",
    "employnotfa", "employnotmo", "talkfather", "talkmother",
    "talkstepmo", "famhelp", "famsup", "famtalk", "famdec",
    "MBMI", "IRFAS", "IRRELFAS_LMH", "IOTF4", "oweight_who"
]

# social media columns to aggregate
emc_cols = [
    "emcsocmed1", "emcsocmed2", "emcsocmed3",
    "emcsocmed4", "emcsocmed5", "emcsocmed6",
    "emcsocmed7", "emcsocmed8", "emcsocmed9"
]

y = ["thinkbody", "feellow", "sleepdificulty"]

# Preprocess the data
clean_data = preprocessing.preprocess_hbsc_data(
    filepath, all_columns_minus_socmed, emc_cols, y
)


# most important X
X = clean_data[["bodyweight", "bodyheight", "emcsocmed_sum",
                "nervous", "irritable", "lifesat", "breakfastwd",
                "health", "fruits_2", "headache", "fight12m", "friendcounton",
                "softdrinks_2", "dizzy", "sweets_2", "friendhelp"]]

# corresponding column names:
column_names = ["bodyweight", "bodyheight", "emcsocmed_sum",
                "nervous", "irritable", "lifesat", "breakfastwd",
                "health", "fruits_2", "headache", "fight12m",
                "friendcounton", "softdrinks_2", "dizzy",
                "sweets_2", "friendhelp"]

# Clean data for target labels
Y1 = clean_data["thinkbody"]
Y2 = clean_data["feellow"]
Y3 = clean_data["sleepdificulty"]

# Force to numeric
Y1 = pd.to_numeric(Y1, errors='raise')
Y2 = pd.to_numeric(Y2, errors='raise')
Y3 = pd.to_numeric(Y3, errors='raise')

# Size variable for input layer
size_input = X.shape[1]


# Run KNN model for Body Image
acc, X_tr, X_te, predict_proba, f1_score_knn = KNN_solver(X, Y1)
print("Accuracy for Body Image:", acc)
print("F1 score for Body Image:", f1_score_knn)
# KNN_shap_graphs(X_tr, X_te, predict_proba,
# "Body Image", column_names=column_names)


# Run the KNN model for Feeling Low
acc, X_tr, X_te, predict_proba, f1_score_knn = KNN_solver(X, Y2)
print("Accuracy for Feeling Low:", acc)
print("F1 score for Feeling Low:", f1_score_knn)
# KNN_shap_graphs(X_tr, X_te, predict_proba,
# "Feeling Low", column_names=column_names)

# Run the KNN model for Feeling Low
acc, X_tr, X_te, predict_proba, f1_score_knn = KNN_solver(X, Y3)
print("Accuracy for Sleep Difficulty:", acc)
print("F1 score for Sleep Difficulty:", f1_score_knn)
# KNN_shap_graphs(X_tr, X_te, predict_proba,
# "Sleep Difficulty", column_names=column_names)


# Test-train split
(X_train, X_test, Y1_train, Y1_test, Y2_train,
 Y2_test, Y3_train, Y3_test) = test_train_split(
    X, Y1, Y2, Y3
)

# Build Neural Network
(neural_network, X_train, X_test, scaler,
 val_acc1, val_acc2, val_acc3,
 metrics_dict) = build_neural_network(
    X_train, X_test,
    Y1_train, Y1_test,
    Y2_train, Y2_test,
    Y3_train, Y3_test,
    size_input
)


print("\n=== Neural Network Validation Metrics ===")
print(
    f"Think Body - Val Accuracy: {val_acc1:.3f}, "
    f"F1: {metrics_dict['think_body']['f1_score']:.3f}, "
    f"AUC: {metrics_dict['think_body']['auc_score']:.3f}"
)
print(
    f"Feeling Low - Val Accuracy: {val_acc2:.3f}, "
    f"F1: {metrics_dict['feeling_low']['f1_score']:.3f}, "
    f"AUC: {metrics_dict['feeling_low']['auc_score']:.3f}"
)
print(
    f"Sleep Difficulty - Val Accuracy: {val_acc3:.3f}, "
    f"F1: {metrics_dict['sleep_difficulty']['f1_score']:.3f}, "
    f"AUC: {metrics_dict['sleep_difficulty']['auc_score']:.3f}"
)


# The following comments are for feature importance
# X_train.sample(10000, random_state=42)
# .to_csv("shap_background.csv", index=False)

# normal shap graph - overall importance
# NN_shap_graphs(
#    neural_network,
#    X_train,
#    column_names
# )

# averaged shap graph over 5 model trainings - overall importance
# averaged_NN_shap_graphs(build_neural_network, X_train,
# X_test, Y1_train, Y1_test, Y2_train, Y2_test,
# Y3_train, Y3_test, size_input, column_names)

# averaged shap graph over n model trainings - importance per outcome
averaged_NN_shap_graphs_per_output(
    build_neural_network, X_train, X_test,
    Y1_train, Y1_test, Y2_train, Y2_test, Y3_train, Y3_test,
    size_input, column_names, n_runs=1
)
# comparing to random guessing


# Compute the random baseline per target
def compute_majority_baseline(y, label: str = ""):
    counter = Counter(y)
    most_common_class, count = counter.most_common(1)[0]
    baseline_accuracy = count / len(y)
    print(f"Majority class for {label}: {most_common_class}")
    print(f"Baseline accuracy for {label}: {baseline_accuracy:.3f}")
    return most_common_class, baseline_accuracy


# Baseline evaluation
randomguessing_bodyimage = (compute_majority_baseline
                            (Y1, label="Body Image"))
randomguessing_feellinglow = (compute_majority_baseline
                              (Y2, label="Feeling Low"))
randomguessing_sleepdifficulty = (compute_majority_baseline
                                  (Y3, label="Sleep Difficulty"))

comparison = {
    "Target": ["Body Image", "Feeling Low", "Sleep Difficulty"],
    "Majority Baseline Accuracy": [
        round(randomguessing_bodyimage[1], 3),
        round(randomguessing_feellinglow[1], 3),
        round(randomguessing_sleepdifficulty[1], 3)
    ],
    "Validation Accuracy (Neural Network)": [
        round(val_acc1, 3),
        round(val_acc2, 3),
        round(val_acc2, 3)
    ],
    "Above Baseline?": [
        val_acc1 > randomguessing_bodyimage[1],
        val_acc2 > randomguessing_feellinglow[1],
        val_acc3 > randomguessing_sleepdifficulty[1]
    ]
}

df_comparison = pd.DataFrame(comparison)
print(tabulate(df_comparison, headers='keys', tablefmt='pretty'))
