import os
from tabulate import tabulate
from project_name.data import preprocessing
from project_name.models.KNN import *
from project_name.models.NeuralNetwork import *
from project_name.features.featureimportance import *
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
print(clean_data.head())

# input and output
# initial selection
# X = clean_data[["sex", "health", "timeexe",
# "sweets_2", "beenbullied", "emcsocmed_sum"]]
# corresponding column names
# column_names = ["sex", "subj. health",
# "vig. exercise", "eating sweets", "being bullied", "social media"]

# all X
# X = clean_data[["emcsocmed_sum",
# "fasfamcar", "fasbedroom",
# "fascomputers", "fasbathroom",
# "fasdishwash", "fasholidays",
# "health", "lifesat", "headache",
# "stomachache", "backache",
# "irritable", "nervous", "dizzy",
# "physact60", "breakfastwd",
# "breakfastwe", "fruits_2", "vegetables_2",
# "sweets_2", "softdrinks_2",
# "fmeal", "toothbr", "timeexe", "smokltm",
# "smok30d_2", "alcltm",
# "alc30d_2", "drunkltm", "drunk30d",
# "cannabisltm_2", "cannabis30d_2",
# "bodyweight", "bodyheight", "likeschool",
# "schoolpressure", "studtogether",
# "studhelpful", "studaccept", "teacheraccept",
# "teachercare", "teachertrust",
# "bulliedothers", "beenbullied", "cbulliedothers",
# "cbeenbullied", "fight12m",
# "injured12m", "friendhelp", "friendcounton",
# "friendshare", "friendtalk",
# "hadsex", "agesex", "contraceptcondom",
# "contraceptpill", "motherhome1",
# "fatherhome1", "stepmohome1", "stepfahome1",
# "fosterhome1", "elsehome1_2",
# "employfa", "employmo", "employnotfa",
# "employnotmo", "talkfather", "talkmother",
# "talkstepmo", "famhelp", "famsup", "famtalk",
# "famdec", "MBMI", "IRFAS",
# "IRRELFAS_LMH", "IOTF4", "oweight_who"]]

# corresponding column names:
# column_names = ["emcsocmed_sum", "fasfamcar", "fasbedroom", "fascomputers", "fasbathroom", "fasdishwash", "fasholidays", "health", "lifesat", "headache", "stomachache", "backache", "irritable", "nervous", "dizzy", "physact60", "breakfastwd", "breakfastwe", "fruits_2", "vegetables_2", "sweets_2", "softdrinks_2", "fmeal", "toothbr", "timeexe", "smokltm", "smok30d_2", "alcltm", "alc30d_2", "drunkltm", "drunk30d", "cannabisltm_2", "cannabis30d_2", "bodyweight", "bodyheight", "likeschool", "schoolpressure", "studtogether", "studhelpful", "studaccept", "teacheraccept", "teachercare", "teachertrust", "bulliedothers", "beenbullied", "cbulliedothers", "cbeenbullied", "fight12m", "injured12m", "friendhelp", "friendcounton", "friendshare", "friendtalk", "hadsex", "agesex", "contraceptcondom", "contraceptpill", "motherhome1", "fatherhome1", "stepmohome1", "stepfahome1", "fosterhome1", "elsehome1_2", "employfa", "employmo", "employnotfa", "employnotmo", "talkfather", "talkmother", "talkstepmo", "famhelp", "famsup", "famtalk", "famdec", "MBMI", "IRFAS", "IRRELFAS_LMH", "IOTF4", "oweight_who"]

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

Y1 = clean_data["thinkbody"]
Y2 = clean_data["feellow"]
Y3 = clean_data["sleepdificulty"]

Y1 = pd.to_numeric(Y1, errors='raise')
Y2 = pd.to_numeric(Y2, errors='raise')
Y3 = pd.to_numeric(Y3, errors='raise')

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
# KNN_shap_graphs(X_tr, X_te, predict_proba,
# "Body Image", column_names=column_names)

acc, X_tr, X_te, predict_proba = KNN_solver(X, Y2)
print("Accuracy for Feeling Low:", acc)
# KNN_shap_graphs(X_tr, X_te, predict_proba,
# "Feeling Low", column_names=column_names)

acc, X_tr, X_te, predict_proba = KNN_solver(X, Y3)
print("Accuracy for Sleep Difficulty:", acc)
# KNN_shap_graphs(X_tr, X_te, predict_proba,
# "Sleep Difficulty", column_names=column_names)

# run the neural network
# test-train split
(X_train, X_test, Y1_train, Y1_test, Y2_train,
 Y2_test, Y3_train, Y3_test) = test_train_split(
    X, Y1, Y2, Y3
)
# run NN
(neural_network, X_train, X_test, scaler,
 val_acc1, val_acc2, val_acc3) = build_neural_network(
    X_train, X_test,
    Y1_train, Y1_test,
    Y2_train, Y2_test,
    Y3_train, Y3_test,
    size_input
)
print(type(neural_network))
# X_train.sample(10000, random_state=42)
# .to_csv("shap_background.csv", index=False)

# normal shap graph - overall importance
NN_shap_graphs(
    neural_network,
    X_train,
    column_names
 )

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