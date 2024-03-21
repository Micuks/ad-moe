# %%
import torch
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from run import MoEAnomalyDetection
from tqdm import tqdm
import gc


def evaluate_model(name: str, scores, y_truth, threshold=0.5):
    if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
        print("Scores contain NaN or inf.")
    valid_idx = np.isfinite(scores)
    y_truth_valid = y_truth[valid_idx]
    scores_valid = scores[valid_idx]
    print(scores_valid)

    predict_labels = lambda scores, threshold=0.5: (scores > threshold).astype(int)
    y_pred = predict_labels(scores, threshold)
    auc = roc_auc_score(y_truth_valid, scores_valid)
    acc = accuracy_score(y_truth_valid, y_pred)
    recall = recall_score(y_truth_valid, y_pred)
    f1 = f1_score(y_truth_valid, y_pred)
    # print(f"{name}: AUC:{auc} Acc:{acc} Recall:{recall} f1:{f1}")
    return (auc, acc, recall, f1)


seed = 42


def model_fit(model_name: str, model_class, seed):
    try:
        # fit
        start_time = time.time()
        model = model_class(model_name=model_name, seed=seed)
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("X_train contains NaN or inf")
        model.fit(X_train, y_train)
        end_time = time.time()
        fit_time = end_time - start_time

        # predict
        # val
        start_time = time.time()
        if model_name in ["DAGMM"]:
            score = model.predict_score(X_test=X_val, X_train=X_train)
        else:
            score = model.predict_score(X_val)
        end_time = time.time()
        val_predict_time = end_time - start_time
        print(score)
        val_auc, val_acc, val_recall, val_f1 = evaluate_model(model_name, score, y_val)

        # test
        start_time = time.time()
        score = model.predict_score(X_test)
        end_time = time.time()
        test_predict_time = end_time - start_time
        test_auc, test_acc, test_recall, test_f1 = evaluate_model(
            model_name, score, y_test
        )

        return (
            fit_time,
            val_predict_time,
            test_predict_time,
            {
                "val_auc": val_auc,
                "val_acc": val_acc,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "test_auc": test_auc,
                "test_acc": test_acc,
                "test_recall": test_recall,
                "test_f1": test_f1,
            },
        )
    except Exception as e:
        print(f"Error running {model_name}: {e}")
        # Print e's traceback
        print(e.with_traceback())


basepath = "./data"

train_csvs = [
    # "memory_limit_merge_label",
    "io_saturation_merge_ob_2024-03-0706_11_36UTC_label",
    "io_saturation_merge_ob_2024-03-0717_44_47UTC_label",
    # "memory_limit_merge_ob_2024-03-0508_09_42UTC_label",
    "memory_limit_merge_ob_2024-03-0508_56_01UTC_label",
    "memory_limit_merge_ob_2024-03-0509_42_19UTC_label",
    # "memory_limit_merge_ob_2024-03-0510_28_34UTC_label",
    # "memory_limit_merge_ob_2024-03-0511_14_49UTC_label",
    # "memory_limit_4_16_merge_ob_2024-03-1016_04_40UTC_label",
    # "memory_limit_4_16_merge_ob_2024-03-1017_05_55UTC_label",
]

test_csvs = [
    # "memory_limit_merge_label",
    "io_saturation_merge_ob_2024-03-0706_11_36UTC_label",
    "io_saturation_merge_ob_2024-03-0717_44_47UTC_label",
    # "memory_limit_merge_ob_2024-03-0508_09_42UTC_label",
    # "memory_limit_merge_ob_2024-03-0508_56_01UTC_label",
    # "memory_limit_merge_ob_2024-03-0509_42_19UTC_label",
    "memory_limit_merge_ob_2024-03-0510_28_34UTC_label",
    "memory_limit_merge_ob_2024-03-0511_14_49UTC_label",
    "memory_limit_4_16_merge_ob_2024-03-1016_04_40UTC_label",
    "memory_limit_4_16_merge_ob_2024-03-1017_05_55UTC_label",
]

train_tasks = [
    "io_saturation_merge",
    "memory_limit_merge",
    # "memory_limit_4_16_merge",
]

test_tasks = [
    "io_saturation_merge",
    "memory_limit_merge",
    "memory_limit_4_16_merge",
]


def get_dfs(tasks: list):
    return [[] for _ in range(len(tasks))]


dfs_train = get_dfs(train_tasks)
dfs_test = get_dfs(test_tasks)

f = lambda x: os.path.join(basepath, x + ".csv")
for csv in train_csvs:
    df = pd.read_csv(f(csv))
    for i, task in enumerate(train_tasks):
        if task in csv:
            dfs_train[i].append(df)
            break

for csv in test_csvs:
    df = pd.read_csv(f(csv))
    for i, t in enumerate(test_tasks):
        if t in csv:
            dfs_test[i].append(df)
            break

dfs_train_all = [pd.concat(dfs, axis=0) for dfs in dfs_train]
dfs_test_all = [pd.concat(dfs, axis=0) for dfs in dfs_test]

num_norm = lambda x: len(x[x["label"] == 0])
num_anom = lambda x: len(x[x["label"] != 0])
print(
    f"dfs_train_all[{[df.shape for df in dfs_train_all]}], normal[{[num_norm(df) for df in dfs_train_all]}], anomaly[{[num_anom(df) for df in dfs_train_all]}]"
)
print(
    f"dfs_test_all[{[df.shape for df in dfs_test_all]}], normal[{[num_norm(df) for df in dfs_test_all]}], anomaly[{[num_anom(df) for df in dfs_test_all]}]"
)

# %% [markdown]
# # Data cleaning
# ## Remove irrelevant data
# Remove those columns whose data never change

# %%
# Find those columns in df_all that are always zero
zero_cols = dfs_train_all[0].columns[(dfs_train_all[0] == 0).all()]
print(len(zero_cols))

# Drop those cols
dfs_train_all = [df.drop(zero_cols, axis=1) for df in dfs_train_all]
dfs_test_all = [df.drop(zero_cols, axis=1) for df in dfs_test_all]

# %% [markdown]
# # Dataset construction
#
# Reconstruct dataset with respect to anomaly ratio


# %%
def construct_dataset(dfs, anomaly_ratio=0.1):
    new_dfs = []
    for df in dfs:
        num_normal = num_anom(df)
        required_num_anomaly = int(num_normal * (anomaly_ratio / (1 - anomaly_ratio)))

        normal_samples = df[df["label"] == 0]
        anomaly_samples = df[df["label"] != 0].sample(
            n=required_num_anomaly, random_state=42
        )

        new_dfs.append(
            pd.concat([normal_samples, anomaly_samples]).sample(
                frac=1, random_state=42, axis=0
            )
        )

    return new_dfs


dfs_train_all = construct_dataset(dfs_train_all, anomaly_ratio=0.2)
dfs_test_all = construct_dataset(dfs_test_all, anomaly_ratio=0.5)

print(
    f"dfs_train_all[{[df.shape for df in dfs_train_all]}], normal[{[num_norm(df) for df in dfs_train_all]}], anomaly[{[num_anom(df) for df in dfs_train_all]}]"
)
print(
    f"dfs_test_all[{[df.shape for df in dfs_test_all]}], normal[{[num_norm(df) for df in dfs_test_all]}], anomaly[{[num_anom(df) for df in dfs_test_all]}]"
)


# %% [markdown]
# # Without meta-learning
# First test those models without meta-learning

# %%
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, recall_score, f1_score

epochs = 10
seed = 42

total_train = pd.concat(dfs_train_all)

X = total_train.drop(columns=["label"]).to_numpy()
y = total_train["label"].to_numpy()
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=seed, shuffle=True
)

total_test = pd.concat(dfs_test_all)
X_test = total_test.drop(columns=["label"]).to_numpy()
y_test = total_test["label"].to_numpy()
print(f"X[{X.shape}] y[{y.shape}]")

model = MoEAnomalyDetection(seed=seed)
fit_time, val_predict_time, test_predict_time, metrics = model_fit(
    "MoEAnomalyDetection", MoEAnomalyDetection, seed
)
print(
    f"{model_name}: {metrics}, ",
    f"fitting time: {fit_time}, val inference time: {val_predict_time}, test inference time: {test_predict_time}",
)
gc.collect()

# %% [markdown]
# # Compare all baselines
# Compare all baselines, including unsupervised, semi-supervised and full-supervised models

# %%


exit()

# %%
model_dict = {}

# Unsupervised algorithms
from adbench.baseline.PyOD import PYOD
from adbench.baseline.DAGMM.run import DAGMM

# from pyod
for _ in [
    "IForest",
    "OCSVM",
    "CBLOF",
    "COF",
    "COPOD",
    "ECOD",
    "FeatureBagging",
    "HBOS",
    "KNN",
    "LODA",
    "LOF",
    "LSCP",
    "MCD",
    "PCA",
    "SOD",
    "SOGAAL",
    "MOGAAL",
    "DeepSVDD",
]:
    model_dict[_] = PYOD
# from dagmm
model_dict["DAGMM"] = DAGMM

# Semi-supervised algorithms
from adbench.baseline.PyOD import PYOD
from adbench.baseline.GANomaly.run import GANomaly
from adbench.baseline.DeepSAD.src.run import DeepSAD
from adbench.baseline.REPEN.run import REPEN
from adbench.baseline.DevNet.run import DevNet
from adbench.baseline.PReNet.run import PReNet
from adbench.baseline.FEAWAD.run import FEAWAD

model_dict.update(
    {
        "GANomaly": GANomaly,
        "DeepSAD": DeepSAD,
        "REPEN": REPEN,
        "DevNet": DevNet,
        "PReNet": PReNet,
        "FEAWAD": FEAWAD,
        "XGBOD": PYOD,
    }
)

# Full-supervised algorithms
from adbench.baseline.Supervised import supervised
from adbench.baseline.FTTransformer.run import FTTransformer

# from sklearn
for _ in ["LR", "NB", "SVM", "MLP", "RF", "LGB", "XGB", "CatB"]:
    model_dict[_] = supervised
# ResNet and FTTransformer for tabular data
for _ in ["ResNet", "FTTransformer"]:
    model_dict[_] = FTTransformer

# Remove the computational-expensive models
for _ in ["SOGAAL", "MOGAAL", "LSCP", "MCD", "FeatureBagging"]:
    if _ in model_dict.keys():
        model_dict.pop(_)


# %%

results = []
# Model fitting
for model_name, model_class in tqdm(model_dict.items()):
    try:
        fit_time, val_predict_time, test_predict_time, metrics = model_fit(
            model_name, model_class, seed
        )
        results.append(
            [model_name, metrics, fit_time, val_predict_time, test_predict_time]
        )
        print(
            f"{model_name}: {metrics}, ",
            f"fitting time: {fit_time}, val inference time: {val_predict_time}, test inference time: {test_predict_time}",
        )
        gc.collect()
    except Exception as e:
        print(e)

# %%
results

# %%
results
