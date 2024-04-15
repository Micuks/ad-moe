import torch
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import gc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, recall_score, f1_score
from run import MoEAnomalyDetection
from load_ob import OceanBase_Dataset
from load_dbpa import DBPA_Dataset
import io
import json


seed = 42
results = []


def evaluate_model(name: str, scores, y_truth, threshold=0.5):
    if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
        print("Scores contain NaN or inf.")
    valid_idx = np.isfinite(scores)
    y_truth_valid = y_truth[valid_idx]
    scores_valid = scores[valid_idx]
    # print(scores_valid)

    predict_labels = lambda scores, threshold=0.5: (scores > threshold).astype(int)
    y_pred = predict_labels(scores, threshold)
    auc = roc_auc_score(y_truth_valid, scores_valid)
    acc = accuracy_score(y_truth_valid, y_pred)
    # recall = recall_score(y_truth_valid, y_pred)
    # f1 = f1_score(y_truth_valid, y_pred)
    # print(f"{name}: AUC:{auc} Acc:{acc} Recall:{recall} f1:{f1}")
    return (auc, acc)
    # return (auc, acc, recall, f1)


def model_fit(model_name: str, model_class, seed):
    try:
        # fit
        start_time = time.time()
        model = model_class(model_name=model_name, seed=seed)
        # if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        #     print("X_train contains NaN or inf")
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
        # print(score)
        # val_auc, val_acc, val_recall, val_f1 = evaluate_model(model_name, score, y_val)
        val_auc, val_acc = evaluate_model(model_name, score, y_val)

        # test
        start_time = time.time()
        score = model.predict_score(X_test)
        end_time = time.time()
        test_predict_time = end_time - start_time
        # test_auc, test_acc, test_recall, test_f1 = evaluate_model(
        #     model_name, score, y_test
        # )
        test_auc, test_acc = evaluate_model(model_name, score, y_test)

        return (
            fit_time,
            val_predict_time,
            test_predict_time,
            {
                "val_auc": val_auc,
                "val_acc": val_acc,
                # "val_recall": val_recall,
                # "val_f1": val_f1,
                "test_auc": test_auc,
                "test_acc": test_acc,
                # "test_recall": test_recall,
                # "test_f1": test_f1,
            },
        )
    except Exception as e:
        print(f"Error running {model_name}: {e}")
        # Print e's traceback
        print(e.with_traceback())


# Train on OceanBase dataset
# dataset = OceanBase_Dataset(anomaly_ratio=0.2)
dataset = DBPA_Dataset(anomaly_ratio=0.2)
# dfs_train_all, dfs_test_all = ob_dataset.load_dataset(drop_zero_cols=True)
dfs_train_all, dfs_test_all = dataset.load_dataset(drop_zero_cols=False)

# %% [markdown]
# # Without meta-learning
# First test those models without meta-learning

# %%

epochs = 10
seed = 42

total_train = pd.concat(dfs_train_all)
total_test = pd.concat(dfs_test_all)
total_train, total_test = dataset._drop_nans(total_train, total_test)
# total_train["label"] = total_train["label"].astype(int)
# total_test["label"] = total_test["label"].astype(int)

print("total_train", total_train.describe())
print("total_test", total_test.describe())

X = total_train.drop(columns=["label"]).to_numpy()
y = total_train["label"].to_numpy()
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=seed, shuffle=True
)

X_test = total_test.drop(columns=["label"]).to_numpy()
y_test = total_test["label"].to_numpy()
print(f"X[{X.shape}] y[{y.shape}]")

model = MoEAnomalyDetection(seed=seed)
model_name = "MoEAnomalyDetection"
fit_time, val_predict_time, test_predict_time, metrics = model_fit(
    "MoEAnomalyDetection", MoEAnomalyDetection, seed
)
results.append([model_name, metrics, fit_time, val_predict_time, test_predict_time])
print(
    f"{model_name}: {metrics}, ",
    f"fitting time: {fit_time}, val inference time: {val_predict_time}, test inference time: {test_predict_time}",
)
gc.collect()

# %% [markdown]
# # Compare all baselines
# Compare all baselines, including unsupervised, semi-supervised and full-supervised models

# exit()

# Unsupervised algorithms
from adbench.baseline.PyOD import PYOD
from adbench.baseline.DAGMM.run import DAGMM

model_dict = {}

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
with io.open("data.json", "w") as f:
    json.dump(results, f)
print(results)
