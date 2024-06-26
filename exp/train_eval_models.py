import torch
import time
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, recall_score, f1_score
from mymodel.run import MoEAnomalyDetection
import io
import json
import traceback
from train_eval_models import *
from prepare_data import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from sklearn.metrics import mean_squared_error

unsupservised_models = [
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
    "DAGMM",
    "AutoEncoder-MoEAnomalyDetection",
]
semi_supervised_models = [
    "GANomaly",
    "DeepSAD",
    "REPEN",
    "DevNet",
    "PReNet",
    "FEAWAD",
    "XGBOD",
]
supervised_models = [
    "LR",
    "NB",
    "SVM",
    "MLP",
    "RF",
    "LGB",
    "XGB",
    "CatB",
    "ResNet",
    "FTTransformer",
    "MLP-MoEAnomalyDetection",
]


def get_mymodel_name(expert_type="mlp"):
    if expert_type == "mlp":
        model_name = "MLP-MoEAnomalyDetection"
    elif expert_type == "autoencoder":
        model_name = "AutoEncoder-MoEAnomalyDetection"
    else:
        model_name = "MoEAnomalyDetection"

    return model_name


def calculate_threshold(errors, sensitivity=2):
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    return mean_error + sensitivity * std_error


def evaluate_model(name: str, scores, y_truth, x_original=None, threshold=0.5):

    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)
    if not isinstance(y_truth, np.ndarray):
        y_truth = np.array(y_truth)

    if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
        print("Scores contain NaN or inf.")

    predict_labels = lambda scores, threshold=0.5: (scores > threshold).astype(int)

    # MSE loss models
    if name in unsupservised_models:
        threshold = calculate_threshold(scores)
        # mse = mean_squared_error(y_truth, scores)
        mse = np.mean((x_original - scores) ** 2, axis=1)
        print(mse.shape)
        y_pred = predict_labels(mse, threshold)
        print(f"y_truth[{set(a for a in y_truth)}]")
        print(f"y_pred[{set(a for a in y_pred)}]")
        auc = roc_auc_score(y_truth, y_pred)
        acc = accuracy_score(y_truth, y_pred)
        recall = recall_score(y_truth, y_pred)
        f1 = f1_score(y_truth, y_pred)
    else:
        valid_idx = np.isfinite(scores)
        y_truth_valid = y_truth[valid_idx]
        scores_valid = scores[valid_idx]
        y_pred = predict_labels(scores_valid, threshold)
        print(f"y_truth[{set(a for a in y_truth)}]")
        print(f"y_pred[{set(a for a in y_pred)}]")
        auc = roc_auc_score(y_truth_valid, scores_valid)
        acc = accuracy_score(y_truth_valid, y_pred)
        recall = recall_score(y_truth_valid, y_pred)
        f1 = f1_score(y_truth_valid, y_pred)

    return (auc, acc, recall, f1)


def model_fit(
    model_name: str,
    model_class,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    expert_type="mlp",
    logs_interval=10,
    epochs=50,
    seed=42,
):
    # fit
    if "MoEAnomalyDetection" in model_name:
        model = model_class(
            model_name=model_name,
            expert_type=expert_type,
            logs_interval=logs_interval,
            epochs=epochs,
            seed=seed,
        )
    else:
        model = model_class(model_name=model_name, seed=seed)

    start_time = time.time()
    # if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
    #     print("X_train contains NaN or inf")
    if isinstance(X_train, np.ndarray):
        X_train = torch.from_numpy(X_train).float()
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
    val_auc, val_acc, val_recall, val_f1 = evaluate_model(
        model_name, score, y_val, x_original=X_val
    )
    # val_auc, val_acc = evaluate_model(model_name, score, y_val)

    # test
    start_time = time.time()
    if model_name in ["DAGMM"]:
        score = model.predict_score(X_test=X_test)
    else:
        score = model.predict_score(X_test)
    end_time = time.time()
    test_predict_time = end_time - start_time
    test_auc, test_acc, test_recall, test_f1 = evaluate_model(
        model_name, score, y_test, x_original=X_test
    )
    # test_auc, test_acc = evaluate_model(model_name, score, y_test)

    data = (
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
    return data

    # print(f"Error running {model_name}: {e}")
    # raise Exception(e)


def split_train_val_dataset(dataset, val_ratio=0.1):
    """
    Split train and val dataset
    """
    total_indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(total_indices, test_size=val_ratio)

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset


def prepare_task_splits(train_datasets, val_ratio=0.1):
    """
    Split train and val dataset
    """
    train_split = []
    val_split = []
    for dataset in train_datasets:
        train_part, val_part = split_train_val_dataset(dataset, val_ratio)
        train_split.append(train_part)
        val_split.append(val_part)
    return train_split, val_split


def model_meta_fit(
    model_name: str,
    model_class,
    train_datasets,
    test_datasets,
    expert_type="mlp",
    epochs=50,
    seed=42,
    logs_interval=10,
):
    train_split, val_split = prepare_task_splits(train_datasets, val_ratio=0.1)
    # meta-training style fitting
    start_time = time.time()
    model = model_class(
        model_name=model_name,
        expert_type=expert_type,
        epochs=epochs,
        seed=seed,
        logs_interval=logs_interval,
    )

    # Meta-training
    model.meta_fit(train_split)
    end_time = time.time()
    fit_time = end_time - start_time

    # predict
    # val
    val_scores = []
    val_labels = []
    val_predict_times = []
    X_vals = []
    for task in val_split:
        for X_val, y_val in DataLoader(
            task, batch_size=model.batch_size, shuffle=False
        ):
            start_time = time.time()
            score = model.predict_score(X_val)
            end_time = time.time()
            val_predict_time = end_time - start_time

            X_vals.extend(X_val)
            val_scores.extend(score)
            val_labels.extend(y_val.numpy())
            val_predict_times.append(val_predict_time)

    val_auc, val_acc, val_recall, val_f1 = evaluate_model(
        model_name, val_scores, val_labels, x_original=X_vals
    )
    mean_val_predict_time = np.mean(val_predict_times)

    # test
    test_scores = []
    test_labels = []
    test_predict_times = []
    X_tests = []
    for task in test_datasets:
        for X_test, y_test in DataLoader(
            task, batch_size=model.batch_size, shuffle=False
        ):
            start_time = time.time()
            score = model.predict_score(X_test)
            end_time = time.time()
            test_predict_time = end_time - start_time

            X_tests.extend(X_test)
            test_scores.extend(score)
            test_labels.extend(y_test.numpy())
            test_predict_times.append(test_predict_time)

    test_auc, test_acc, test_recall, test_f1 = evaluate_model(
        model_name, score, y_test, x_original=X_tests
    )
    mean_test_predict_time = np.mean(test_predict_times)

    return (
        fit_time,
        mean_val_predict_time,
        mean_test_predict_time,
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


def train_eval_mymodel_vanilla(
    results: list,
    seed: int,
    expert_type="mlp",
    subset=1,
    epochs=50,
    logs_interval=10,
    scaler_transform=True,
    dataset_name="dbpa",
):
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_vanilla(
        subset=subset,
        scaler_transform=scaler_transform,
        dataset_name=dataset_name,
    )

    model = MoEAnomalyDetection
    print(expert_type)
    model_name = get_mymodel_name(expert_type)
    fit_time, val_predict_time, test_predict_time, metrics = model_fit(
        model_name,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        expert_type=expert_type,
        epochs=epochs,
        seed=seed,
        logs_interval=logs_interval,
    )
    results.append([model_name, metrics, fit_time, val_predict_time, test_predict_time])
    print(
        f"{model_name}: {metrics}, ",
        f"fitting time: {fit_time}, val inference time: {val_predict_time}, test inference time: {test_predict_time}",
    )
    gc.collect()

    return results


def train_eval_mymodel_meta(
    results: list,
    seed: int,
    subset=1,
    epochs=50,
    logs_interval=10,
    expert_type="mlp",
    scaler_transform=True,
    dataset_name="dbpa",
):
    train_datasets, test_datasets = prepare_data_meta_training(
        subset=subset, scaler_transform=scaler_transform, dataset_name=dataset_name
    )
    model = MoEAnomalyDetection
    model_name = get_mymodel_name(expert_type)
    fit_time, val_predict_time, test_predict_time, metrics = model_meta_fit(
        model_name,
        model,
        train_datasets,
        test_datasets,
        expert_type=expert_type,
        epochs=epochs,
        seed=seed,
        logs_interval=logs_interval,
    )
    results.append([model_name, metrics, fit_time, val_predict_time, test_predict_time])
    print(
        f"{model_name}: {metrics}, ",
        f"fitting time: {fit_time}, val inference time: {val_predict_time}, test inference time: {test_predict_time}",
    )
    gc.collect()

    return results


def train_eval_baselines_vanilla(results: list, seed: int):
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_vanilla()
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

    # Model fitting
    for model_name, model_class in tqdm(model_dict.items()):
        try:
            fit_time, val_predict_time, test_predict_time, metrics = model_fit(
                model_name,
                model_class,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                seed,
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

    return results
