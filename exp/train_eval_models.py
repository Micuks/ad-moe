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


def model_fit(
    model_name: str,
    model_class,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    seed=42,
):
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
        traceback.print_exc()


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
        val_split.append(val_split)
    return train_split, val_split


def model_meta_fit(
    model_name: str,
    model_class,
    train_datasets,
    test_datasets,
    seed=42,
):
    try:
        train_split, val_split = prepare_task_splits(train_datasets, val_ratio=0.1)
        # meta-training style fitting
        start_time = time.time()
        model = model_class(model_name=model_name, seed=seed)

        # Meta-training
        model.meta_fit(train_datasets)
        end_time = time.time()
        fit_time = end_time - start_time

        # predict
        # val
        val_scores = []
        val_labels = []
        val_predict_times = []
        for task in val_split:
            for X_val, y_val in DataLoader(
                task, batch_size=model.batch_size, shuffle=False
            ):
                start_time = time.time()
                score = model.predict_score(X_val)
                end_time = time.time()
                val_predict_time = end_time - start_time

                val_scores.extend(score.numpy())
                val_labels.extend(y_val.numpy())
                val_predict_times.append(val_predict_time)

        val_auc, val_acc = evaluate_model(model_name, val_scores, val_labels)
        mean_val_predict_time = np.mean(val_predict_times)

        # test
        test_scores = []
        test_labels = []
        test_predict_times = []
        for task in test_datasets:
            for X_test, y_test in DataLoader(
                task, batch_size=model.batch_size, shuffle=False
            ):
                start_time = time.time()
                score = model.predict(X_test)
                end_time = time.time()
                test_predict_time = end_time - start_time

                test_predict_times.append(test_predict_time)
                test_scores.extend(score.numpy())
                test_labels.extend(y_test)

        test_auc, test_acc = evaluate_model(model_name, score, y_test)
        mean_test_predict_time = np.mean(test_predict_times)

        return (
            fit_time,
            mean_val_predict_time,
            mean_test_predict_time,
            {
                "val_auc": val_auc,
                "val_acc": val_acc,
                "test_auc": test_auc,
                "test_acc": test_acc,
            },
        )

    except Exception as e:

        traceback.print_exc()


def train_eval_mymodel_vanilla(results: list, seed: int):
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_vanilla()

    model = MoEAnomalyDetection(seed=seed)
    model_name = "MoEAnomalyDetection"
    fit_time, val_predict_time, test_predict_time, metrics = model_fit(
        model_name,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        seed,
    )
    results.append([model_name, metrics, fit_time, val_predict_time, test_predict_time])
    print(
        f"{model_name}: {metrics}, ",
        f"fitting time: {fit_time}, val inference time: {val_predict_time}, test inference time: {test_predict_time}",
    )
    gc.collect()

    return results


def train_eval_mymodel_meta(results: list, seed: int):
    train_datasets, test_datasets = prepare_data_meta_training()
    model = MoEAnomalyDetection(seed=seed)
    model_name = "MoEAnomalyDetection"
    fit_time, val_predict_time, test_predict_time, metrics = model_meta_fit(
        model_name,
        model,
        train_datasets,
        test_datasets,
        seed,
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
