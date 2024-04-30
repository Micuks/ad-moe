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
from mymodel.run import MoEAnomalyDetection
from load_ob import OceanBase_Dataset
from load_dbpa import DBPA_Dataset
import io
import json
import traceback
from train_eval_models import *
from datetime import datetime


def main():
    seed = 42
    epochs = 5
    subset = 0.1
    logs_interval = 5
    expert_type = "autoencoder"  # or 'mlp'
    results = []

    # # Train model w/o meta-learning
    train_eval_mymodel_vanilla(
        results,
        seed,
        subset=subset,
        epochs=epochs,
        logs_interval=logs_interval,
        expert_type=expert_type,
    )
    # train_eval_baselines_vanilla(results, seed)

    # Train model with meta-learning
    train_eval_mymodel_meta(
        results,
        seed,
        subset=subset,
        epochs=epochs,
        logs_interval=logs_interval,
        expert_type=expert_type,
    )

    d = datetime.now()
    with io.open(f"./results/data{d}.json", "w") as f:
        json.dump(results, f)
    print(results)


if __name__ == "__main__":
    main()
