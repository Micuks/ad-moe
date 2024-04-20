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


def main():
    seed = 42
    results = []

    # Train model w/o meta-learning
    train_eval_mymodel_vanilla(results, seed)
    train_eval_baselines_vanilla(results, seed)

    # Train model with meta-learning
    train_eval_mymodel_meta(results, seed)

    with io.open("./data1.json", "w") as f:
        json.dump(results, f)
    print(results)


if __name__ == "__main__":
    main()
