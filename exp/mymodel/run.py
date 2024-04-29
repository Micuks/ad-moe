import torch
import pandas as pd
import numpy as np
import random
from torch import nn
from mymodel.model import _MOEAnomalyDetection
from mymodel.fit import fit, meta_fit
import traceback
import torch.nn.functional as F


class MoEAnomalyDetection:
    def __init__(
        self,
        seed: int,
        model_name="MoEAnomalyDetection",
        expert_type="mlp",
        num_experts: int = 8,
        epochs: int = 50,
        batch_num: int = 20,
        batch_size: int = 128,
        lr: float = 1e-4,
        meta_lr: float = 1e-4,
        inner_update_steps=5,
        inner_lr=1e-3,  # LR for task-specific updates
        meta_batch_size=32,  # Number of tasks per meta-update
        weight_decay: float = 1e-5,
        logs_interval=10,
    ):
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logs_interval = logs_interval
        self.model_name = model_name
        self.expert_type = expert_type
        self.num_experts = num_experts

        # hyper parameters
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.lr = lr
        self.meta_lr = meta_lr
        self.inner_update_steps = inner_update_steps
        self.inner_lr = inner_lr
        self.meta_batch_size = meta_batch_size
        self.weight_decay = weight_decay

    def set_seed(self, seed):
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def fit(self, X_train, y_train, ratio=None):
        try:
            X_train = X_train.astype("float32")
            self.X_train_tensor = torch.from_numpy(X_train).float()
            input_size = X_train.shape[1]
        except Exception as e:
            raise ValueError(e)

        self.y_train = y_train

        self.set_seed(self.seed)
        self.model = _MOEAnomalyDetection(
            input_size=input_size,
            num_experts=self.num_experts,
            expert_type=self.expert_type,
        )
        self.model.apply(self.init_weights)
        self.model.to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # training
        fit(
            X_train_tensor=self.X_train_tensor,
            y_train=self.y_train,
            model=self.model,
            optimizer=optimizer,
            epochs=self.epochs,
            batch_num=self.batch_num,
            batch_size=self.batch_size,
            device=self.device,
        )

        return self

    def check_datasets_consistency(self, datasets):
        expected_features = None
        for dataset in datasets:
            for data, _ in dataset:
                if expected_features is None:
                    expected_features = data.shape[0]
                elif data.shape[0] != expected_features:
                    raise ValueError(
                        f"this dataset has {data.shape[0]} features, expected {expected_features}"
                    )

        return expected_features

    def meta_fit(self, train_datasets, ratio=None):
        if not train_datasets or len(train_datasets[0]) == 0:
            raise ValueError("train_datasets is empty or the first dataset is empty.")

        input_size = self.check_datasets_consistency(train_datasets)

        self.set_seed(self.seed)
        self.model = _MOEAnomalyDetection(
            input_size=input_size,
            num_experts=self.num_experts,
            expert_type=self.expert_type,
        )
        self.model.apply(self.init_weights)
        self.model.to(self.device)

        # Configs
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        meta_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.meta_lr, weight_decay=self.weight_decay
        )

        # Meta-training
        meta_fit(
            train_datasets=train_datasets,
            model=self.model,
            optimizer=optimizer,
            meta_optimizer=meta_optimizer,
            epochs=self.epochs,
            batch_num=self.batch_num,
            batch_size=self.batch_size,
            meta_batch_size=self.meta_batch_size,
            inner_update_steps=self.inner_update_steps,
            inner_lr=self.inner_lr,
            device=self.device,
            logs_interval=self.logs_interval,
        )

        return self

    def predict_score(self, X, num=30) -> np.ndarray:
        self.model = self.model.eval()

        X = X if torch.is_tensor(X) else torch.from_numpy(X)
        X = X.float().to(self.device)

        scores = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i : i + self.batch_size]

                batch_scores = self.model(batch_X)

                batch_scores = batch_scores.cpu().numpy()
                scores.extend(batch_scores)

        return np.array(scores)
