import torch
import numpy as np
import random
from torch import nn
from mymodel.model import _MOEAnomalyDetection
from mymodel.fit import fit, meta_fit


class MoEAnomalyDetection:
    def __init__(
        self,
        seed: int,
        model_name="MLP",
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
    ):
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        input_size = X_train.shape[1]
        self.X_train_tensor = torch.from_numpy(X_train).float()
        self.y_train = y_train

        self.set_seed(self.seed)
        self.model = _MOEAnomalyDetection(
            input_size=input_size, num_experts=self.num_experts
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

    def meta_fit(self, train_datasets, ratio=None):
        if not train_datasets or len(train_datasets[0]) == 0:
            raise ValueError("train_datasets is empty or the first dataset is empty.")

        sample_data, _ = train_datasets[0][0]
        input_size = sample_data.shape[0]

        self.set_seed(self.seed)
        self.model = _MOEAnomalyDetection(
            input_size=input_size, num_experts=self.num_experts
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
        )

        return self

    def predict_score(self, X, num=30):
        self.model = self.model.eval()

        X = X if torch.is_tensor(X) else torch.from_numpy(X)
        X = X.float().to(self.device)

        score = []

        with torch.no_grad():
            for i in range(len(X)):
                index_normal = np.random.choice(
                    np.where(self.y_train == 1)[0], num, replace=True
                )
                index_anomaly = np.random.choice(
                    np.where(self.y_train == 0)[0], num, replace=True
                )
                X_train_normal = self.X_train_tensor[index_normal].to(self.device)
                X_train_anomaly = self.X_train_tensor[index_anomaly].to(self.device)

                batch_score = self.model(torch.cat((X_train_normal, X_train_anomaly)))

                batch_score = torch.mean(batch_score).cpu()
                batch_score = batch_score.numpy()[()]

                score.append(batch_score)

        return np.array(score)
