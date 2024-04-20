from load_ob import OceanBase_Dataset
from load_dbpa import DBPA_Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch


def prepare_data_vanilla():
    # dataset = OceanBase_Dataset(anomaly_ratio=0.2)
    dataset = DBPA_Dataset(anomaly_ratio=0.2)
    # dfs_train_all, dfs_test_all = ob_dataset.load_dataset(drop_zero_cols=True)
    dfs_train_all, dfs_test_all = dataset.load_dataset(drop_zero_cols=False)

    # First test those models without meta-learning

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

    return (X_train, y_train, X_val, y_val, X_test, y_test)


class TaskSpecificDataset(Dataset):
    def __init__(self, df) -> None:
        self.data = df.drop(columns=["label"]).to_numpy()
        self.labels = df["label"].to_numpy()
        self.data = torch.from_numpy(self.data)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def prepare_data_meta_training():
    """
    Task-specific meta-training
    """
    dataset = DBPA_Dataset(anomaly_ratio=0.2)
    dfs_train_all, dfs_test_all = dataset.load_dataset()
    train_datasets = [TaskSpecificDataset(df) for df in dfs_train_all]
    test_datasets = [TaskSpecificDataset(df) for df in dfs_test_all]

    return (train_datasets, test_datasets)
