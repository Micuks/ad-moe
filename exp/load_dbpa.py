import io
import os
import pandas as pd
import pickle as pkl
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] %(message)s"
)


class DBPA_Dataset:
    def __init__(self, anomaly_ratio=0.2) -> None:
        self.logger = logging.getLogger(__name__)
        self.anomaly_ratio = anomaly_ratio
        self.random_state = 42
        self.basepath = "./data/dbpa"

        self.train_pickles = [
            "small_shared_buffer_data",
            "io_saturation_data",
            "heavy_workload_data",
            "normal_data",
        ]

        self.test_pickles = [
            "small_shared_buffer_data",
            "io_saturation_data",
            "heavy_workload_data",
            "lock_waits_data",
            "missing_indexes_data",
            "too_many_indexes_data",
            "normal_data",
        ]

        self.train_specs = [
            "mysql_32_128",
        ]

        self.test_specs = [
            "mysql_32_128",
            "mysql_32_256",
        ]

        self._num_norm = lambda x: len(x[x[len(x.columns) - 1] == 0])
        self._num_anom = lambda x: len(x[x[len(x.columns) - 1] != 0])

    def _is_anomaly(self, label: str) -> bool:
        anomalies = ["knob_innodb_buffer"]
        normal = ["normal"]
        if label in normal:
            return False
        else:
            return True

    def _data_to_df(self, data: list) -> pd.DataFrame:
        df = []
        for d in data:
            Xs = d[0]
            y = d[1]

            for x in Xs:
                x = x.append(1 if self._is_anomaly(y) else 0)
            df.extend(Xs)

        df = pd.DataFrame(df)
        # WARNING: fill NaN with 0 instead of -1 or whatever
        df.iloc[:, -1] = df.iloc[:, -1].fillna(0).astype(int)

        return df

    def load_dataset(self, drop_zero_cols=False):
        basepath = self.basepath
        train_specs = self.train_specs
        test_specs = self.test_specs
        train_pickles = self.train_pickles
        test_pickles = self.test_pickles
        dfs_train = []
        dfs_test = []

        fname = lambda x: os.path.join(basepath, "/".join(x) + ".pickle")
        for spec in train_specs:
            for pickle in train_pickles:
                with io.open(fname([spec, pickle]), "rb") as f:
                    data = pkl.load(f)
                    df = self._data_to_df(data)
                    dfs_train.append(df)

        for spec in test_specs:
            for pickle in test_pickles:
                with io.open(fname([spec, pickle]), "rb") as f:
                    data = pkl.load(f)
                    df = self._data_to_df(data)
                    dfs_test.append(df)

        dfs_train_all = dfs_train
        dfs_test_all = dfs_test

        if drop_zero_cols:
            dfs_train_all, dfs_test_all = self._drop_zero_cols(
                dfs_train_all, dfs_test_all
            )

        # Dataset Construction: reconstruct dataset with repect to anomaly ratio
        anomaly_ratio = self.anomaly_ratio
        dfs_train_all = self._construct_dataset(
            dfs_train_all, anomaly_ratio=anomaly_ratio
        )
        dfs_test_all = self._construct_dataset(
            dfs_test_all, anomaly_ratio=anomaly_ratio
        )

        return dfs_train_all, dfs_test_all

    def _drop_zero_cols(
        self, dfs_train: pd.DataFrame, dfs_test: pd.DataFrame
    ) -> list[pd.DataFrame, pd.DataFrame]:
        zero_cols = dfs_train[0].columns[(dfs_train[0] == 0).all()]
        print(len(zero_cols))

        # drop those
        dfs_train = [df.drop(zero_cols, axis=1) for df in dfs_train]
        dfs_test = [df.drop(zero_cols, axis=1) for df in dfs_test]

        return dfs_train, dfs_test

    def _drop_nans(
        self, dfs_train: pd.DataFrame, dfs_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Drop columns that are entirely NaN and fill remaining NaNs with column means
        """
        nan_cols_union = set(dfs_train.columns[dfs_train.isnull().all()]).union(
            set(dfs_test.columns[dfs_test.isnull().all()])
        )

        if nan_cols_union:
            self.logger.info(f"NaN Columns to be dropped: {nan_cols_union}")

        dfs_train.drop(columns=nan_cols_union, inplace=True)
        dfs_test.drop(columns=nan_cols_union, inplace=True)

        # fill remaining NaNs with column means
        dfs_train.fillna(dfs_train.mean(), inplace=True)
        dfs_test.fillna(dfs_test.mean(), inplace=True)
        self.logger.info("Filled remaining NaNs with column means.")

        return dfs_train, dfs_test

    def _construct_dataset(self, dfs: list, anomaly_ratio=0.2) -> list:
        new_dfs = []
        for df in dfs:
            num_normal = self._num_anom(df)
            required_num_anomaly = int(
                num_normal * (anomaly_ratio / (1 - anomaly_ratio))
            )

            normal_samples = df[df[len(df.columns) - 1] == 0]
            anomaly_labels = df[df[len(df.columns) - 1] != 0].sample(
                n=required_num_anomaly, random_state=self.random_state
            )

            new_df = pd.concat([normal_samples, anomaly_labels]).sample(
                frac=1, random_state=self.random_state, axis=0
            )
            # Remove timestamp column, and name label column as label
            new_df.drop(df.columns[0], axis=1, inplace=True)
            new_df.columns = [*new_df.columns[:-1], "label"]

            new_dfs.append(new_df)

        return new_dfs
