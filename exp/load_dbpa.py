import io
from sklearn.preprocessing import StandardScaler
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

        self.label_col = lambda x: x.iloc[:, -1:]
        self._num_norm = lambda x: len(x[x.iloc[:, -1:] == 0])
        self._num_anom = lambda x: len(x[x.iloc[:, -1:] != 0])

    def _mark_label(self, label: str) -> int:
        anomalies = [
            "[109]",
            "[115]",
            "[117]",
            "[118]",
            "[126]",
            "[171]",
            "[26]",
            "[5, 17]",
            "[5, 18]",
            "[50]",
            "[51]",
            "[53]",
            "[54]",
            "[56]",
            "[6, 18]",
            "[6, 19]",
            "[6, 20]",
            "[6, 21]",
            "[6, 22]",
            "[6, 23]",
            "[6, 24]",
            "[6, 25]",
            "[6, 26]",
            "[6, 27]",
            "[6, 28]",
            "[6, 29]",
            "[6, 30]",
            "[6, 36]",
            "[6, 48]",
            "[69]",
            "[7, 19]",
            "[73]",
            "[76]",
            "[77]",
            "knob_innodb_buffer",
            "manyindex",
        ]
        normal = ["normal"]
        # if str(label) in normal:
        #     return 0
        # else:
        #     return anomalies.index(str(label)) + 1
        if str(label) in normal:
            return 0
        else:
            return 1

    def _data_to_df(self, data: list) -> pd.DataFrame:
        df = []
        for d in data:
            Xs = d[0]
            y = d[1]

            for x in Xs:
                x = x.append(self._mark_label(y))
            df.extend(Xs)

        df = pd.DataFrame(df)
        # WARNING: fill NaN with 0 instead of -1 or whatever
        df.iloc[:, -1] = df.iloc[:, -1].fillna(0).astype(int)

        return df

    def load_dataset(self, drop_zero_cols=False, scaler_transform=True, subset=1):
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

        common_columns = set(dfs_train[0].columns)
        for df in dfs_train + dfs_test:
            common_columns.intersection_update(df.columns)

        dfs_train = [df[list(common_columns)] for df in dfs_train]
        dfs_test = [df[list(common_columns)] for df in dfs_test]

        if subset:
            dfs_train = [
                df.sample(frac=subset, random_state=self.random_state)
                for df in dfs_train
            ]
            dfs_test = [
                df.sample(frac=subset, random_state=self.random_state)
                for df in dfs_test
            ]

        # # Drop Timestamp
        # dfs_train_notimestamp = [
        #     df.drop(df.columns[0], axis=1, inplace=False) for df in dfs_train
        # ]
        # dfs_test_notimestamp = [
        #     df.drop(df.columns[0], axis=1, inplace=False) for df in dfs_test
        # ]

        # if scaler_transform:
        #     combined_train = pd.concat(dfs_train_notimestamp, ignore_index=True, axis=0)

        #     scaler = StandardScaler()
        #     scaler.fit(combined_train)

        #     dfs_train_all = [
        #         pd.DataFrame(scaler.transform(df), columns=df.columns)
        #         for df in dfs_train_notimestamp
        #     ]
        #     dfs_test_all = [
        #         pd.DataFrame(scaler.transform(df), columns=df.columns)
        #         for df in dfs_test_notimestamp
        #     ]
        # else:
        #     dfs_train_all = dfs_train_notimestamp
        #     dfs_test_all = dfs_test_notimestamp

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
        # Aggregate all normal data
        normal_data_pool = pd.concat(
            [df[df.iloc[:, -1] == 0] for df in dfs], axis=0
        ).sample(frac=1, random_state=self.random_state)

        for df in dfs:
            if df[df.iloc[:, -1] != 0].empty:
                continue

            num_anomaly = self._num_anom(df)
            required_num_normal = int(num_anomaly * (1 - anomaly_ratio) / anomaly_ratio)

            if required_num_normal > len(normal_data_pool):
                normal_samples = normal_data_pool.sample(
                    n=required_num_normal,
                    replace=True,
                    random_state=self.random_state,
                )
            else:
                normal_samples = normal_data_pool.sample(
                    n=required_num_normal,
                    replace=False,
                    random_state=self.random_state,
                )

            combined_df = pd.concat([df[df.iloc[:, -1] != 0], normal_samples]).sample(
                frac=1, random_state=self.random_state
            )
            if len(combined_df.columns) <= 1:
                raise ValueError("Insufficient feature columns retained for training")
            # Drop timestamp, rename label
            combined_df = combined_df.drop(columns=[combined_df.columns[0]])
            combined_df.columns = [*combined_df.columns[:-1], "label"]

            new_dfs.append(combined_df)

        return new_dfs
