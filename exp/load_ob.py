import os
import pandas as pd


class OceanBase_Dataset:
    def __init__(self, anomaly_ratio=0.2) -> None:
        self.anomaly_ratio = anomaly_ratio
        self.random_state = 42
        self.basepath = "./data"

        self.train_csvs = [
            # "memory_limit_merge_label",
            # "io_saturation_merge_ob_2024-03-0706_11_36UTC_label",
            "io_saturation_merge_ob_2024-03-0717_44_47UTC_label",
            # "memory_limit_merge_ob_2024-03-0508_09_42UTC_label",
            # "memory_limit_merge_ob_2024-03-0508_56_01UTC_label",
            # "memory_limit_merge_ob_2024-03-0509_42_19UTC_label",
            "memory_limit_merge_ob_2024-03-0510_28_34UTC_label",
            "memory_limit_merge_ob_2024-03-0511_14_49UTC_label",
            "memory_limit_4_16_merge_ob_2024-03-1016_04_40UTC_label",
            "memory_limit_4_16_merge_ob_2024-03-1017_05_55UTC_label",
        ]

        self.test_csvs = [
            # "memory_limit_merge_label",
            "io_saturation_merge_ob_2024-03-0706_11_36UTC_label",
            "io_saturation_merge_ob_2024-03-0717_44_47UTC_label",
            # "memory_limit_merge_ob_2024-03-0508_09_42UTC_label",
            "memory_limit_merge_ob_2024-03-0508_56_01UTC_label",
            "memory_limit_merge_ob_2024-03-0509_42_19UTC_label",
            "memory_limit_merge_ob_2024-03-0510_28_34UTC_label",
            "memory_limit_merge_ob_2024-03-0511_14_49UTC_label",
            "memory_limit_4_16_merge_ob_2024-03-1016_04_40UTC_label",
            "memory_limit_4_16_merge_ob_2024-03-1017_05_55UTC_label",
        ]

        self.train_tasks = [
            # "io_saturation_merge",
            "memory_limit_merge",
            "memory_limit_4_16_merge",
        ]

        self.test_tasks = [
            "io_saturation_merge",
            "memory_limit_merge",
            "memory_limit_4_16_merge",
        ]

        self.num_norm = lambda x: len(x[x["label"] == 0])
        self.num_anom = lambda x: len(x[x["label"] != 0])

    def _get_dfs(self, tasks: list):
        return [[] for _ in range(len(tasks))]

    def load_dataset(self, drop_zero_cols=False) -> tuple[list, list]:
        basepath = self.basepath
        train_csvs = self.train_csvs
        test_csvs = self.test_csvs
        train_tasks = self.train_tasks
        test_tasks = self.test_tasks

        dfs_train = self._get_dfs(train_tasks)
        dfs_test = self._get_dfs(test_tasks)

        f = lambda x: os.path.join(basepath, x + ".csv")
        for csv in train_csvs:
            df = pd.read_csv(f(csv))
            for i, task in enumerate(train_tasks):
                if task in csv:
                    dfs_train[i].append(df)
                    break

        for csv in test_csvs:
            df = pd.read_csv(f(csv))
            for i, t in enumerate(test_tasks):
                if t in csv:
                    dfs_test[i].append(df)
                    break

        dfs_train_all = [pd.concat(dfs, axis=0) for dfs in dfs_train]
        dfs_test_all = [pd.concat(dfs, axis=0) for dfs in dfs_test]
        print(
            f"dfs_train_all[{[df.shape for df in dfs_train_all]}], normal[{[self.num_norm(df) for df in dfs_train_all]}], anomaly[{[self.num_anom(df) for df in dfs_train_all]}]"
        )
        print(
            f"dfs_test_all[{[df.shape for df in dfs_test_all]}], normal[{[self.num_norm(df) for df in dfs_test_all]}], anomaly[{[self.num_anom(df) for df in dfs_test_all]}]"
        )

        if drop_zero_cols:
            # # Data cleaning
            # ## Remove irrelevant data
            # Remove those columns whose data never change

            dfs_train_all, dfs_test_all = self._drop_zero_cols(
                dfs_train_all, dfs_test_all
            )

        # # Dataset construction
        #
        # Reconstruct dataset with respect to anomaly ratio

        dfs_train_all = self.construct_dataset(
            dfs_train_all, anomaly_ratio=self.anomaly_ratio
        )
        dfs_test_all = self.construct_dataset(dfs_test_all, anomaly_ratio=0.5)

        print(
            f"dfs_train_all[{[df.shape for df in dfs_train_all]}], normal[{[self.num_norm(df) for df in dfs_train_all]}], anomaly[{[self.num_anom(df) for df in dfs_train_all]}]"
        )
        print(
            f"dfs_test_all[{[df.shape for df in dfs_test_all]}], normal[{[self.num_norm(df) for df in dfs_test_all]}], anomaly[{[self.num_anom(df) for df in dfs_test_all]}]"
        )

        return dfs_train_all, dfs_test_all

    # FIXME: Can this really able to drop all zero columns?
    def _drop_zero_cols(
        self, dfs_train_all, dfs_test_all
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Find those columns in df_all that are always zero
        zero_cols = dfs_train_all[0].columns[(dfs_train_all[0] == 0).all()]
        print(len(zero_cols))

        # Drop those cols
        dfs_train_all = [df.drop(zero_cols, axis=1) for df in dfs_train_all]
        dfs_test_all = [df.drop(zero_cols, axis=1) for df in dfs_test_all]

        return dfs_train_all, dfs_test_all


def _construct_dataset(self, dfs, anomaly_ratio=0.1):
    new_dfs = []
    for df in dfs:
        num_normal = self.num_anom(df)
        required_num_anomaly = int(num_normal * (anomaly_ratio / (1 - anomaly_ratio)))

        normal_samples = df[df["label"] == 0]
        anomaly_samples = df[df["label"] != 0].sample(
            n=required_num_anomaly, random_state=self.random_state
        )

        new_dfs.append(
            pd.concat([normal_samples, anomaly_samples]).sample(
                frac=1, random_state=self.random_state, axis=0
            )
        )

    return new_dfs
