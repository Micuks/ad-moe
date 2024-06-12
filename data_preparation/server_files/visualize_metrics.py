import pandas as pd
import matplotlib.pyplot as plt


def plot_time_series(csv_file, time_col, data_col):
    data = pd.read_csv(csv_file)
    data[time_col] = pd.to_datetime(data[time_col])

    data.set_index(time_col, inplace=True)
