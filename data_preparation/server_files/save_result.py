# %%
import sys
import pickle
import os
import pandas as pd

global data_dir
data_dir = "../../data/ob"
os.makedirs(data_dir, exist_ok=True)

csv_file_path = ""


def save_file():
    print("filepath: ", csv_file_path)
    df = pd.read_csv(csv_file_path, skiprows=5)
    print("Convert timestamp to datetime")
    df["time"] = pd.to_datetime(df["time"], unit="s")

    data = []
    for _, row in df.iterrows():
        metrics = row.tolist()
        metrics = [metrics]
        assert len(metrics[0]) > 1
        formatted_row = [metrics, "knob_shared_buffers"]
        data.append(formatted_row)

    # Save as a pickle file
    pickle_filename = csv_file_path.split(".csv")[0] + ".pkl"
    pickle_file_path = os.path.join(data_dir, pickle_filename)
    with open(pickle_file_path, "wb") as file:
        pickle.dump(data, file)

    print(f"Data saved as pickle file: {pickle_file_path}")
    return df


# %%
def main():
    if len(sys.argv) < 2:
        print("Usage: save_resupt.py <csv_file_path>")
        sys.exit(1)
    global csv_file_path
    csv_file_path = sys.argv[1]
    if not os.path.isfile(csv_file_path):
        print(f"File {csv_file_path} does not exist")
        sys.exit(1)

    save_file().head()


if __name__ == "__main__":
    main()
