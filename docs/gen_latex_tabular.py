import json
import os
import io

# Sample JSON data as a list of entries
data = [
    [
        "IForest",
        {
            "val_auc": 0.9544715447154472,
            "val_acc": 0.7321428571428571,
            "val_recall": 0.0,
            "val_f1": 0.0,
            "test_auc": 0.4511637548449207,
            "test_acc": 0.2886416861826698,
            "test_recall": 0.0,
            "test_f1": 0.0,
        },
        0.2879786491394043,
        0.03004312515258789,
        0.3106415271759033,
    ],
    [
        "CBLOF",
        {
            "val_auc": 0.5967479674796747,
            "val_acc": 0.26785714285714285,
            "val_recall": 1.0,
            "val_f1": 0.4225352112676056,
            "test_auc": 0.3832513905235715,
            "test_acc": 0.7113583138173302,
            "test_recall": 1.0,
            "test_f1": 0.8313376667807048,
        },
        2.4187216758728027,
        0.13985180854797363,
        0.10178685188293457,
    ],
]

# Start building the LaTeX tabular
latex_table = r"""\begin{table}[htbp!]
\centering
\caption{Test Metrics of Various Models}
\label{tab:test_metrics}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{AUC} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{Recall} \\
\midrule
"""

in_filename = "../results/data_baselines_dbpa.json"
output_filename = "./output_" + os.path.basename(in_filename) + ".txt"
f_round = lambda x: round(x, 4)

with io.open(in_filename, "r") as f_json:
    data = json.load(f_json)

    # Iterate over each entry in the JSON data
    for entry in data:
        model = entry[0]
        metrics = entry[1]
        # Extract the required test metrics
        test_auc = f_round(metrics.get("test_auc", 0))
        test_acc = f_round(metrics.get("test_acc", 0))
        test_f1 = f_round(metrics.get("test_f1", 0))
        test_recall = f_round(metrics.get("test_recall", 0))

        # Add the model and its metrics to the table
        latex_table += f"{model} & {test_auc:.4f} & {test_acc:.4f} & {test_f1:.4f} & {test_recall:.4f} \\\\\n"

    # End the table
    latex_table += r"""\bottomrule
\end{tabular}
\end{table}
    """

    # Print the LaTeX table
    print(latex_table)
    with io.open(output_filename, "w") as f:
        f.write(latex_table)
