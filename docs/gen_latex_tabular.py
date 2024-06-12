import json
import os
import io

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

in_filename = "../results/data2024-05-06 21:57:31.115958.json"
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
