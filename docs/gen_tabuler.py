# -*- encoding: utf-8 -*-
import io
import json
import subprocess


prefix = """
#set text(
  font: "Songti SC"
)
#table(
  columns: 5,
  table.header[*Model*][ROC-AUC][Accuracy][Recall][F1-Score],
"""

suffix = ")"
body = []
output_filename = "./output.typ"


f_round = lambda x: round(x, 4)
with io.open("./data.json", "r") as f_json:
    data = json.load(f_json)
    for slice in data:
        model_name, metrics, _, _, _ = slice
        print(slice)
        body.append(
            f"[{model_name}],[{f_round(metrics['test_auc'])}],[{f_round(metrics['test_acc'])}],[{f_round(metrics['test_recall'])}],[{f_round(metrics['test_f1'])}],\n"
        )

with io.open(output_filename, "w") as f:
    f.write(prefix)
    for line in body:
        f.write(line)
    f.write(suffix)

subprocess.run(["typst", "compile", f"{output_filename}"])
