# -*- encoding: utf-8 -*-
import io
import json
import subprocess


prefix = """
#set text(
  font: "Songti SC"
)
#set table.hline(stroke: 0.6pt)
#set table(align: (x, _) => if x == 0 {left} else {right})
#show table.cell.where(x: 0): smallcaps
#figure(
  caption: [*Model* ROC-AUC and Accuracy on the *\*What\** dataset.],
  table(
    columns: 3,
    stroke: none,
    table.header[*Model*][*ROC-AUC*][*Accuracy*],
    table.hline(y: 0, stroke: 1pt),
    table.hline(),
)
"""

suffix = """    table.hline(stroke: 1pt),
)
"""
body = []
output_filename = "./output.typ"


f_round = lambda x: round(x, 4)
with io.open("./data1.json", "r") as f_json:
    data = json.load(f_json)
    for slice in data:
        model_name, metrics, _, _, _ = slice
        print(slice)
        body.append(
            f"[{model_name}],[{f_round(metrics['test_auc'])}],[{f_round(metrics['test_acc'])}],\n"
        )

with io.open(output_filename, "w") as f:
    f.write(prefix)
    for line in body:
        f.write(line)
    f.write(suffix)

subprocess.run(["typst", "compile", f"{output_filename}"])
