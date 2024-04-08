
#set text(
  font: "Songti SC"
)
#set table.hline(stroke: 0.6pt)
#set table(align: (x, _) => if x == 0 {left} else {right})
#show table.cell.where(x: 0): smallcaps
#table(
  columns: 3,
  stroke: none,
  table.header[*Model*][*ROC-AUC*][*Accuracy*],
  table.hline(y: 0, stroke: 1pt),
  table.hline(),
[IForest],[0.4512],[0.2886],
[CBLOF],[0.3833],[0.7114],
[COF],[0.5146],[0.7114],
[KNN],[0.3788],[0.7114],
[LOF],[0.3355],[0.7114],
[PCA],[0.3607],[0.7114],
[SOD],[0.5429],[0.7114],
[*MoE-MLP*(ours)],[*0.6495*],[*0.7114*],
table.hline(),
[GANomaly],[0.5],[0.7114],
[DeepSAD],[0.3803],[0.7114],
[REPEN],[0.4007],[0.7114],
[PReNet],[0.6495],[0.2886],
table.hline(),
[LR],[0.6467],[0.2886],
[NB],[0.4995],[0.7106],
[SVM],[0.5],[0.2886],
[MLP],[0.5],[0.2886],
[RF],[0.9991],[0.9957],
[LGB],[0.9671],[0.853],
[XGB],[0.9625],[0.8458],
[CatB],[0.9988],[0.9969],
[ResNet],[0.5],[0.2886],
 table.hline(stroke: 1pt),
)