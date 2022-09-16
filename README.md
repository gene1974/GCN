# GCN

A pytorch implementation of GCN(Graph Convolution over Pruned Dependency Trees Improves Relation
Extraction).

During training, we use early stopping and use dev.json as cross validation set. During testing, we use the official scorer to get precision, recall, and F1.

Training：

```bash
python main.py train -model cgcn -nlayer 1
```

Test：

```bash
python main.py test -model cgcn -nlayer 1 -time {MODEL_TIME}
```

