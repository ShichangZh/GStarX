<!-- #region -->
## GNN Explanation Baseline Implementations

### Usage
- Run a baslineline method to explain trained GNN models
  - A simple example, replace method with `gnn_explainer`, `pgexplainer`, `subgraphx`, `graphsvx`, or `orphicx`.
```bash
python baselines/run_method.py models='gcn' datasets='bace'
```

### Remark
Our baseline code is heavily based on the DIG library with small changes. We summarize the difference in `difference_from_dig.md`

### Reference
The `gnn_explainer`, `pgexplainer`, and `subgraphx` implementations are based on the DIG library.

https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph

The `graphsvx` implementaiton is based on the official GraphSVX code, where datasets processing and GNNs are changed to align with DIG.

https://github.com/AlexDuvalinho/GraphSVX

The `orphicx` implementaiton is based on the official OrphicX code, where datasets processing and GNNs are changed to align with DIG.

https://github.com/WanyuGroup/CVPR2022-OrphicX
<!-- #endregion -->
