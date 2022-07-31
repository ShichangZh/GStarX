<!-- #region -->
### Our code is heavily based on the DIG library, especially the baseline implementations. However, there are a few differences. Users may choose to run baselines by importing them from DIG directly. We summarize the difference of our code from DIG 0.1.2 below


- We used our own evaluation metric function instead of the original `XCollector` because we need to compute `norm-fidelity`, `norm-inv-fidelity`, and `h-fidelity` as described in the paper, and the XCollector doesn't allow that. It can be checked that the `fidelity`, `inv-fidelity`, and `sparsity` results are the same for ours and the `XCollector`.

- Added a new `remove` option for the subgraph building method, where the unselected nodes are directly removed.

- `mc_l_shapley` in `shapley.py`
    - The `mc_l_shapley` function is meant to do Monte Carlo (MC) sampling to approximate the Local-Shapley value. A number `sample_num` specifies how many samples should be generated. We added a branch so that when the total number of combinations is less than `sample_num`, the function directly compute the result rather than doing MC.

- Sparsity computation
    - The `eval_related_pred` function in `base_explainer.py` computes the sparsity of an explanation. However, the sparsity is computed as 1 - selected_edges / all_edges, which is different from the original definition of sparsity as 1 - selected_nodes / all_nodes. We thus correct this computation for a fair comparison between all models.
    - Also, for the `eval_related_pred` function in `base_explainer.py` and the `control_sparsity` called in `gnnexplainer.py` and `pgexplainer_edges.py`, the sparsity was computed using `hard_edge_masks`, which may include self-loops when the `add_self_loop` argument was set to be True for the GNN. Therefore, the sparsity needs to be normalized, other wise the computed sparsity will be much smaller (include more edges/nodes) than the real sparsity. We thus correct this computation for a fair comparison between all models.

- Others:
    - Changed a few variable names to make similar variables easier to distinguish
<!-- #endregion -->
