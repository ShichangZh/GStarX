import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf
from dataset import get_dataset, get_dataloader
from gnnNets import get_gnnNets
from utils import check_dir, get_logger, PlotUtils
from baselines.methods import GNNExplainer
from baselines.baselines_utils import evaluate_related_preds_list

IS_FRESH = False


@hydra.main(config_path="../config", config_name="config")
def pipeline(config):
    cwd = os.path.dirname(os.path.abspath(__file__))
    pwd = os.path.dirname(cwd)

    config.datasets.dataset_root = os.path.join(pwd, "datasets")
    config.models.gnn_saving_path = os.path.join(pwd, "checkpoints")
    config.explainers.explanation_result_path = os.path.join(cwd, "results")
    config.log_path = os.path.join(cwd, "log")

    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]

    explainer_name = config.explainers.explainer_name
    log_file = (
        f"{explainer_name}_{config.datasets.dataset_name}_{config.models.gnn_name}.log"
    )
    logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    logger.debug(OmegaConf.to_yaml(config))

    if torch.cuda.is_available():
        device = torch.device("cuda", index=config.device_id)
    else:
        device = torch.device("cpu")

    # bbbp warning
    dataset = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    if config.models.param.graph_classification:
        dataloader_params = {
            "batch_size": config.models.param.batch_size,
            "random_split_flag": config.datasets.random_split_flag,
            "data_split_ratio": config.datasets.data_split_ratio,
            "seed": config.datasets.seed,
        }
        loader = get_dataloader(dataset, **dataloader_params)
        test_indices = loader["test"].dataset.indices
        if config.datasets.data_explain_cutoff > 0:
            test_indices = test_indices[: config.datasets.data_explain_cutoff]

    else:
        node_indices_mask = (dataset.data.y != 0) * dataset.data.test_mask
        node_indices = torch.where(node_indices_mask)[0]

    model = get_gnnNets(
        input_dim=dataset.num_node_features,
        output_dim=dataset.num_classes,
        model_config=config.models,
    )

    state_dict = torch.load(
        os.path.join(
            config.models.gnn_saving_path,
            config.datasets.dataset_name,
            f"{config.models.gnn_name}_"
            f"{len(config.models.param.gnn_latent_dim)}l_best.pth",
        )
    )["net"]
    model.load_state_dict(state_dict)

    model.to(device)

    explanation_saving_path = os.path.join(
        config.explainers.explanation_result_path,
        config.datasets.dataset_name,
        config.models.gnn_name,
        explainer_name,
    )
    check_dir(explanation_saving_path)
    gnn_explainer = GNNExplainer(
        model,
        epochs=config.explainers.param.epochs,
        lr=config.explainers.param.lr,
        explain_graph=config.models.param.graph_classification,
    )

    gnn_explainer.device = device

    plot_utils = PlotUtils(config.datasets.dataset_name, is_show=False)
    related_preds_list = []
    for i, data in enumerate(tqdm(dataset[test_indices])):
        idx = test_indices[i]
        data.to(device)
        prediction = model(data).argmax(-1).item()
        example_path = os.path.join(explanation_saving_path, f"example_{idx}.pt")
        if not IS_FRESH and os.path.isfile(example_path):
            edge_masks = torch.load(os.path.join(example_path))
            edge_masks = [edge_mask.to(device) for edge_mask in edge_masks]
            logger.debug(f"Load example {idx}.")
            edge_masks, hard_edge_masks, related_preds = gnn_explainer(
                data.x,
                data.edge_index,
                sparsity=config.explainers.sparsity,
                num_classes=dataset.num_classes,
                edge_masks=edge_masks,
            )
        else:
            edge_masks, hard_edge_masks, related_preds = gnn_explainer(
                data.x,
                data.edge_index,
                sparsity=config.explainers.sparsity,
                num_classes=dataset.num_classes,
            )
            edge_masks = [edge_mask.to("cpu") for edge_mask in edge_masks]
            torch.save(edge_masks, example_path)

        related_preds = related_preds[prediction]
        hard_edge_masks = hard_edge_masks[prediction]
        related_preds_list += [related_preds]

        if config.save_plot:
            logger.debug(f"Plotting example {idx}.")
            from utils import fidelity_normalize_and_harmonic_mean, to_networkx
            from baselines.baselines_utils import hard_edge_masks2coalition

            coalition = hard_edge_masks2coalition(
                data, hard_edge_masks, config.models.param.add_self_loop
            )
            f = related_preds["origin"] - related_preds["maskout"]
            inv_f = related_preds["origin"] - related_preds["masked"]
            sp = related_preds["sparsity"]
            n_f, n_inv_f, h_f = fidelity_normalize_and_harmonic_mean(f, inv_f, sp)
            title_sentence = f"fide: {f:.3f}, inv-fide: {inv_f:.3f}, h-fide: {h_f:.3f}"

            if hasattr(dataset, "supplement"):
                words = dataset.supplement["sentence_tokens"][str(idx)]
            else:
                words = None

            explained_example_plot_path = os.path.join(
                explanation_saving_path, f"example_{idx}.png"
            )
            plot_utils.plot(
                to_networkx(data),
                coalition,
                x=data.x,
                words=words,
                title_sentence=title_sentence,
                figname=explained_example_plot_path,
            )

    metrics = evaluate_related_preds_list(related_preds_list, logger)
    metrics_str = ",".join([f"{m : .4f}" for m in metrics])
    print(metrics_str)


if __name__ == "__main__":
    import sys

    sys.argv.append("explainers=gnnexplainer")
    pipeline()
