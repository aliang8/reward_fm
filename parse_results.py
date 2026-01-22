import os
import json
from rfm.data.datasets.name_mapping_final import DS_SHORT_NAME_MAPPING
from rfm.data.dataset_category import DATASET_MAP
from rich.console import Console
from rich.table import Table

RFM_1M_ID = "rfm-1m-id"
RFM_1M_OOD = "rfm-1m-ood"
RFM_1M_ID_DATASETS = DATASET_MAP[RFM_1M_ID]["eval"]
RFM_1M_OOD_DATASETS = DATASET_MAP[RFM_1M_OOD]["eval"]

keys_to_remove = ["mit_franka", "libero", "metaworld", "racer", "roboreward", "roboarena"]

model_names = [
    "ReWiND",
    "VLAC-2B",
    "RoboReward-4B",
    "RoboReward-8B",
    "RFM-4B",
]

model_to_results_dir = {
    "ReWiND": "",
    "VLAC-2B": "/home/azure/reward_fm/baseline_eval_output/vlac_InternRobotics_VLAC/",
    "RoboReward-4B": "",
    "RoboReward-8B": "",
    "RFM-4B": "",
}

def load_and_filter_results(results_file: str, id_datasets: list, ood_datasets: list, eval_type: str):
    """Load metrics.json and filter by ID and OOD datasets."""
    print("Loading results for ", eval_type, "from", results_file)
    if not results_file:
        return {}, {}
    with open(os.path.join(results_file, eval_type, "metrics.json"), "r") as f:
        results = json.load(f)

    return (
        {key: value for key, value in results.items() if key.split("/")[0] in id_datasets},
        {key: value for key, value in results.items() if key.split("/")[0] in ood_datasets},
    )

# Load results for each model
model_to_results = {model: {} for model in model_names}

for eval_type in ["reward_alignment", "policy_ranking"]:
    for model in model_names:
        model_results_file = model_to_results_dir[model]
        model_to_results[model][eval_type] = load_and_filter_results(
            model_results_file, RFM_1M_ID_DATASETS, RFM_1M_OOD_DATASETS, eval_type=eval_type
        )
        
def render_terminal_table(
    model_to_results,
    eval_type,
    model_names,
):
    # Extract id and ood results for the given eval_type
    id_results_by_model = {m: model_to_results[m][eval_type][0] for m in model_names}
    ood_results_by_model = {m: model_to_results[m][eval_type][1] for m in model_names}

    console = Console()

    table = Table(
        title="VOC (Pearson r ↑)",
        show_lines=False,
        header_style="bold",
    )

    table.add_column("Split", justify="left")
    table.add_column("Dataset", justify="left")

    for m in model_names:
        table.add_column(m, justify="right")

    def add_block(split_name, results_by_model):
        results_by_model = {m: {k: v for k, v in results_by_model[m].items() if "pearson" in k or "kendall_rewind_last" in k} for m in model_names}

        # collect all unique datasets from all models
        datasets = []
        for model_results in results_by_model.values():
            for ds in model_results.keys():
                if ds not in datasets:
                    datasets.append(ds)

        for ds in datasets:
            row = [split_name, DS_SHORT_NAME_MAPPING[ds.split("/")[0]]]
            for m in model_names:
                v = results_by_model[m].get(ds, "--")
                row.append(f"{v:.3f}" if isinstance(v, float) else "--")
            table.add_row(*row)
            split_name = ""  # only show once

        # separator before average
        table.add_section()

        avg_row = ["", "Average"]
        for m in model_names:
            vals = [
                v for k, v in results_by_model[m].items()
                if isinstance(v, float)
            ]
            avg = sum(vals) / len(vals) if vals else "--"
            avg_row.append(f"{avg:.3f}" if avg != "--" else "--")

        table.add_row(*avg_row)

    add_block("RFM-ID", id_results_by_model)
    table.add_section()
    add_block("RFM-OOD", ood_results_by_model)

    console.print(table)

render_terminal_table(model_to_results, "reward_alignment", model_names)

render_terminal_table(model_to_results, "policy_ranking", model_names)