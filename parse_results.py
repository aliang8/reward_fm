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

vlac_results_file = "/home/azure/reward_fm/baseline_eval_output/vlac_InternRobotics_VLAC/"
roboreward_results_file = ""
rfm_results_file = ""

keys_to_remove = ["mit_franka", "libero", "metaworld", "racer", "roboreward", "roboarena"]

model_names = [
    "ReWiND",
    "VLAC-2B",
    "RoboReward-4B",
    "RoboReward-8B",
    "RFM-4B",
]

def load_and_filter_results(results_file: str, id_datasets: list, ood_datasets: list):
    """Load metrics.json and filter by ID and OOD datasets."""
    if not results_file:
        return {}, {}
    with open(os.path.join(results_file, "metrics.json"), "r") as f:
        results = json.load(f)
    return (
        {ds: results[ds] for ds in id_datasets if ds in results},
        {ds: results[ds] for ds in ood_datasets if ds in results},
    )

# Load results for each model
vlac_id_results, vlac_ood_results = load_and_filter_results(
    vlac_results_file, RFM_1M_ID_DATASETS, RFM_1M_OOD_DATASETS
)
roboreward_4b_id_results, roboreward_4b_ood_results = load_and_filter_results(
    roboreward_results_file, RFM_1M_ID_DATASETS, RFM_1M_OOD_DATASETS
)
rfm_4b_id_results, rfm_4b_ood_results = load_and_filter_results(
    rfm_results_file, RFM_1M_ID_DATASETS, RFM_1M_OOD_DATASETS
)

id_results = {
    "ReWiND": {},
    "VLAC-2B": vlac_id_results,
    "RoboReward-4B": roboreward_4b_id_results,
    "RoboReward-8B": {},
    "RFM-4B": rfm_4b_id_results,
}

ood_results = {
    "ReWiND": {},
    "VLAC-2B": vlac_ood_results,
    "RoboReward-4B": roboreward_4b_ood_results,
    "RoboReward-8B": {},
    "RFM-4B": rfm_4b_ood_results,
}

def render_terminal_table(
    id_results_by_model,
    ood_results_by_model,
    model_names,
):
    console = Console()

    table = Table(
        title="Granular VOC (Pearson r ↑)",
        show_lines=False,
        header_style="bold",
    )

    table.add_column("Split", justify="left")
    table.add_column("Dataset", justify="left")

    for m in model_names:
        table.add_column(m, justify="right")

    def add_block(split_name, results_by_model):
        # collect dataset order from first model
        datasets = list(next(iter(results_by_model.values())).keys())

        for ds in datasets:
            row = [split_name, ds]
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
                v for v in results_by_model[m].values()
                if isinstance(v, float)
            ]
            avg = sum(vals) / len(vals) if vals else "--"
            avg_row.append(f"{avg:.3f}" if avg != "--" else "--")

        table.add_row(*avg_row)

    add_block("RFM-ID", id_results_by_model)
    table.add_section()
    add_block("RFM-OOD", ood_results_by_model)

    console.print(table)

render_terminal_table(id_results, ood_results, model_names)