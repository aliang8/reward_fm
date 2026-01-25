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

model_to_results_dir = {
    "GVL": "/gpfs/home/jessezha/scrubbed_storage/reward_fm/baseline_eval_output/gvl",
    "ReWiND": "",
    # "VLAC-2B": "/home/azure/reward_fm/baseline_eval_output/vlac_InternRobotics_VLAC/",
    "VLAC-2B": "/gpfs/home/jessezha/scrubbed_storage/reward_fm/baseline_eval_output/vlac_InternRobotics_VLAC",
    "RoboReward-4B": "/gpfs/home/jessezha/scrubbed_storage/reward_fm/baseline_eval_output/roboreward_teetone_RoboReward-4B copy/",
    "RoboReward-8B": "/gpfs/home/jessezha/scrubbed_storage/reward_fm/baseline_eval_output/roboreward_teetone_RoboReward-8B copy/",
    "RFM-4B Prog Only": "/gpfs/home/jessezha/scrubbed_storage/reward_fm/baseline_eval_output/rfm_rfm_1m_ablation_prog_only_8frames_ckpt-latest-avg-2metrics=0_6650_step=5750",
    "RFM-4B Prog Pref": "/gpfs/home/jessezha/scrubbed_storage/reward_fm/baseline_eval_output/rfm_rfm_1m_ablation_prog_pref_8frames_ckpt-avg-2metrics=0_7977_step=2250",
    # "RFM-4B-All": "/gpfs/home/jessezha/scrubbed_storage/reward_fm/baseline_eval_output/rfm_rewardfm_rfm_qwen_pref_prog_4frames_all_strategy/",
    "RFM-4B-8frames": "/gpfs/home/jessezha/scrubbed_storage/reward_fm/baseline_eval_output/rfm_aliangdw_qwen4b_pref_prog_succ_8_frames_all/",
}

model_names = list(model_to_results_dir.keys())

metric_to_key = {
    "reward_alignment": "pearson",
    "policy_ranking": "kendall_last",
}

# eval_types = ["reward_alignment", "policy_ranking"]
eval_types = ["policy_ranking"]

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

for eval_type in eval_types:
    for model in model_names:
        model_results_file = model_to_results_dir[model]
        model_to_results[model][eval_type] = load_and_filter_results(
            model_results_file, RFM_1M_ID_DATASETS, RFM_1M_OOD_DATASETS, eval_type=eval_type
        )
        
def render_terminal_table(
    model_to_results,
    eval_type,
    model_names,
    metric_key : str = "pearson"
):
    # Extract id and ood results for the given eval_type
    id_results_by_model = {m: model_to_results[m][eval_type][0] for m in model_names}
    ood_results_by_model = {m: model_to_results[m][eval_type][1] for m in model_names}

    console = Console()

    table = Table(
        title=f"{eval_type.capitalize()} ({metric_key.capitalize()} ↑)",
        show_lines=False,
        header_style="bold",
    )

    table.add_column("Split", justify="left")
    table.add_column("Dataset", justify="left")

    for m in model_names:
        table.add_column(m, justify="right")

    def add_block(split_name, results_by_model):
        results_by_model = {m: {k: v for k, v in results_by_model[m].items() if metric_key in k} for m in model_names}

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

for idx, eval_type in enumerate(eval_types):
    metric_key = metric_to_key[eval_type]
    print(f"Rendering {eval_type.capitalize()} ({metric_key.capitalize()} ↑)")
    render_terminal_table(model_to_results, eval_type, model_names, metric_key=metric_key)