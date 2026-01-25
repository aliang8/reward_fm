import ast
import os
import json
from rfm.data.datasets.name_mapping_final import DS_SHORT_NAME_MAPPING
from rfm.data.dataset_category import DATASET_MAP
from rich.console import Console
from rich.table import Table

LIBERO_PI0 = "libero_pi0"
LIBERO_PI0_DATASETS = DATASET_MAP[LIBERO_PI0]["eval"]

RFM_1M_ID = "rfm-1m-id"
RFM_1M_OOD = "rfm-1m-ood"
RFM_1M_ID_DATASETS = DATASET_MAP[RFM_1M_ID]["eval"]
RFM_1M_OOD_DATASETS = DATASET_MAP[RFM_1M_OOD]["eval"]

DATASETS = RFM_1M_ID_DATASETS + RFM_1M_OOD_DATASETS

# model_to_results_dir = {
#     "LIBERO Prog Only": "/gpfs/home/jessezha/scrubbed_storage/reward_fm/baseline_eval_output/rfm_libero_ablation_prog_only_lora_ft_4frames_2000steps_ckpt-avg-3metrics=0_6280_step=1000",
#     "LIBERO Pref Prog": "/gpfs/home/jessezha/scrubbed_storage/reward_fm/baseline_eval_output/rfm_libero_ablation_progpref_lora_ft_4frames_2000steps_ckpt-avg-3metrics=0_6809_step=450",
#     "RobotRFM": "/gpfs/home/jessezha/scrubbed_storage/reward_fm/baseline_eval_output/rfm_libero_ablation_prog_pref_with_fail_lora_ft_4frames_2000steps_ckpt-avg-3metrics=0_7650_step=700"
# }

model_to_results_dir = {
    "RFM-4B Prog Only": "/gpfs/home/jessezha/scrubbed_storage/reward_fm/baseline_eval_output/rfm_rfm_1m_ablation_prog_only_8frames_ckpt-latest-avg-2metrics=0_6650_step=5750",
    "RFM-4B Prog Pref": "/gpfs/home/jessezha/scrubbed_storage/reward_fm/baseline_eval_output/rfm_rfm_1m_ablation_prog_pref_8frames_ckpt-avg-2metrics=0_7977_step=2250",
    "RFM-4B-8frames": "/gpfs/home/jessezha/scrubbed_storage/reward_fm/baseline_eval_output/rfm_aliangdw_qwen4b_pref_prog_succ_8_frames_all/",
}

model_names = list(model_to_results_dir.keys())

metric_to_keys = {
    "reward_alignment": ["pearson"],
    "policy_ranking": ["kendall_last", "avg_succ_fail_diff_last"],
}

# eval_types = ["reward_alignment", "policy_ranking"]
# eval_types = ["policy_ranking"]
eval_types = ["reward_alignment"]

def dataset_key_matches(key: str, datasets: list) -> bool:
    """Check if a metrics key matches any of the datasets (handles both string and list datasets)."""
    key_ds = key.split("/")[0]
    for ds in datasets:
        # If ds is a list, compare against its string representation
        if isinstance(ds, list):
            if key_ds == str(ds):
                return True
        else:
            if key_ds == ds:
                return True
    return False

def get_dataset_display_name(ds_key: str) -> str:
    """Get a display name for a dataset key (handles both string and list-style keys)."""
    # Check if it's a stringified list like "['ds1', 'ds2']"
    if ds_key.startswith("[") and ds_key.endswith("]"):
        # Parse the list and get the first dataset's short name
        try:
            ds_list = ast.literal_eval(ds_key)
            if ds_list and isinstance(ds_list, list):
                # Use the first dataset's short name
                return DS_SHORT_NAME_MAPPING.get(ds_list[0], ds_list[0])
        except (ValueError, SyntaxError):
            pass
    return DS_SHORT_NAME_MAPPING.get(ds_key, ds_key)

def load_and_filter_results(results_file: str, datasets: list, eval_type: str):
    """Load metrics.json and filter by datasets."""
    print("Loading results for ", eval_type, "from", results_file)
    if not results_file:
        print("No results file found for ", eval_type, "from", results_file)
        return {}
    with open(os.path.join(results_file, eval_type, "metrics.json"), "r") as f:
        results = json.load(f)

    print("Loaded results for ", eval_type, "from", results_file)
    return {key: value for key, value in results.items() if dataset_key_matches(key, datasets)}

# Load results for each model
model_to_results = {model: {} for model in model_names}

for eval_type in eval_types:
    for model in model_names:
        model_results_file = model_to_results_dir[model]
        model_to_results[model][eval_type] = load_and_filter_results(
            model_results_file, DATASETS, eval_type=eval_type
        )
        
def render_terminal_table(
    model_to_results,
    eval_type,
    model_names,
    metric_keys: list
):
    # Extract results for the given eval_type
    results_by_model = {m: model_to_results[m][eval_type] for m in model_names}

    console = Console()

    table = Table(
        title=f"{eval_type.replace('_', ' ').title()}",
        show_lines=False,
        header_style="bold",
    )

    # Collect all unique datasets from all models
    all_keys = set()
    for model_results in results_by_model.values():
        all_keys.update(model_results.keys())
    
    # Extract unique dataset prefixes
    datasets = []
    for key in all_keys:
        ds = key.split("/")[0]
        if ds not in datasets:
            datasets.append(ds)

    # Model column
    table.add_column("Model", justify="left")

    # Add columns for each dataset + metric combination
    for ds in datasets:
        ds_display = get_dataset_display_name(ds)
        for metric_key in metric_keys:
            table.add_column(f"{ds_display}\n{metric_key}", justify="right")

    # Add rows for each model
    for m in model_names:
        row = [m]
        for ds in datasets:
            for metric_key in metric_keys:
                # Find the key that matches this dataset and metric
                full_key = f"{ds}/{metric_key}"
                v = results_by_model[m].get(full_key, "--")
                row.append(f"{v:.3f}" if isinstance(v, float) else "--")
        table.add_row(*row)

    console.print(table)

for idx, eval_type in enumerate(eval_types):
    metric_keys = metric_to_keys[eval_type]
    print(f"Rendering {eval_type.replace('_', ' ').title()} ({', '.join(metric_keys)})")
    render_terminal_table(model_to_results, eval_type, model_names, metric_keys=metric_keys)