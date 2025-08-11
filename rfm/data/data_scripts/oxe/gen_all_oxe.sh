OXE_VALID_DATASETS=(
    "austin_buds_dataset_converted_externally_to_rlds"
    "austin_sirius_dataset_converted_externally_to_rlds"
    "berkeley_cable_routing"
    "berkeley_fanuc_manipulation"
    "bc_z"
    "bridge_v2"
    "dlr_edan_shared_control_converted_externally_to_rlds"
    "droid"
    "fmb"
    "fractal20220817_data"
    "furniture_bench_dataset_converted_externally_to_rlds"
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds"
    "jaco_play"
    "language_table"
    "stanford_hydra_dataset_converted_externally_to_rlds"
    "taco_play"
    "toto"
    "ucsd_kitchen_dataset_converted_externally_to_rlds"
    "utaustin_mutex"
    "viola"
)

for dataset_name in ${OXE_VALID_DATASETS[@]}; do
    uv run rfm/data/generate_hf_dataset.py \
        --config_path=rfm/configs/data_gen_configs/oxe.yaml \
        --output.output_dir ~/scratch_data/oxe_rfm \
        --dataset.dataset_name $dataset_name
done