# ROBOREWARD-1M
#uv run scripts/generate_vqa_dataset.py \
#    --num_epochs 10.0 \
#    --output_path vqa_datasets/roboreward_train_10epochs \
#    --seed 42 \
#    --num_workers 32 \
#    --config_overrides data.max_frames=32 "data.sample_type_ratio=[1.0, 1.0, 0.0]" "data.train_datasets=[jesbu1_roboreward_rfm_roboreward_train]" data.min_frames_per_trajectory=1  data.dataset_type=strategy_first
#
## ROBOREWARD-Eval
#uv run scripts/generate_vqa_dataset.py \
#    --num_epochs 1 \
#    --output_path vqa_datasets/roboreward_val_0.1epoch \
#    --seed 42 \
#    --num_workers 32 \
#    --eval_mode \
#    --config_overrides data.max_frames=32 "data.sample_type_ratio=[1.0, 1.0, 0.0]" "data.eval_datasets=[jesbu1_roboreward_rfm_roboreward_val]" data.min_frames_per_trajectory=1 data.dataset_type=strategy_first


# ROBOREWARD-1M Prog Only
uv run scripts/generate_vqa_dataset.py \
    --num_epochs 10.0 \
    --output_path vqa_datasets/roboreward_train_10epochs_prog_only \
    --seed 42 \
    --num_workers 32 \
    --config_overrides data.max_frames=32 "data.sample_type_ratio=[0.0, 1.0, 0.0]" "data.train_datasets=[jesbu1_roboreward_rfm_roboreward_train]" data.min_frames_per_trajectory=1  data.dataset_type=strategy_first

# ROBOREWARD-Eval Prog Only
uv run scripts/generate_vqa_dataset.py \
    --num_epochs 1.0 \
    --output_path vqa_datasets/roboreward_val_1epoch_prog_only \
    --seed 42 \
    --num_workers 32 \
    --eval_mode \
    --config_overrides data.max_frames=32 "data.sample_type_ratio=[0.0, 1.0, 0.0]" "data.eval_datasets=[jesbu1_roboreward_rfm_roboreward_val]" data.min_frames_per_trajectory=1 data.dataset_type=strategy_first
