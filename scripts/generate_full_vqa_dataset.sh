# RFM-10M
uv run scripts/generate_vqa_dataset.py \
    --num_epochs 10 \
    --output_path vqa_datasets/rfm_train_10epochs \
    --seed 42 \
    --num_workers 36 \
    --config_overrides data.max_frames=32 "data.sample_type_ratio=[1.0, 1.0, 0.0]" "data.train_datasets=[franka,oxe,others,libero,suboptimal_fail,paired,mw]" data.min_frames_per_trajectory=1  data.dataset_type=strategy_first

# RFM-Eval
uv run scripts/generate_vqa_dataset.py \
    --num_epochs 0.1 \
    --output_path vqa_datasets/rfm_val_0.1epoch \
    --seed 42 \
    --num_workers 16 \
    --eval_mode \
    --config_overrides data.max_frames=32 "data.sample_type_ratio=[1.0, 1.0, 0.0]" "data.train_datasets=[franka,oxe,others,libero,suboptimal_fail,paired,mw]" data.min_frames_per_trajectory=1 data.dataset_type=strategy_first

# RFM-Prog Only
uv run scripts/generate_vqa_dataset.py \
    --num_epochs 10.0 \
    --output_path vqa_datasets/rfm_train_10epochs_prog_only \
    --seed 42 \
    --num_workers 36 \
    --config_overrides data.max_frames=32 "data.sample_type_ratio=[0.0, 1.0, 0.0]" "data.train_datasets=[franka,oxe,others,libero,suboptimal_fail,paired,mw]" data.min_frames_per_trajectory=1  data.dataset_type=strategy_first

# RFM-Eval Prog Only
uv run scripts/generate_vqa_dataset.py \
    --num_epochs 0.1 \
    --output_path vqa_datasets/rfm_val_0.1epoch_prog_only \
    --seed 42 \
    --num_workers 16 \
    --eval_mode \
    --config_overrides data.max_frames=32 "data.sample_type_ratio=[0.0, 1.0, 0.0]" "data.eval_datasets=[franka,oxe,others,libero,suboptimal_fail,paired,mw]" data.min_frames_per_trajectory=1 data.dataset_type=strategy_first