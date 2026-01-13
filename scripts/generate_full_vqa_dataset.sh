# RFM-50M
uv run scripts/generate_vqa_dataset.py \
    --num_epochs 10.0 \
    --output_path vqa_datasets/rfm_train_10epochs \
    --seed 42 \
    --num_workers 32 \
    --config_overrides data.max_frames=32 "data.sample_type_ratio=[1.0, 1.0, 0.0]" "data.train_datasets=[franka,oxe,others,libero,suboptimal_fail,paired,mw]" data.min_frames_per_trajectory=1  data.dataset_type=strategy_first


uv run scripts/generate_vqa_dataset.py \
    --num_epochs 1.0 \
    --output_path vqa_datasets/rfm_val_1epoch \
    --seed 42 \
    --num_workers 32 \
    --eval_mode \
    --config_overrides data.max_frames=32 "data.sample_type_ratio=[1.0, 1.0, 0.0]" "data.train_datasets=[franka,oxe,others,libero,suboptimal_fail,paired,mw]" data.min_frames_per_trajectory=1 data.dataset_type=strategy_first