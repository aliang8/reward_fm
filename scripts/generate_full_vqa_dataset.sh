# RFM-50M
uv run scripts/generate_vqa_dataset.py \
    --num_samples 10_000_000 \
    --save_batch_size 1_000_000 \
    --output_path vqa_datasets/rfm_train_10m \
    --seed 42 \
    --num_workers 32 \
    --config_overrides data.max_frames=32 "data.sample_type_ratio=[1.0, 1.0, 0.0]" "data.train_datasets=[franka,oxe,others,libero,suboptimal_fail,paired,mw]" data.min_frames_per_trajectory=2 


uv run scripts/generate_vqa_dataset.py \
    --num_samples 50_000 \
    --save_batch_size 50_000 \
    --output_path vqa_datasets/rfm_val_50k \
    --seed 42 \
    --num_workers 32 \
    --eval_mode \
    --config_overrides data.max_frames=32 "data.sample_type_ratio=[1.0, 1.0, 0.0]" "data.eval_datasets=[franka,oxe,others,libero,suboptimal_fail,paired,mw]" data.min_frames_per_trajectory=2 