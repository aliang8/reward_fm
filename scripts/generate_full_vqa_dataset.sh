# RFM-50M
uv run scripts/generate_vqa_dataset.py \
    --num_samples 50_000_000 \
    --save_batch_size 1_000_000 \
    --output_path vqa_datasets/rfm_train_50m \
    --seed 42 \
    --num_workers 32 \
    --config_overrides data.max_frames=32 "data.sample_type_ratio=[1.0, 1.0, 0.0]" "data.train_datasets=[franka,oxe,others,libero,suboptimal_fail,paired,mw]" data.min_frames_per_trajectory=2 

