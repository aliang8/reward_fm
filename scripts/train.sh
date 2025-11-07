uv run accelerate launch --config_file rfm/configs/fsdp.yaml --num_processes=2 \
    train.py \
    --config_paths rfm/configs/config.yaml rfm/configs/data/oxe_mw.yaml \
    --logging.use_wandb true \
    --debug false \
    --data.sample_type_ratio '[0, 1, 0]' \
    --model.train_progress_head true \
    --model.train_success_head false \
    --training.exp_name rfm_st-0_1_0_pref-f_prog-t_succ-f