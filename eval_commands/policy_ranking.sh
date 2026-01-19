# ReWIND
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=rewind \
    model_path=rewardfm/rewind-scale-rfm1M-32layers-8frame-20260118-180522 \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rfm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=8 \
    model_config.batch_size=64

# VlAC
uv run --extra vlac --python .venv-vlac/bin/python rfm/evals/run_baseline_eval.py \
    reward_model=vlac \
    model_path=InternRobotics/VLAC \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rfm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.pad_frames=false \
    max_frames=64 \
    custom_eval.num_examples_per_quality_pr=1000

# ROBOREWARD
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=roboreward \
    model_path=teetone/RoboReward-8B \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rfm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.pad_frames=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=64

# RFM
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=rfm \
    model_path=rewardfm/rfm_qwen_pref_prog_4frames_all_strategy \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rfm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=4 \
    model_config.batch_size=32