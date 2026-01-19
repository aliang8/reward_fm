# ReWIND
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=rewind \
    model_path=rewardfm/rewind-scale-rfm1M-32layers-8frame-20260118-180522 \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rfm-1m-id,rfm-1m-ood] \
    custom_eval.use_frame_steps=true \
    custom_eval.subsample_n_frames=5 \
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=8 \
    model_config.batch_size=64

# VLAC
uv run --extra vlac --python .venv-vlac/bin/python  rfm/evals/run_baseline_eval.py \
    reward_model=vlac \
    model_path=InternRobotics/VLAC \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rfm-1m-id,rfm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.reward_alignment_max_trajectories=30 \
    custom_eval.pad_frames=false \
    max_frames=64

# ROBOREWARD
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=roboreward \
    model_path=teetone/RoboReward-4B \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rfm-1m-id,rfm-1m-ood] \
    custom_eval.use_frame_steps=true \
    custom_eval.subsample_n_frames=5 \
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=64 \
    model_config.batch_size=32

# RFM 
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=rfm \
    model_path=rewardfm/rfm_qwen_pref_prog_4frames_all_strategy \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rfm-1m-id,rfm-1m-ood] \
    custom_eval.use_frame_steps=true \
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=4 \
    model_config.batch_size=32