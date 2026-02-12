# ReWIND
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rewind \
    model_path=rewardfm/rewind-scale-rfm1M-32layers-8frame-20260118-180522 \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
    custom_eval.use_frame_steps=true \
    custom_eval.subsample_n_frames=5 \
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=8 \
    model_config.batch_size=64

# Robo-Dopamine (run with venv Python so vLLM is found; do not use uv run)
.venv-robodopamine/bin/python robometer/evals/run_baseline_eval.py \
    reward_model=robodopamine \
    model_path=tanhuajie2001/Robo-Dopamine-GRM-3B \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=64 \
    model_config.batch_size=1

# VLAC
uv run --extra vlac --python .venv-vlac/bin/python  robometer/evals/run_baseline_eval.py \
    reward_model=vlac \
    model_path=InternRobotics/VLAC \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.reward_alignment_max_trajectories=30 \
    custom_eval.pad_frames=false \
    max_frames=64

# ROBOREWARD
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=roboreward \
    model_path=teetone/RoboReward-4B \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
    custom_eval.use_frame_steps=true \
    custom_eval.subsample_n_frames=5 \
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=64 \
    model_config.batch_size=32

# RFM 
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rbm \
    model_path=aliangdw/qwen4b_pref_prog_succ_8_frames_all \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
    custom_eval.use_frame_steps=true \
    custom_eval.subsample_n_frames=5 \
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=4 \
    model_config.batch_size=32

# LIBERO ABLATION
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rbm \
    model_path="/gpfs/home/jessezha/scrubbed_storage/reward_fm/logs/libero_ablation_prog_only_lora_ft_4frames_2000steps/ckpt-avg-3metrics\=0.6280_step\=1000" \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[libero_pi0] \
    custom_eval.use_frame_steps=false \
    custom_eval.reward_alignment_max_trajectories=100 \
    max_frames=4 \
    model_config.batch_size=32