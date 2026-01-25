# GVL
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=gvl \
    model_config.provider=openai \
    model_config.model_name=gpt-4o-mini \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rfm-1m-id,rfm-1m-ood] \
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=15

# ReWIND
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=rewind \
    model_path="/home/azure/reward_fm/logs/rfm-1m-id_ablation_rewind_bs512_prog_pref_4frames_continuous_scaled/ckpt-latest-avg-2metrics\=0.6334_step\=6100" \
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
    model_path=aliangdw/qwen4b_pref_prog_succ_8_frames_all \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rfm-1m-id,rfm-1m-ood] \
    custom_eval.use_frame_steps=true \
    custom_eval.subsample_n_frames=5 \
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=4 \
    model_config.batch_size=32

# LIBERO ABLATION
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=rfm \
    model_path="/gpfs/home/jessezha/scrubbed_storage/reward_fm/logs/libero_ablation_prog_only_lora_ft_4frames_2000steps/ckpt-avg-3metrics\=0.6280_step\=1000" \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[libero_pi0] \
    custom_eval.use_frame_steps=false \
    custom_eval.reward_alignment_max_trajectories=100 \
    max_frames=4 \
    model_config.batch_size=32


/gpfs/home/jessezha/scrubbed_storage/reward_fm/logs/rfm_1m_ablation_prog_only_8frames/ckpt-latest-avg-2metrics\=0.3964_step\=5250/

/gpfs/home/jessezha/scrubbed_storage/reward_fm/logs/rfm_1m_ablation_prog_pref_8frames/ckpt-latest-avg-2metrics\=0.5777_step\=2000/