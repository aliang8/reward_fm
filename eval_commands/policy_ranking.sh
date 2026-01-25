# GVL
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=gvl \
    model_config.provider=openai \
    model_config.model_name=gpt-4o-mini \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rfm-1m-ood] \
    max_frames=15

# ReWIND
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=rewind \
    model_path="/home/azure/reward_fm/logs/rfm-1m-id_ablation_rewind_bs512_prog_pref_4frames_continuous_scaled/ckpt-latest-avg-2metrics\=0.6334_step\=6100" \
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

ant_rfm_qwen4b_4gpu_bs16_pref_prog_succ_8_frames_all_discrete_10_bins_part2


# RFM 8 frame model DISCRETE
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=rfm \
    model_path="/gpfs/home/jessezha/scrubbed_storage/reward_fm/logs/ant_rfm_qwen4b_4gpu_bs16_pref_prog_succ_8_frames_all_discrete_10_bins_part2/ckpt-avg-5metrics\=0.7155_step\=4500" \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rfm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=8 \
    model_config.batch_size=32


# LIBERO ABLATION

"/gpfs/home/jessezha/scrubbed_storage/reward_fm/logs/libero_ablation_progpref_lora_ft_4frames_2000steps/ckpt-avg-3metrics\=0.6809_step\=450/"
"/gpfs/home/jessezha/scrubbed_storage/reward_fm/logs/libero_ablation_prog_pref_with_fail_lora_ft_4frames_2000steps/ckpt-avg-3metrics\=0.7650_step\=700/"
"/gpfs/home/jessezha/scrubbed_storage/reward_fm/logs/libero_ablation_prog_only_lora_ft_4frames_2000steps/ckpt-avg-3metrics\=0.6280_step\=1000"

uv run python rfm/evals/run_baseline_eval.py \
    reward_model=rfm \
    model_path="/gpfs/home/jessezha/scrubbed_storage/reward_fm/logs/libero_ablation_prog_only_lora_ft_4frames_2000steps/ckpt-avg-3metrics\=0.6280_step\=1000" \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[libero_pi0] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=20 \
    max_frames=4 \
    model_config.batch_size=32