# ReWIND
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rewind \
    model_path=rewardfm/rewind-scale-rfm1M-32layers-8frame-20260118-180522 \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rfm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=8 \
    model_config.batch_size=64

# Robo-Dopamine (run with venv Python so vLLM is found; do not use uv run)
.venv-robodopamine/bin/python robometer/evals/run_baseline_eval.py \
    reward_model=robodopamine \
    model_path=tanhuajie2001/Robo-Dopamine-GRM-3B \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rfm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=64 \
    model_config.batch_size=1

# VlAC
uv run --extra vlac --python .venv-vlac/bin/python robometer/evals/run_baseline_eval.py \
    reward_model=vlac \
    model_path=InternRobotics/VLAC \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rfm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.pad_frames=false \
    max_frames=64 \
    custom_eval.num_examples_per_quality_pr=1000

# ROBOREWARD
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=roboreward \
    model_path=teetone/RoboReward-8B \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rfm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.pad_frames=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=64

# RFM 8 frame model DISCRETE
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rfm \
    model_path=aliangdw/qwen4b_pref_prog_succ_8_frames_all_part2 \
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

uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rfm \
    model_path="/gpfs/home/jessezha/scrubbed_storage/reward_fm/logs/libero_ablation_prog_only_lora_ft_4frames_2000steps/ckpt-avg-3metrics\=0.6280_step\=1000" \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[libero_pi0] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=20 \
    max_frames=4 \
    model_config.batch_size=32