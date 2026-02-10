# GVL
export GEMINI_API_KEY="your-api-key-here"
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=gvl \
    model_config.model_name=gemini-2.5-flash-lite \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    max_frames=8

uv run python rfm/evals/run_baseline_eval.py \
    reward_model=gvl \
    model_config.provider=openai \
    model_config.model_name=gpt-4o-mini \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    max_frames=8

# ReWIND
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=rewind \
    model_path=aliangdw/rewind_rfm-1m-id_continuous_scaled \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    custom_eval.use_frame_steps=false \
    max_frames=8 \
    model_config.batch_size=64

# VLAC
uv run --extra vlac --python .venv-vlac/bin/python python rfm/evals/run_baseline_eval.py \
    reward_model=vlac \
    model_path=InternRobotics/VLAC \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    custom_eval.pad_frames=false \
    max_frames=64

# ROBOREWARD
# without koch
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=roboreward \
    model_path=teetone/RoboReward-8B \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    max_frames=64

# on all
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=roboreward \
    model_path=teetone/RoboReward-8B \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking,jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all]] \
    max_frames=64

# RFM
# without koch
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=rfm \
    model_path="/gpfs/home/jessezha/scrubbed_storage/reward_fm/logs/ant_rfm_qwen4b_4gpu_bs16_pref_prog_succ_8_frames_all_discrete_10_bins_part2/ckpt-avg-5metrics\=0.7155_step\=4500" \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    max_frames=8 \
    model_config.batch_size=32

# on all
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=rfm \
    model_path=aliangdw/qwen4b_pref_prog_succ_8_frames_all_part2 \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking,jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all]] \
    max_frames=8 \
    model_config.batch_size=32