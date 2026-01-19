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
    roboreward_model_path=teetone/RoboReward-8B \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    roboreward_max_new_tokens=128 \
    gvl_max_frames=64

# on all
uv run python rfm/evals/run_baseline_eval.py \
    reward_model=roboreward \
    roboreward_model_path=teetone/RoboReward-8B \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking,jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all]] \
    roboreward_max_new_tokens=128 \
    gvl_max_frames=64