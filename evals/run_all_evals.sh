#!/bin/bash
# LIBERO regular datasets
echo "Running LIBERO regular dataset evaluations" 

# Run success/failure evaluation
uv run python evals/run_model_eval.py \
      --config rfm/configs/eval_config.yaml \
      --set data.batch_size=16 \
      --set num_batches=2 \
      --set data.eval_datasets=[\"abraranwar/libero_rfm\",\"ykorkmaz/libero_failure_rfm\"] \
      --set data.eval_subsets=[\"libero256_10\",\"libero_10_failure\"] \
      --set data.dataset_type=success_failure \
      --use-async \
      --max_concurrent=4 2>&1 

# Run reward alignment evaluation for LIBERO10
echo "Running LIBERO10 reward alignment evaluation"
uv run python evals/run_model_eval.py \
      --config rfm/configs/eval_config.yaml \
      --set num_batches=2 \
      --set data.batch_size=16 \
      --set data.eval_datasets=[\"abraranwar/libero_rfm\"] \
      --set data.eval_subsets=[\"libero256_10\"] \
      --set data.dataset_type=reward_alignment \
      --use-async \
      --max_concurrent=4 2>&1 

# Run reward alignment evaluation for MetaWorld eval
echo "Running MetaWorld reward alignment evaluation"
uv run python evals/run_model_eval.py \
      --config rfm/configs/eval_config.yaml \
      --set num_batches=2 \
      --set data.batch_size=16 \
      --set data.eval_datasets=[\"HenryZhang/metaworld_rewind_rfm_eval\"] \
      --set data.eval_subsets=[\"metaworld_rewind_eval\"] \
      --set data.dataset_type=reward_alignment \
      --use-async \
      --max_concurrent=4 2>&1 

# Run wrong task preference evaluation
echo "Running wrong task preference evaluation"
uv run python evals/run_model_eval.py \
      --config rfm/configs/eval_config.yaml \
      --set num_batches=10 \
      --set data.batch_size=16 \
      --set data.eval_datasets=[\"HenryZhang/metaworld_rewind_rfm_eval\"] \
      --set data.eval_subsets=[\"metaworld_rewind_eval\"] \
      --set data.dataset_type=wrong_task \
      --use-async \
      --max_concurrent=4 2>&1 

# Run progress evaluation for policy ranking 
echo "Running progress evaluation for policy ranking"
uv run python evals/run_model_eval.py \
      --config rfm/configs/eval_config.yaml \
      --set num_batches=10 \
      --set data.batch_size=16 \
      --set data.eval_datasets=[\"aliangdw/metaworld_rfm\"] \
      --set data.eval_subsets=[\"metaworld\"] \
      --set data.dataset_type=policy_ranking \
      --use-async \
      --max_concurrent=4 2>&1

# # Run confusion matrix evaluation
# uv run python evals/run_model_eval.py \
#       --config rfm/configs/eval_config.yaml \
#       --batch_size=16 \
#       --set data.eval_datasets=[\"abraranwar/libero_rfm\",\"ykorkmaz/libero_failure_rfm\"] \
#       --set data.eval_subsets=[\"libero_10\",\"libero_10_failure\"] \
#       --set data.dataset_type=confusion_matrix 2>&1 

# # for subset in "libero_10" "libero_goal" "libero_object" "libero_spatial"; do
# for subset in "libero_10"; do
#     echo "=== Evaluating subset: $subset ===" | tee -a evals/logs/libero_regular_${TIMESTAMP}.log
#     echo "Start time: $(date)" | tee -a evals/logs/libero_regular_${TIMESTAMP}.log
    
#     uv run python evals/run_model_eval.py \
#       --config_path=rfm/configs/config.yaml \
#       --server_url=http://localhost:8000 \
#       --batch_size=32 \
#       --set data.eval_datasets=[\"abraranwar/libero_rfm\",\"ykorkmaz/libero_failure_rfm\"] \
#       --set data.eval_subsets=[\"$subset\",\"libero_10_failure\"] \
#       --set data.dataset_type=success_failure 2>&1 | tee -a evals/logs/libero_regular_${TIMESTAMP}.log
    
#     echo "End time: $(date)" | tee -a evals/logs/libero_regular_${TIMESTAMP}.log
#     echo "=== Completed subset: $subset ===" | tee -a evals/logs/libero_regular_${TIMESTAMP}.log
#     echo "" | tee -a evals/logs/libero_regular_${TIMESTAMP}.log
# done

echo "Compiling results"
uv run python evals/compile_results.py \
      --config rfm/configs/eval_config.yaml \
      --eval_logs_dir eval_logs

echo "Completed all LIBERO regular dataset evaluations at $(date)" | tee -a evals/logs/libero_regular_${TIMESTAMP}.log
echo "All evaluations completed! Check logs in eval_logs/"
echo "Regular dataset log: eval_logs/libero_regular_${TIMESTAMP}.log"