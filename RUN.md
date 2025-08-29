# Instructions for setup -> training -> eval 

```bash
# clone codebase
git clone https://github.com/aliang8/reward_fm.git 
git checkout anthony

# download dataset to local 
# might have to run this multiple times if it crashes
./setup.sh

# preprocess dataset 
uv run python3 scripts/preprocess_dataset.py

# train
# look at rfm/configs/config.yaml for parameters
./train.sh 

# upload your trained model to huggingface 
uv run python3 scripts/upload_to_hub.py --model_dir=logs/rfm_v3/checkpoint-900/ --hub_model_id=aliangdw/rfm_v3

# start eval server in one terminal
uv run python3 evals/qwen_server.py 

# run eval in another terminal
# look at rfm/configs/eval_config.yaml for eval parameters
# NOTE: remember to change model_path in eval_configs
uv run python3 evals/run_model_eval.py

# visualize results 
uv run python3 evals/compile_result.py {insert json file from eval}
```