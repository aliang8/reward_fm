# Instructions for setup -> training -> eval 

```bash
# clone codebase
git clone https://github.com/aliang8/reward_fm.git 
git checkout anthony

# get uv 
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# install dependencies
uv pip install huggingface_hub

export RFM_DATASET_PATH=/scr/shared/reward_fm/rfm_dataset
export RFM_PROCESSED_DATASETS_PATH=/scr/shared/reward_fm/processed_dataset

# download dataset to local 
# might have to run this multiple times if it crashes
./setup.sh

# install ffmpeg if not already have
sudo apt-get install ffmpeg

# preprocess dataset 
uv run python3 scripts/preprocess_datasets.py --cache_dir=$RFM_PROCESSED_DATASETS_PATH

# train
# look at rfm/configs/config.yaml for parameters
./train.sh 

# upload your trained model to huggingface 
# this should take a couple minutes
uv run python3 scripts/upload_to_hub.py --model_dir=logs/rfm_v3/checkpoint-900/ --hub_model_id=aliangdw/rfm_v3

# start eval server in one terminal
uv run python3 evals/qwen_server.py --num_gpus=2

# run eval in another terminal
# look at rfm/configs/eval_config.yaml for eval parameters
# NOTE: remember to change model_path in eval_configs
uv run python3 evals/run_model_eval.py

# visualize results 
uv run python3 evals/compile_results.py {insert json file from eval}
```