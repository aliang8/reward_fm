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

# # also install flash-attention-2, this works on snoopy A6000
# # this might take a while to run
# uv run pip uninstall -y flash-attn
# export TORCH_CUDA_ARCH_LIST="8.6"     # (or "8.0;8.6" if you also run on A100)
# MAX_JOB=4 uv run pip install --no-build-isolation --no-cache-dir flash-attn -v

export RFM_DATASET_PATH=/scr/shared/reward_fm/rfm_dataset
export RFM_PROCESSED_DATASETS_PATH=/scr/shared/reward_fm/processed_dataset
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# download dataset to local 
# might have to run this multiple times if it crashes
./scripts/download_data.sh

# install ffmpeg if not already have
sudo apt-get install ffmpeg

# preprocess dataset 
uv run python3 rfm/data/scripts/preprocess_datasets.py \
    --config_path=rfm/configs/preprocess.yaml \
    cache_dir=$RFM_PROCESSED_DATASETS_PATH

# train
# look at rfm/configs/config.yaml for parameters
./scripts/train.sh 

# upload your trained model to huggingface 
# this should take a couple minutes
uv run python3 rfm/utils/upload_to_hub.py --model_dir=logs/rfm_v3/checkpoint-900/ --hub_model_id=aliangdw/rfm_v3

# start eval server in one terminal
uv run python3 evals/eval_server.py --num_gpus=2

# run eval in another terminal
# look at rfm/configs/eval_config.yaml for eval parameters
# NOTE: remember to change model_path in eval_configs
uv run python3 evals/run_model_eval.py

# visualize results 
uv run python3 evals/compile_results.py {insert json file from eval}
```


```
# Docker setup

docker build -t rfm-dev:latest .

docker run --rm -it --gpus all --ipc=host \
  --user $(id -u):$(id -g) \
  -v /scr/aliang80/reward_fm:/workspace \
  -v /scr/shared/reward_fm/rfm_dataset:/scr/shared/reward_fm/rfm_dataset:ro \
  -v /scr/shared/reward_fm/processed_datasets:/scr/shared/reward_fm/processed_datasets:ro \
  -e RFM_DATASET_PATH=/scr/shared/reward_fm/rfm_dataset \
  -e RFM_PROCESSED_DATASETS_PATH=/scr/shared/reward_fm/processed_datasets \
  rfm-dev:latest


```