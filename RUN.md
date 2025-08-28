# Instructions for setup -> training -> eval 

```bash
## clone codebase
git clone https://github.com/aliang8/reward_fm.git 
git checkout anthony

# download dataset to local 
# might have to run this multiple times if it crashes
./setup.sh

# preprocess dataset 
uv run python3 scripts/preprocess_dataset.py

# train
./train.sh

# start eval server 
uv run python3 evals/qwen_server.py 

# run eval 
uv run python3 evals/run_model_eval.py

# visualize results 
uv run python3 evals/compile_result.py {insert json file}
```