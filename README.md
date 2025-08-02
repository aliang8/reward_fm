# Train Reward Foundation Model

```bash
# generate dataset and push to hub
python data/generate_hf_dataset.py \
    --config_path=configs/data_gen.yaml \
    --dataset.dataset_path=LIBERO/libero/datasets/libero_90 \
    --dataset.dataset_name=libero_90
```


```bash
# training
accelerate launch --config_file configs/fsdp.yaml train.py --config_path=configs/config.yaml

# eval
accelerate launch --config_file configs/fsdp.yaml train.py --mode=evaluate

```