# Train DPO Reward Model

```bash
# training
accelerate launch --config_file fsdp.yaml train_dpo.py --config_path=config.yaml

# eval
accelerate launch --config_file fsdp.yaml train_dpo.py --mode=evaluate

```