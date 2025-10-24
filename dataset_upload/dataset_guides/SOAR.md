# SOAR Dataset Guide

This guide explains how to integrate and use the SOAR RLDS dataset with the RFM pipeline (non-streaming, local TFDS builders).

Source: `https://github.com/rail-berkeley/soar?tab=readme-ov-file#using-soar-data`

## Overview

- SOAR data is available in RLDS format. We support loading local TFDS builders for multiple splits (e.g., `success`, `failure`).
- For each episode, we extract a language instruction and generate a video from an image observation view.

## Directory Structure

```
<dataset_path>/
  rlds/
    success/
      1.0.0/
        dataset_info.json
        features.json
        ... TFRecord shards ...
    failure/
      1.0.0/
      ...
```

## Configuration (configs/data_gen_configs/soar.yaml)

```yaml
# configs/data_gen_configs/soar.yaml

dataset:
  dataset_path: ./datasets/soar
  dataset_name: soar
  rlds_splits: ["success", "failure"]

output:
  output_dir: ./rfm_dataset/soar_rfm
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false
  num_workers: 4

hub:
  push_to_hub: true
  hub_repo_id: soar_rfm
```

## Usage

```bash
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/soar.yaml
```

This will:
- Iterate the requested RLDS splits under `rlds/`
- Convert `steps` to numpy, read `language_instruction` (or similar)
- Generate web-optimized videos from an available image observation key
- Create a HuggingFace dataset ready to push/save

## Notes

- We detect the instruction from `language_instruction` or related keys at step-level or in `observation`.
- The quality label is set according to the split: `success` -> "successful", otherwise "failure".
- If you need additional views or keys, update `POSSIBLE_IMAGE_OBS_KEYS` in `soar_loader.py`. 
