uv run python rfm/data/generate_hf_dataset.py \
    --config_path=rfm/configs/data_gen_configs/libero.yaml\
    --dataset.dataset_path=deps/libero/LIBERO/libero/datasets/libero_10 \
    --dataset.dataset_name=libero_10

uv run python rfm/data/generate_hf_dataset.py \
    --config_path=rfm/configs/data_gen_configs/libero.yaml\
    --dataset.dataset_path=deps/libero/LIBERO/libero/datasets/libero_90 \
    --dataset.dataset_name=libero_90

uv run python rfm/data/generate_hf_dataset.py \
    --config_path=rfm/configs/data_gen_configs/libero.yaml\
    --dataset.dataset_path=deps/libero/LIBERO/libero/datasets/libero_spatial \
    --dataset.dataset_name=libero_spatial

uv run python rfm/data/generate_hf_dataset.py \
    --config_path=rfm/configs/data_gen_configs/libero.yaml\
    --dataset.dataset_path=deps/libero/LIBERO/libero/datasets/libero_goal \
    --dataset.dataset_name=libero_goal

uv run python rfm/data/generate_hf_dataset.py \
    --config_path=rfm/configs/data_gen_configs/libero.yaml\
    --dataset.dataset_path=deps/libero/LIBERO/libero/datasets/libero_object \
    --dataset.dataset_name=libero_object