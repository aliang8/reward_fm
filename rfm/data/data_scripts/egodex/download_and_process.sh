#!/bin/bash


for part in test part1 part3 part4 part5; do

    echo "Processing ${part}..."

    # Download the dataset. 
    # Example curl "https://ml-site.cdn-apple.com/datasets/egodex/test.zip" -o test.zip
    # Download it to ~/egodex/${part}.zip
    echo "curl "https://ml-site.cdn-apple.com/datasets/egodex/${part}.zip" -o ${HOME}/egodex/${part}.zip"
    # curl "https://ml-site.cdn-apple.com/datasets/egodex/${part}.zip" -o ${HOME}/egodex/${part}.zip
    # unzip -d ${HOME}/egodex/${part} ${HOME}/egodex/${part}.zip

    rm ${HOME}/egodex/${part}.zip


    uv run python rfm/data/generate_hf_dataset.py \
        --config_path=rfm/configs/data_gen_configs/egodex.yaml \
        --dataset.dataset_path="${HOME}/egodex/${part}/${part}" \
        --dataset.dataset_name="egodex_${part}" \
        --hub.push_to_hub=true

    echo "Done processing ${part}..."

    # Delete the dataset
    echo "Deleting ${HOME}/egodex/${part}..."
    exit 1
    rm -rf ${HOME}/egodex/${part}
    echo "Done deleting ${HOME}/egodex/${part}..."

done