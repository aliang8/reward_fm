#!/bin/bash
# check RFM_PROCESSED_DATASETS_PATH is set
if [ -z "$RFM_PROCESSED_DATASETS_PATH" ]; then
    echo "RFM_PROCESSED_DATASETS_PATH is not set"
    exit 1
fi

# download processed datasets
hf download rewardfm/processed_datasets --repo-type dataset --local-dir=$RFM_PROCESSED_DATASETS_PATH