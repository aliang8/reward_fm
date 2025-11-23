FROM foundationmodels/flash-attention:latest
WORKDIR /workspace

ENV HOME=/workspace \
    RFM_DATASET_PATH=/scr/shared/reward_fm/rfm_dataset \
    RFM_PROCESSED_DATASETS_PATH=/scr/shared/reward_fm/processed_datasets \
    TORCH_CUDA_ARCH_LIST=8.6 \
    PIP_NO_CACHE_DIR=1 \
    MAX_JOBS=1

# RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

# FIXED: copy from $HOME where the installer wrote the binaries
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    install -D -m 0755 "$HOME/.local/bin/uv"  /usr/local/bin/uv && \
    install -D -m 0755 "$HOME/.local/bin/uvx" /usr/local/bin/uvx

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
