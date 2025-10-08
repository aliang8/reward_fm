jid=$(sbatch --parsable docker/reserve.sbatch)
srun --jobid $jid --pty bash

echo ${SLURM_STEP_GPUS}
GIDS=$(id -G)


docker run --rm -it --ipc=host \
  --gpus '"device=5,6"' \
  --user "$(id -u):$(id -g)" $(for g in $GIDS; do printf -- "--group-add %s " "$g"; done) \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -e HOME=/workspace \
  -v /scr/aliang80/reward_fm:/workspace \
  -v /scr/shared/reward_fm/rfm_dataset:/scr/shared/reward_fm/rfm_dataset:ro \
  -v /scr/shared/reward_fm/processed_datasets:/scr/shared/reward_fm/processed_datasets:ro \
  rfm-dev:latest bash