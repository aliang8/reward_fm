#!/bin/bash
#SBATCH -J rfm
#SBATCH --gres=gpu:2               # number of GPUs to reserve
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH --time=48:00:00

trap "exit 0" SIGTERM SIGINT
sleep infinity & wait