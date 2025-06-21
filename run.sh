#!/bin/zsh

source ~/.zshrc
conda activate pql

export LD_LIBRARY_PATH=$(conda info --base)/envs/pql/lib/:$LD_LIBRARY_PATH
NUM_ENVS=4096
export CUDA_VISIBLE_DEVICES=4,5

run_seeds() {
    local UTD_RATIO_INVERSE=$1
    local BATCH_SIZE=$2
    local LEARNING_RATE=$3

    for SEED in 42 43 44 45 46; do
        python scripts/train_pql.py \
            task=Ant num_envs=$NUM_ENVS \
            algo.batch_size=$BATCH_SIZE \
            algo.critic_sample_ratio=$(( NUM_ENVS / UTD_RATIO_INVERSE )) \
            algo.actor_lr=$LEARNING_RATE \
            algo.critic_lr=$LEARNING_RATE \
            seed=$SEED || true
    done
}

run_seeds 2048 512 0.0002
run_seeds 4096 512 0.0002
run_seeds 8192 512 0.0002
run_seeds 16384 512 0.0002
