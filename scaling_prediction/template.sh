#!/bin/zsh -l

# ------ pjsub option -------- #
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L proc-crproc=4096
#PJM -L elapse=1:10:00
#PJM -o logs/%j.log
#PJM -j

export OMP_NUM_THREADS=40
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

conda activate pql

nvidia-smi
nproc
ulimit -u
ulimit -Hu
python -V
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"

export LD_LIBRARY_PATH=$(conda info --base)/envs/pql/lib/:$LD_LIBRARY_PATH
NUM_ENVS=4096

# --- params settings --- #
TASK="Ant"
UTD_RATIO_INVERSE=2048
BATCH_SIZE=2048
LEARNING_RATE=0.0001
NUM_ENVS=4096
SEED=42
USE_PAL=False
WANDB_PROJECT="scales-pred-nopal-3"
REPLAY_BUFFER_SIZE=1e6
RUN_ID=0
SCRIPT_NAME="script_file_name.sh"
# --- params settings --- #

echo "### STARTING RUN: SEED=$SEED, UTD_RATIO=$UTD_RATIO_INVERSE, BATCH_SIZE=$BATCH_SIZE ###"

python scripts/train_pql.py \
    "task=$TASK" \
    "num_envs=$NUM_ENVS" \
    "run_id=$RUN_ID" \
    "algo.num_gpus=1" \
    "algo.num_cpus=20" \
    "algo.memory_size=$REPLAY_BUFFER_SIZE" \
    "algo.pal=$USE_PAL" \
    "logging.wandb.project=\"$WANDB_PROJECT\"" \
    "algo.batch_size=$BATCH_SIZE" \
    "algo.critic_sample_ratio=$(echo "scale=4; $NUM_ENVS / $UTD_RATIO_INVERSE" | bc)" \
    "algo.utd_ratio_inverse=$UTD_RATIO_INVERSE" \
    "algo.actor_lr=$LEARNING_RATE" \
    "algo.critic_lr=$LEARNING_RATE" \
    "logging.shell_script_name=\"$SCRIPT_NAME\"" \
    "seed=$SEED"

if [ $? -ne 0 ]; then
    echo "!!!!!! ERROR DETECTED on SEED=$SEED. Stopping script. !!!!!!"
    exit 1
fi

echo "### FINISHED RUN: SEED=$SEED. Waiting for a few minutes before the next run. ###"
sleep 30
echo "-----------------------------------------------------"
