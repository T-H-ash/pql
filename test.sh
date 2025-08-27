#!/bin/zsh -l


# export OMP_NUM_THREADS=40
export CUDA_VISIBLE_DEVICES=0

# conda activate pql

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
WANDB_PROJECT="pql-test"
UTD_RATIO_INVERSE=1024
LEARNING_RATE=0.0009
SEED=44
TASK="Ant"
USE_PAL=True
REPLAY_BUFFER_SIZE=5000000
BATCH_SIZE=8192
# --- params settings --- #

echo "### STARTING RUN: SEED=$SEED, UTD_RATIO=$UTD_RATIO_INVERSE, BATCH_SIZE=$BATCH_SIZE ###"

python scripts/train_pql.py \
    task=$TASK \
    num_envs=$NUM_ENVS \
    run_id=$RUN_ID \
    algo.num_gpus=1 \
    algo.num_cpus=20 \
    algo.memory_size=$REPLAY_BUFFER_SIZE \
    algo.pal=$USE_PAL \
    logging.wandb.project=$WANDB_PROJECT \
    algo.batch_size=$BATCH_SIZE \
    algo.critic_sample_ratio=$(echo "scale=4; $NUM_ENVS / $UTD_RATIO_INVERSE" | bc) \
    algo.utd_ratio_inverse=$UTD_RATIO_INVERSE \
    algo.actor_lr=$LEARNING_RATE \
    algo.critic_lr=$LEARNING_RATE \
    logging.shell_script_name=$(basename "$0") \
    seed=$SEED

if [ $? -ne 0 ]; then
    echo "!!!!!! ERROR DETECTED on SEED=$SEED. Stopping script. !!!!!!"
    exit 1
fi

echo "### FINISHED RUN: SEED=$SEED. Waiting for a few minutes before the next run. ###"
sleep 30
echo "-----------------------------------------------------"
