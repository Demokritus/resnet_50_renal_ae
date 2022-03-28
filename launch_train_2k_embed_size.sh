#!/bin/bash -l
#SBATCH --partition=excellence-exclusive
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --account=UniKoeln
#SBATCH --output=last_log_run_train_parallel.log
#SBATCH --error=last_errors_run_train_parallel.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsergei@uni-koeln.de

LATENT_DIMS=(131072 16384 4096 2048 1024)

export LATENT_DIM=${LATENT_DIMS[3]}
export MASTER_PORT="9006"
export N_EPOCHS=1000
# export DIM=16
export BATCH_SIZE=4
export N_GPUS=1
export SCALE_FACTOR=1.0
export LEARNING_R=0.0001
export DIR_CHECKPOINT=$2


export workdir=$1
#mkdir -p $workdir

cd $workdir

python3 train_parallel.py -l $LEARNING_R -s $SCALE_FACTOR -g $N_GPUS -b $BATCH_SIZE -e $N_EPOCHS \
    -D $DIR_CHECKPOINT"_latent_dim_"$LATENT_DIM -f -1 -S $LATENT_DIM -M $MASTER_PORT
