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


export N_EPOCHS=1000
export DIM=16
export BATCH_SIZE=4
export N_GPUS=2
export SCALE_FACTOR=1.0
export LEARNING_R=0.0001
export DIR_CHECKPOINT=$2


export workdir=$1
#mkdir -p $workdir

cd $workdir

python3 train_parallel.py -l $LEARNING_R -s $SCALE_FACTOR -g $N_GPUS -b $BATCH_SIZE -e $N_EPOCHS \
    -D $DIR_CHECKPOINT
