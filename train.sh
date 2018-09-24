# DATADIR="/deep/group/med/mimic-iii/train/"
# DATADIR="/Users/dmorina/mnt/cdr_mimic/"
DATADIR="/deep/group/sharonz/cdr_mimic/data"
NAME="CRPS_INTVL_LR1e2"
NUM_EPOCHS=1
LOSS_FN='crps'
USE_INTVL='true'
MODEL="SimpleNN"
LR=1e-2
NUM_WORKERS=0
EPOCHS_PER_SAVE=1
BATCH_SIZE=50
GPU_ID=0

ARGUMENTS="--data_dir $DATADIR 
	         --name $NAME 
           --num_epochs $NUM_EPOCHS 
           --loss_fn $LOSS_FN
           --model $MODEL 
           --use_intvl $USE_INTVL
           --batch_size $BATCH_SIZE 
           --lr $LR 
           --epochs_per_save $EPOCHS_PER_SAVE 
           --num_workers $NUM_WORKERS 
           --gpu_ids $GPU_ID 
           --optimizer adam
           --verbose"

python train.py ${ARGUMENTS}
