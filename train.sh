# DATADIR="/deep/group/med/mimic-iii/train/"
# DATADIR="/Users/dmorina/mnt/cdr_mimic/"
DATADIR="/deep/group/sharonz/cdr_mimic/data"
NAME="DEBUG_PARALLELIZATION_BATCHSIZE"
NUM_EPOCHS=1000
MODEL="SimpleNN"
LR=1e-6
NUM_WORKERS=1
EPOCHS_PER_SAVE=1
BATCH_SIZE=1

ARGUMENTS="--data_dir $DATADIR --name $NAME --num_epochs $NUM_EPOCHS --model $MODEL --batch_size $BATCH_SIZE --lr $LR --epochs_per_save $EPOCHS_PER_SAVE --num_workers $NUM_WORKERS --verbose"

python train.py ${ARGUMENTS}
