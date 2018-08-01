# DATADIR="/deep/group/med/mimic-iii/train/"
# DATADIR="/Users/dmorina/mnt/cdr_mimic/"
DATADIR="/deep/group/sharonz/cdr_mimic/data/"


ARGUMENTS="--data_dir $DATADIR --name SIMPLEMIMIC --num_epochs 100 --model SimpleNN"

python train.py ${ARGUMENTS}
