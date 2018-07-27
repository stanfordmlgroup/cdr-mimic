# DATADIR="/deep/group/med/mimic-iii/train/"
DATADIR="/Users/dmorina/mnt/cdr_mimic/"

ARGUMENTS="--data_dir $DATADIR --name SIMPLEMIMIC --num_epochs 5 --model SimpleNN"

python3 train.py ${ARGUMENTS}
