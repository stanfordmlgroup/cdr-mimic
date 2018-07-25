# DATADIR="/deep/group/med/mimic-iii/train/"
DATADIR="/Users/dmorina/mnt/cdr_mimic/"

ARGUMENTS="--data_dir $DATADIR --name SIMPLEMIMIC --model SimpleNN"

python3 train.py ${ARGUMENTS}
