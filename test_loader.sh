# DATADIR="/deep/group/med/mimic-iii/train/"
DATADIR="/Users/dmorina/mnt/cdr_mimic/"

ARGUMENTS="--datadir $DATADIR --verbose"

python3 test_loader.py ${ARGUMENTS}
