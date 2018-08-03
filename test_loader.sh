# DATADIR="/deep/group/med/mimic-iii/train/"
# DATADIR="/Users/dmorina/mnt/cdr_mimic/"
DATADIR="/deep/group/tony/cdr-mimic/data/icd_codes_only"

ARGUMENTS="--data_dir $DATADIR --verbose"

python test_loader.py ${ARGUMENTS}
