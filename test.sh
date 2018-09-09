DATADIR="/deep/group/sharonz/cdr_mimic/data"

ARGUMENTS="
    --datadir $DATADIR
    --phase test
    --name debugging_test
    --ckpt_path ckpts/DEBUG_DATA_SPLIT/best.pth.tar
    --gpu_ids 0
    "
python test.py ${ARGUMENTS}
