DATADIR="/deep/group/sharonz/cdr_mimic/data"
NUM_WORKERS=0

ARGUMENTS="
    --data_dir $DATADIR
    --phase test
    --num_workers $NUM_WORKERS
    --name DEBUG_TESTSET
    --ckpt_path ckpts/DEBUG_TESTSET/epoch_81.pth.tar
    --gpu_ids 1
    "
python test.py ${ARGUMENTS}
