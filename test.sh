DATADIR="/deep/group/sharonz/cdr_mimic/data"
NUM_WORKERS=0

ARGUMENTS="
    --data_dir $DATADIR
    --phase test
    --num_workers $NUM_WORKERS
    --name DEBUG_FULL
    --ckpt_path ckpts/DEBUG_FULL/epoch_1.pth.tar
    --gpu_ids 2
    "
python test.py ${ARGUMENTS}
