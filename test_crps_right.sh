DATADIR="/deep/group/sharonz/cdr_mimic/data"
NUM_WORKERS=0

ARGUMENTS="
    --data_dir $DATADIR
    --phase test
    --num_workers $NUM_WORKERS
    --name CRPS_RIGHT
    --ckpt_path ckpts/CRPS_RIGHT/epoch_5.pth.tar
    --gpu_ids 0
    "
python test.py ${ARGUMENTS}
