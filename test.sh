DATADIR="/deep/group/sharonz/cdr_mimic/data"
NUM_WORKERS=0

ARGUMENTS="
    --data_dir $DATADIR
    --phase train
    --num_workers $NUM_WORKERS
    --name CRPS_LR1e2
    --ckpt_path ckpts/CRPS_LR1e2/epoch_1.pth.tar
    --gpu_ids 0
    "
python test.py ${ARGUMENTS}
