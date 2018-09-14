import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable

import pdb

torch.multiprocessing.set_start_method('spawn', force=True)

# MIMIC Dataset
class Dataset(data.Dataset):
    def __init__(self, args, phase, is_training_set=True):
        datadir = Path(args.data_dir)

        src_file_name = phase + '_src.npy'
        tgt_file_name = phase + '_tgt.npy'

        src_path = datadir / src_file_name
        tgt_path = datadir / tgt_file_name

        self.src = torch.from_numpy(np.load(src_path)).cuda()
        self.tgt = torch.from_numpy(np.load(tgt_path)).cuda()

        self.num_classes = 1

        self.is_training_set = is_training_set
        self.phase = phase

        self.D_in = len(self.src[0])

        self.num_examples = len(self.src)
        if args.verbose:
            # print(f'Loaded {src_path} and {tgt_path}')
            print(f"Phase: {phase}, number of examples: {self.num_examples}")

    def __getitem__(self, index):
        src = self.src[index].float()
        tgt = self.tgt[index].float()
        print('src', src.shape, src.dtype)
        print('tgt', tgt.shape, tgt.dtype)
        return src, tgt

    def __len__(self):
        return len(self.src)

def get_loader(args, phase='train', is_training=True):
    dataset = Dataset(args, phase)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=is_training)
    loader.phase = phase
    # loader.dataset = dataset

    num_demographics = 2
    loader.D_in = dataset.D_in
    return loader


