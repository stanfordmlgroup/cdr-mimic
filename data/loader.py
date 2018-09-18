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

        self.src = np.load(src_path)
        self.tgt = np.load(tgt_path)

        self.num_classes = 1

        self.is_training_set = is_training_set
        self.phase = phase

        self.w2i = pd.read_csv("data/vocab_w2i.csv", header = None)
        self.w2i = self.w2i.set_index(0)
        self.vector_size = len(self.w2i) + 2
        self.D_in = self.vector_size

        self.num_examples = len(self.src)
        if args.verbose:
            print(f"Phase: {phase}, number of examples: {self.num_examples}")

    def __getitem__(self, index):
        vector = np.zeros(self.vector_size)
        vector[0] = 1 if self.src[index][0] == "M" else 0
        vector[1] = self.src[index][1]
        indices = self.w2i.loc[self.src[index][2:]]
        vector[indices + 2] = 1
        tgt = self.tgt[index]
        print("index: %d" % index)
        return torch.FloatTensor(vector), torch.FloatTensor(tgt)

    def __len__(self):
        return len(self.src)

def get_loader(args, phase='train', is_training=True):
    dataset = Dataset(args, phase)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=is_training)
    loader.phase = phase
    num_demographics = 2
    loader.D_in = dataset.D_in
    return loader


