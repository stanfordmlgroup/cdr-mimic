import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable

SRC_FILE_NAME = 'src_sample.csv'
TGT_FILE_NAME = 'tgt_sample.csv'

# MIMIC Dataset
class Dataset(data.Dataset):
    def __init__(self, args, phase, is_training_set=True):
        datadir = Path(args.data_dir)

        src_csv_path = datadir / SRC_FILE_NAME
        tgt_csv_path = datadir / TGT_FILE_NAME

        self.df_src = pd.read_csv(src_csv_path, delimiter='\n', header=None).values
        self.df_tgt = pd.read_csv(tgt_csv_path, delimiter=',', header=None).values

        if args.verbose:
            print(f"{phase} number of examples: {len(self.df_src)}")

        self.num_classes = 1

        index = 0
        self.vocab_w2i = {}
        self.vocab_i2w = {}
        self.encoded_df_src = []

        self.is_training_set = is_training_set
        self.phase = phase

        for i, row in enumerate(self.df_src):
            parsed_row = row[0].replace(" ", "").replace("'", "").split(',')
            encoded_row = np.zeros(len(parsed_row))

            # Demographics: (1) gender and (2) age of prediction
            NUM_DEMOGRAPHICS = 2

            gender = 0 if parsed_row[0] == 'M' else 1
            encoded_row[0] = gender

            age_of_pred = parsed_row[1]
            encoded_row[1] = age_of_pred
            
            # After demographics, the rest are ICD codes
            icd_codes = parsed_row[2:]

            for j, word in enumerate(icd_codes):
                # Add to vocab dictionaries
                if word not in self.vocab_w2i.keys():
                    self.vocab_w2i[word] = index
                    self.vocab_i2w[index] = word
                    index += 1
                # Encode strings to indexes
                encoded_row[j + NUM_DEMOGRAPHICS] = self.vocab_w2i[word]
            self.encoded_df_src.append(encoded_row)
        self.encoded_df_src = np.array(self.encoded_df_src)

        src_lengths = np.array([len(i) for i in self.encoded_df_src])
        self.max_src_len = np.amax(src_lengths)

        self.src_tensor = torch.FloatTensor(np.size(self.encoded_df_src), int(self.max_src_len)).fill_(-1)
        for i, row in enumerate(self.encoded_df_src):
            self.src_tensor[i, :np.size(row)] = torch.from_numpy(row)

    def __getitem__(self, index):
        src = self.src_tensor[index]
        tgt = self.df_tgt[index]

        # tgt = np.array([np.array([time.mktime(time.strptime(tgt[0], "%Y-%m-%d"))]), tgt[1]])
        tgt = torch.FloatTensor(tgt)

        return src, tgt

    def __len__(self):
        return len(self.df_src)

def get_loader(args, phase='train', is_training=True):
    dataset = Dataset(args, phase)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=is_training)
    loader.phase = phase
    if is_training:
        D_in = dataset.max_src_len
        return loader, D_in
    else:
        return loader

