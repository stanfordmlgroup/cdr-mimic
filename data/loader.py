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

        self.num_examples = len(self.df_src)
        if args.verbose:
            print(f"{phase} number of examples: {self.num_examples}")

        self.num_classes = 1

        vocab_index = 0
        self.vocab_w2i = {}
        self.vocab_i2w = {}
        self.encoded_src = []

        self.is_training_set = is_training_set
        self.phase = phase

        src_demographics = []
        src_icd_codes = []
        for i, row in enumerate(self.df_src):
            parsed_row = row[0].replace(" ", "").replace("'", "").split(',')

            # Demographics: (1) gender and (2) age of prediction
            NUM_DEMOGRAPHICS = 2

            gender = 0 if parsed_row[0] == 'M' else 1
            age_of_pred = float(parsed_row[1])
            src_demographics.append([gender, age_of_pred])
            
            # After demographics, the rest are ICD codes
            unencoded_icd_codes = parsed_row[2:]

            row_encoded_icd_codes = []
            for j, word in enumerate(unencoded_icd_codes):
                # Add to vocab dictionaries
                if word not in self.vocab_w2i.keys():
                    self.vocab_w2i[word] = vocab_index
                    self.vocab_i2w[vocab_index] = word
                    vocab_index += 1
                # Encode strings to indexes
                row_encoded_icd_codes.append(self.vocab_w2i[word])
            src_icd_codes.append(row_encoded_icd_codes)
        src_icd_codes = np.array(src_icd_codes)
        self.vocab_size = vocab_index

        icd_code_lengths = np.array([len(i) for i in src_icd_codes])
        max_icd_codes_len = np.amax(icd_code_lengths)
        self.max_src_len = max_icd_codes_len + NUM_DEMOGRAPHICS

        icd_codes_tensor = torch.LongTensor(np.size(src_icd_codes), int(self.vocab_size)).fill_(0)
        for i, row in enumerate(src_icd_codes):
            row = np.array(row)
            icd_codes_tensor[i, row] = 1.0

        demographics_tensor = torch.FloatTensor(src_demographics)
        self.src_tensor = torch.cat((demographics_tensor, icd_codes_tensor.float()), dim=1)


    def __getitem__(self, index):
        src = self.src_tensor[index]
        tgt = self.df_tgt[index]

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
        D_in = dataset.vocab_size + 2
        print(f'max_src_len {dataset.max_src_len}/ vocab_size {dataset.vocab_size}')
        return loader, D_in
    else:
        return loader

