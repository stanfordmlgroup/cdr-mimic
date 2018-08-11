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

        index = 1
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
                    self.vocab_w2i[word] = index
                    self.vocab_i2w[index] = word
                    index += 1
                # Encode strings to indexes
                row_encoded_icd_codes.append(self.vocab_w2i[word])
            src_icd_codes.append(row_encoded_icd_codes)
        src_icd_codes = np.array(src_icd_codes)

        icd_code_lengths = np.array([len(i) for i in src_icd_codes])
        max_icd_codes_len = np.amax(icd_code_lengths)
        self.max_src_len = max_icd_codes_len + NUM_DEMOGRAPHICS

        icd_codes_tensor = torch.LongTensor(np.size(src_icd_codes), int(max_icd_codes_len)).fill_(0)
        for i, row in enumerate(src_icd_codes):
            row = np.array(row)
            icd_codes_tensor[i, :np.size(row)] = torch.from_numpy(row)

        # One-hot encoding for ICD codes
        one_hot_icd_codes_tensor = torch.eye(index)
        one_hot_icd_codes_tensor = one_hot_icd_codes_tensor[icd_codes_tensor]

        # Concat tensors for demographics and icd codes
        expanded_src_demographics = [[[v for i in range(index)] for v in d] for d in src_demographics]
        demographics_tensor = torch.FloatTensor(expanded_src_demographics)
        # demographics_tensor = demographics_tensor.view(demographics_tensor.size(1), self.num_examples, -1)
        # one_hot_icd_codes_tensor = one_hot_icd_codes_tensor.view(one_hot_icd_codes_tensor.size(1), self.num_examples, -1)
        print(demographics_tensor.size(), one_hot_icd_codes_tensor.size())
        # print(demographics_tensor.expand_as(one_hot_icd_codes_tensor))
        # print(one_hot_icd_codes_tensor)
        # demographics_tensor = demographics_tensor.expand_as(one_hot_icd_codes_tensor)
        self.src_tensor = torch.cat((demographics_tensor, one_hot_icd_codes_tensor), dim=1)
        self.src_tensor = self.src_tensor.view(self.num_examples, -1, index)
        for x in self.src_tensor:
            print('-----')
            print(x)
            print('-----')

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

