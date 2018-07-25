import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable


class Dataset(data.Dataset):
    def __init__(self, args, split):
        datadir = Path(args.data_dir)

        src_csv_path = datadir / 'icd_subject_source_sample.csv'
        tgt_csv_path = datadir / 'icd_subject_target_sample.csv'

        self.df_src = pd.read_csv(src_csv_path, delimiter='\n', header=None).values
        # print(pd.read_csv(tgt_csv_path, delimiter=',').iloc[[0]].values)
        self.df_tgt = pd.read_csv(tgt_csv_path, delimiter=',', header=None).values

        # if args.toy:
        #     df = df.sample(frac=0.01)

        # If binary, no need to project to 2 dimensions (since using BCEWithLogitsLoss)
        self.num_classes = 1

        # if args.verbose:
        #     print(f"{split} number of examples: {len(self.df_src)}")
        # self.df_src = np.array([np.array(i[0].replace(" ", "").replace("'", "").split(',')) for i in self.df_src])
        self.vocab_w2i = {}
        self.vocab_i2w = {}
        index = 0
        self.encoded_df_src = []

        for i, row in enumerate(self.df_src):
            parsed_row = row[0].replace(" ", "").replace("'", "").split(',')
            encoded_row = np.zeros(len(parsed_row))
            for j, word in enumerate(parsed_row):
                # adding to vocab dictionaries
                if word not in self.vocab_w2i.keys():
                    self.vocab_w2i[word] = index
                    self.vocab_i2w[index] = word
                    index += 1
                # encoding strings to indexes
                encoded_row[j] = self.vocab_w2i[word]
            self.encoded_df_src.append(encoded_row)
        self.encoded_df_src = np.array(self.encoded_df_src)

        src_lengths = np.array([len(i) for i in self.encoded_df_src])
        self.max_src_len = np.amax(src_lengths)

        self.src_tensor = torch.FloatTensor(np.size(self.encoded_df_src), self.max_src_len).fill_(-1)
        for i, row in enumerate(self.encoded_df_src):
            self.src_tensor[i, :np.size(row)] = torch.from_numpy(row)

        # if args.weighted_loss:
        #     neg_weight = self.df_tgt.mean()
        #     self.weights = [neg_weight, 1 - neg_weight]

        #     if args.verbose:
        #         print(f"{split} weights: ", end="")
        #         print(*self.weights)

        #         p_count = (self.labels == 1).sum(axis = 0)[0]
        #         n_count = (self.labels == 0).sum(axis = 0)[0]
        #         total = p_count + n_count

        #         random_loss = (self.weights[0] * p_count + self.weights[1] * n_count) *\
        #                                            -np.log(0.5) / total
        #         print (f"{split} random loss: {random_loss}")

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_torch = Variable(torch.FloatTensor(weights_npy).cuda())

        loss = nn.functional.binary_cross_entropy_with_logits(prediction,
                                                              target,
                                                              weight=weights_torch)

        return loss

    def __getitem__(self, index):
        src = self.src_tensor[index]
        tgt = self.df_tgt[index]

        tgt = np.array([time.mktime(time.strptime(tgt[0], "%Y-%m-%d"))])
        tgt = torch.FloatTensor(tgt)

        return src, tgt

    def __len__(self):
        return len(self.df_src)


def load_data(args):
    train_dataset = Dataset(args, 'train')
    # valid_dataset = Dataset(args, 'valid')
    # test_dataset = Dataset(args, 'test')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset,
    #                                            batch_size=args.batch_size,
    #                                            num_workers=args.workers,
    #                                            shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                           batch_size=args.batch_size,
    #                                           num_workers=args.workers,
    #                                           shuffle=False)

    return train_loader  # , valid_loader, test_loader
