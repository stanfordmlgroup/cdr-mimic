import numpy as np
import os
import torch
import torch.nn as nn


from models.layers import *


class QANet(nn.Module):
    def __init__(self,
                 data_dir,
                 word_emb_file,
                 word_emb_size,
                 char_emb_size,
                 bio_emb_size,
                 alphabet_size,
                 max_c_len,
                 max_w_len,
                 num_highway_layers,
                 dropout_prob,
                 **kwargs):
        super(QANet, self).__init__()

        self.data_dir = data_dir
        self.word_emb_file = word_emb_file
        self.max_c_len = max_c_len
        self.max_w_len = max_w_len

        # Get word and char-level embeddings
        word_emb_path = os.path.join(self.data_dir, word_emb_file)
        self.word_embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.load(word_emb_path)), freeze=True)
        self.bio_embedding = nn.Embedding(3, bio_emb_size)
        self.char_embedding = nn.Sequential(nn.Embedding(alphabet_size, char_emb_size),
                                            CharLevelEncoder(char_emb_size, max_w_len))

        # Refine embeddings with highway network
        self.highway = HighwayNetwork(num_highway_layers, word_emb_size + char_emb_size + bio_emb_size, dropout_prob)

        #

    def forward(self, src, src_c, bio):
        # Embedding layer
        w_emb = self.word_embedding(src)
        c_emb = self.char_embedding(src_c)
        b_emb = self.bio_embedding(bio)
        x = torch.cat([w_emb, c_emb, b_emb], dim=-1)
        x = self.highway(x)

        # Encoding layer

        # Attention layer

        # Modeling layer

        return x

    def args_dict(self):
        args_dict = {
            'data_dir': self.data_dir,
            'word_emb_file': self.word_emb_file,
            'char_emb_file': self.char_emb_file,
            'max_c_len': self.max_c_len
        }

        return args_dict
