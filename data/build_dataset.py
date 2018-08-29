import time
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import pdb

SRC_FILE_NAME = 'src.csv'
TGT_FILE_NAME = 'tgt.csv'

# MIMIC Dataset
class Dataset():
    def __init__(self):
        datadir = Path('/deep/group/sharonz/cdr_mimic/data/original')

        src_csv_path = datadir / SRC_FILE_NAME
        tgt_csv_path = datadir / TGT_FILE_NAME

        self.df_src = pd.read_csv(src_csv_path, delimiter='\n', header=None).values
        self.df_tgt = pd.read_csv(tgt_csv_path, delimiter=',', header=None).values

        self.num_examples = len(self.df_src)
        print(f'Num examples is {self.num_examples}')
        vocab_index = 0
        self.vocab_w2i = {}
        self.vocab_i2w = {}
        self.encoded_src = []

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

        print(f'Vocab size is {self.vocab_size}')
        with open('vocab_w2i.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in self.vocab_w2i.items():
                writer.writerow([key, value])
        with open('vocab_i2w.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in self.vocab_i2w.items():
                writer.writerow([key, value])

        icd_code_lengths = np.array([len(i) for i in src_icd_codes])
        max_icd_codes_len = np.amax(icd_code_lengths)
        self.max_src_len = max_icd_codes_len + NUM_DEMOGRAPHICS

        icd_codes_arr = np.zeros((np.size(src_icd_codes), int(self.vocab_size)))
        for i, row in enumerate(src_icd_codes):
            row = np.array(row)
            icd_codes_arr[i, row] = 1.0

        self.src_arr = np.concatenate((src_demographics, icd_codes_arr), axis=1)

        np.savetxt("src_padded.csv", self.src_arr, fmt='%10.5f', delimiter=',')
        np.save('src_padded.npy', self.src_arr)

        np.savetxt("tgt_padded.csv", self.df_tgt, fmt='%10.5f', delimiter=',')
        np.save('tgt_padded.npy', self.df_tgt)


Dataset()