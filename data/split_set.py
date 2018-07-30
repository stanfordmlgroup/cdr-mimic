import pandas as pd
import numpy as np

def split_within_file(df, frac):
        msk = np.random.rand(len(df)) < frac

        train = df[msk]
        valid = df[~msk]

        return train, valid


def main(path='/deep/group/sharonz/cdr_mimic/data/', split_frac=0.8):
	src_ext_path = 'sample_src_3.csv'
	tgt_ext_path = 'sample_tgt_3.csv'
	df_src = pd.read_csv(path + src_ext_path, delimiter='\n', header=None).values
    df_tgt = pd.read_csv(path + tgt_ext_path, delimiter=',').values

    src_train, src_valid = split_within_file(df_src, split_frac)
    tgt_train, tgt_valid = split_within_file(df_tgt, split_frac)

    src_train.to_csv(path + 'train_' + src_ext_path, delimiter="\n", index=False)
    src_valid.to_csv(path + 'valid_' + src_ext_path, delimiter=",", index=False)

    tgt_train.to_csv(path + 'train_' + tgt_ext_path, delimiter="\n", index=False)
    tgt_valid.to_csv(path + 'valid_' + tgt_ext_path, delimiter=",", index=False)

if __name__ == "__main__":
    main()
