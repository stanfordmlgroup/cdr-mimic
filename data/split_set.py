import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

def main(path='/deep/group/sharonz/cdr_mimic/data/', train_frac=0.8, valid_frac=0.1, src_name='src_padded.npy', tgt_name='tgt_padded.npy'):
    X = np.load(path + src_name)
    y = np.load(path + tgt_name)

    print('datasets loaded')

    # Settings for split
    seed = 1

    # Get randomized indices TODO seed permutation
    num_examples = X.shape[0]
    rand_i = np.random.RandomState(seed=seed).permutation(num_examples)
    X = X[rand_i]
    y = y[rand_i]
    train_i = int(num_examples*train_frac)
    valid_i = int(num_examples*valid_frac)

    x_train = X[:train_i]
    x_valid = X[train_i:train_i+valid_i]
    x_test = X[train_i+valid_i:]

    y_train = y[:train_i]
    y_valid = y[train_i:train_i+valid_i]
    y_test = y[train_i+valid_i:]

    print('splits made')

    # import pdb
    # pdb.set_trace()

    np.save('train_src.npy', x_train)
    np.save('valid_src.npy', x_valid)
    np.save('test_src.npy', x_test)

    np.save('train_tgt.npy', y_train)
    np.save('valid_tgt.npy', y_valid)
    np.save('test_tgt.npy', y_test)

    np.savetxt('train_src.csv', x_train, fmt='%10.5f', delimiter='\t')
    np.savetxt('valid_src.csv', x_valid, fmt='%10.5f', delimiter='\t')
    np.savetxt('test_src.csv', x_test, fmt='%10.5f', delimiter='\t')

    np.savetxt('train_tgt.csv', y_train, fmt='%10.5f', delimiter='\t')
    np.savetxt('valid_tgt.csv', y_valid, fmt='%10.5f', delimiter='\t')
    np.savetxt('test_tgt.csv', y_test, fmt='%10.5f', delimiter='\t')



if __name__ == "__main__":
    main()
