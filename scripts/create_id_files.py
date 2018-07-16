"""Create NumPy files of word and char IDs."""
import argparse
import h5py
import numpy as np
import os
import util

from constants import *
from tqdm import tqdm


def main(args):
    # Load word embedding matrix and char embedding matrix
    word_emb_path = os.path.join(args.data_dir, args.word_emb_file)
    word_emb_matrix, word2id = util.get_embeddings(word_emb_path, args.word_emb_size, 97572, 'word')
    print('Got {} word embeddings'.format(len(word2id)))

    char_emb_path = os.path.join(args.data_dir, args.char_emb_file)
    char_emb_matrix, char2id = util.get_embeddings(char_emb_path, args.char_emb_size, 94, 'char')
    print('Got {} char embeddings'.format(len(char2id)))

    bio2id = {'B': B_IN_BIO, 'I': I_IN_BIO, 'O': O_IN_BIO}

    for phase in ('train', 'dev'):
        # Read lines from downloaded files
        src_list, bio_list, tgt_list = load_data(args.data_dir, phase=phase)
        print('Read {} lines for the {} set'.format(len(src_list), phase))
        assert len(src_list) == len(bio_list) and len(bio_list) == len(tgt_list),\
            'src({}), bio({}), tgt({})'.format(len(src_list), len(bio_list), len(tgt_list))

        # Set up for mapping examples to word/char IDs
        n = len(src_list)
        max_c_len = args.max_c_len if phase == 'train' else args.max_c_len_test
        max_q_len = args.max_q_len if phase == 'train' else args.max_q_len_test
        max_w_len = args.max_w_len

        # Create empty arrays of padding
        src_ids = np.full((n, max_c_len), PAD_ID, dtype=np.int32)
        src_c_ids = np.full((n, max_c_len, max_w_len), PAD_ID, dtype=np.int32)
        bio_ids = np.full((n, max_c_len), O_IN_BIO, dtype=np.int32)
        tgt_ids = np.full((n, max_q_len), PAD_ID, dtype=np.int32)
        tgt_c_ids = np.full((n, max_q_len, max_w_len), PAD_ID, dtype=np.int32)

        # Fill arrays with IDs
        for i, (src, bio, tgt) in tqdm(enumerate(zip(src_list, bio_list, tgt_list)), total=n):
            src_words = src.split()[:max_c_len]
            src_ids[i, :len(src_words)] = [word2id.get(w, UNK_ID) for w in src_words]
            src_chars = [[c for c in s] for s in src_words]
            for j, chars in enumerate(src_chars):
                chars = chars[:max_w_len]
                src_c_ids[i, j, :len(chars)] = [char2id.get(c, UNK_ID) for c in chars]

            bio_words = bio.split()[:max_c_len]
            bio_ids[i, :len(bio_words)] = [bio2id[w] for w in bio_words]

            tgt_words = tgt.split()[:max_q_len]
            tgt_ids[i, :len(tgt_words)] = [word2id.get(w, UNK_ID) for w in tgt_words]
            tgt_chars = [[c for c in s] for s in tgt_words]
            for j, chars in enumerate(tgt_chars):
                chars = chars[:max_w_len]
                src_c_ids[i, j, :len(chars)] = [char2id.get(c, UNK_ID) for c in chars]

        # Save arrays filled with IDs
        with h5py.File(os.path.join(args.data_dir, 'data.hdf5'), 'a') as hdf5_fh:
            phase_group = hdf5_fh.create_group(phase)
            phase_group.create_dataset('src_ids'.format(phase), data=src_ids, chunks=True)
            phase_group.create_dataset('src_c_ids'.format(phase), data=src_c_ids, chunks=True)
            phase_group.create_dataset('bio_ids'.format(phase), data=bio_ids, chunks=True)
            phase_group.create_dataset('tgt_ids'.format(phase), data=tgt_ids, chunks=True)

        # Save embedding matrices
        word_emb_path = os.path.join(args.data_dir, 'word_embs.npy')
        np.save(word_emb_path, word_emb_matrix)

        char_emb_path = os.path.join(args.data_dir, 'char_embs.npy')
        np.save(char_emb_path, char_emb_matrix)


def load_data(data_dir, phase):
    """Load data from input files."""
    src_list = []
    bio_list = []
    tgt_list = []

    shuffle_str = '' if phase == 'train' else '.shuffle.dev'
    for lst, ext in [(tgt_list, 'target.txt'), (src_list, 'source.txt'), (bio_list, 'bio')]:
        file_path = os.path.join(data_dir, phase, '{}.txt{}.{}'.format(phase, shuffle_str, ext))
        with open(file_path, 'r') as fh:
            lst += fh.readlines()

    return src_list, bio_list, tgt_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='/deep/group/data/squad_question_generation/',
                        help='Path to data directory with question answering dataset.')
    parser.add_argument('--word_emb_file', type=str, default='word_emb_file.txt',
                        help='Name of pre-processed word embedding file.')
    parser.add_argument('--word_emb_size', type=int, default=300,
                        help='Size of a single word embedding.')
    parser.add_argument('--char_emb_file', type=str, default='char_emb_file.txt',
                        help='Name of pre-processed char embedding file.')
    parser.add_argument('--char_emb_size', type=int, default=200,
                        help='Size of a single char embedding.')
    parser.add_argument("--max_c_len", default=400, help='Maximum length (words) of context at train time.')
    parser.add_argument("--max_q_len", default=80, help='Maximum length (words) of question at train time.')
    parser.add_argument("--max_c_len_test", default=400, help='Maximum length (words) of context at test time.')
    parser.add_argument("--max_q_len_test", default=80, help='Maximum length (words) of question at test time.')
    parser.add_argument("--max_w_len", default=16, help='Maximum length (chars) of a word.')

    main(parser.parse_args())
