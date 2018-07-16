import argparse
import numpy as np

from constants import *
from tqdm import tqdm


def preprocess_word_embs(glove_path, glove_dim, words_seen, output_path):
    """Reads from a GloVe-style .txt file and constructs an embedding matrix and
    mappings from words to word ids. The resulting embedding matrix only includes
    words seen in the example text, saving on memory so we can use the 840B corpus.
    This function produces a word embedding file, and writes it to output_path.
    Input:
      glove_path: path to glove.840B.{glove_dim}d.txt
      glove_dim: integer; needs to match the dimension in glove_path
      words_seen: set. Words seen in the example contexts and questions.
    """

    print("Loading GloVe vectors from file: %s" % glove_path)
    vocab_size = 2196017  # Estimated number of tokens with GloVe Common Crawl vectors
    emb_dict = {}
    glove_dict = {}
    # First pass: Go through glove vecs and add exact word matches.
    print("First pass: Adding exact matches...")
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split()
            word = "".join(line[0:-glove_dim])  # Word may have multiple components
            vector = list(map(float, line[-glove_dim:]))
            if word in words_seen:
                emb_dict[word] = vector
            glove_dict[word] = vector

    # Second pass: Go through glove vecs and add capitalization variants that we don't already have.
    print("Second pass: Adding capitalization variants...")
    for word, vector in tqdm(glove_dict.items(), total=len(glove_dict)):
        for variant in (word, word.lower(), word.capitalize(), word.upper()):
            if variant in words_seen and variant not in emb_dict:
                emb_dict[variant] = vector

    print("Found embeddings for {} out of {} words.".format(len(emb_dict), len(words_seen)))

    # Assign IDs to all words seen in the examples.
    pad_word = "__PAD__"
    unk_word = "__UNK__"
    word2id = {word: i for i, word in enumerate(emb_dict.keys(), NUM_RESERVED_IDS)}
    word2id[pad_word] = PAD_ID
    word2id[unk_word] = UNK_ID
    emb_dict[pad_word] = [0.0 for _ in range(glove_dim)]
    emb_dict[unk_word] = [0.0 for _ in range(glove_dim)]

    # Construct the embedding matrix and write to output file
    print("Creating word embedding file at {}...".format(output_path))
    id2word = {i: word for word, i in word2id.items()}
    with open(output_path, 'w') as fh:
        for i in range(len(id2word)):
            word = id2word[i]
            tokens = [word] + ["{:.5f}".format(x_i) for x_i in emb_dict[word]]
            fh.write(" ".join(tokens) + "\n")

    return word2id


def preprocess_char_embs(char_emb_size, chars_seen, output_path):
    """Constructs random character embeddings for all characters in chars_seen.
    Write a char embedding file to output_path.
    """

    print("Creating character embedding file at {}...".format(output_path))
    emb_dict = {}
    for char in chars_seen:
        emb_dict[char] = [np.random.normal(scale=0.1) for _ in range(char_emb_size)]

    # Assign IDs to all words seen in the examples.
    pad_char = "__PAD__"
    unk_char = "__UNK__"
    char2id = {word: i for i, word in enumerate(emb_dict.keys(), 2)}
    char2id[pad_char] = 0
    char2id[unk_char] = 1
    emb_dict[pad_char] = [0.0 for _ in range(char_emb_size)]
    emb_dict[unk_char] = [0.0 for _ in range(char_emb_size)]

    # Construct the embedding matrix and write to output file
    id2char = {i: word for word, i in char2id.items()}
    with open(output_path, 'w') as fh:
        for i in range(len(id2char)):
            word = id2char[i]
            tokens = [word] + ["{:.5f}".format(x_i) for x_i in emb_dict[word]]
            fh.write(" ".join(tokens) + "\n")

    return char2id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--word_emb_file", default="word_emb_file.txt")
    parser.add_argument("--word_emb_size", default=300)
    parser.add_argument("--char_emb_file", default="char_emb_file.txt")
    parser.add_argument("--char_emb_size", default=200)
    parser.add_argument("--glove_file", default="glove.840B.300d.txt")
    parser.add_argument("--max_w_len", default=16)
    parser.add_argument("--is_training", action='store_true')
    parser.add_argument("--include_words", action='store_true')

    raise NotImplementedError('Need to get words_seen, chars_seen from text files.')
