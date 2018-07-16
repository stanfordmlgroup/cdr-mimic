import argparse
import numpy as np

from tqdm import tqdm
from sys import stderr


def args_to_list(csv, allow_empty, arg_type=int, allow_negative=True):
    """Convert comma-separated arguments to a list.

    Args:
        csv: Comma-separated list of arguments as a string.
        allow_empty: If True, allow the list to be empty. Otherwise return None instead of empty list.
        arg_type: Argument type in the list.
        allow_negative: If True, allow negative inputs.

    Returns:
        List of arguments, converted to `arg_type`.
    """
    arg_vals = [arg_type(d) for d in str(csv).split(',')]
    if not allow_negative:
        arg_vals = [v for v in arg_vals if v >= 0]
    if not allow_empty and len(arg_vals) == 0:
        return None
    return arg_vals


def print_err(*args, **kwargs):
    """Print a message to stderr."""
    print(*args, file=stderr, **kwargs)


def str_to_bool(arg):
    """Convert an argument string into its boolean value.

    Args:
        arg: String representing a bool.

    Returns:
        Boolean value for the string.
    """
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_embeddings(emb_path, emb_length, vocab_size, embedding_type):
    """Read from preprocessed GloVe file and return embedding matrix and
    mappings from words to word ids.

    Args:
        emb_path: Path to preprocessed glove file.
        emb_length: Dimensionality of an embedding.
        vocab_size: Expected number of lines in the embedding file.
        embedding_type: One of 'word', 'char'.

    Returns:
        emb_matrix: Numpy array shape (vocab_size, vec_size) containing word embeddings.
            Only includes embeddings for words that were seen in the dev/train sets.
        str2id: dictionary mapping word/char to the corresponding embedding ID.
    """
    print("Loading {} embeddings from file: {}...".format(embedding_type, emb_path))

    emb_matrix = []
    str2id = {}
    idx = 0
    with open(emb_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            if emb_length != len(vector):
                raise Exception(
                    "{}: Expected vector of size {}, but got vector of size {}.".format(idx, emb_length, len(vector)))
            emb_matrix.append(vector)
            str2id[word] = idx
            idx += 1

    emb_matrix = np.array(emb_matrix, dtype=np.float32)
    print("Loaded {} embedding matrix with shape {}.".format(embedding_type, emb_matrix.shape))

    return emb_matrix, str2id
