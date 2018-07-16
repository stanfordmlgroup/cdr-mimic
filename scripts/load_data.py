import numpy as np

data_folder = '/deep/group/data/squad_question_generation/'


def load_glove():
    """
    creates a dictionary mapping words to vectors from a file in glove format.
    """

    # dims available: 50, 100, 200, 300
    # num tokens available: 6B
    glove_path = '/deep/group/data/glove/glove.6B/glove.6B.50d.txt'
    with open(glove_path) as f:
        glove = {}
        for line in f.readlines():
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            glove[word] = vector
    return glove


def load_data():
    split = 'train'

    target_path = data_folder + 'redistribute/QG/train/train.txt.target.txt'
    source_path = data_folder + 'redistribute/QG/train/train.txt.source.txt'
    bio_path = data_folder + 'redistribute/QG/train/train.txt.bio'

    with open(source_path, 'r') as myfile:
        source = myfile.read()

        word_list_source = list(set(source.split()))
        source = source.split('\n')

    with open(target_path, 'r') as myfile:
        target = myfile.read()
        word_list_target = list(set(target.split()))
        target = target.split('\n')

    """
    BIO TAGS:
    B: word constitutes beginning of answer
    I: word is not B but part of the answer
    O: word is not part of the answer

    """
    with open(bio_path, 'r') as myfile:
        bio = myfile.read()

        bio = bio.split('\n')

    return source, target, bio


def main():
    source, target, bio = load_data()

    for i in range(3):
        print(f"Example no {i}")
        print("Source sentence: ")
        print(source[i])

        print("\n")

        print("Target question: ")
        print(target[i])

        print("bio: ")
        print(bio[i])

    glove = load_glove()

    print("\n")
    print("glove vector for cat: ", glove['cat'])


if __name__ == "__main__":
    main()
