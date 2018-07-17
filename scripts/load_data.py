import numpy as np

data_folder = '/deep/group/med/mimic-iii/'


def load_data():
    split = 'train'

    target_path = data_folder + 'redistribute/QG/train/train.txt.target.txt'
    source_path = data_folder + 'redistribute/QG/train/train.txt.source.txt'

    with open(source_path, 'r') as myfile:
        source = myfile.read()

        word_list_source = list(set(source.split()))
        source = source.split('\n')

    with open(target_path, 'r') as myfile:
        target = myfile.read()
        word_list_target = list(set(target.split()))
        target = target.split('\n')

    return source, target


def main():
    source, target = load_data()

    for i in range(3):
        print(f"Example no {i}")
        print("Source row: ")
        print(source[i])

        print("\n")

        print("Target row: ")
        print(target[i])


if __name__ == "__main__":
    main()
