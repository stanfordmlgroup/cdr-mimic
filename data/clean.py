import numpy as np


def clean_tte(src_path="src.npy", tgt_path="tgt.npy"):
    """
    Remove rows where tte=0. Should be done before spliting dataset.

    Params:
        src_path: string path of numpy source dataset file
        tgt_path: string path of numpy target dataset file
    """
    num_demographics = 2

    src = np.load(src_path)
    tgt = np.load(tgt_path)

    cleaned_src, cleaned_tgt = [], []
    for src_row, tgt_row in zip(src, tgt):
        num_icd = len(src_row) - num_demographics
        is_female = src_row[0] == "F"  # Str M or F -> bool
        age = float(src_row[1])
        tte, is_alive = tgt_row
        if tte == 0:
            continue
        else:
            cleaned_src.append(src_row)
            cleaned_tgt.append(tgt_row)

    np.save('src_uncleaned.npy', src)
    np.save('tgt_uncleaned.npy', tgt)
    np.save('src.npy', np.array(cleaned_src))
    np.save('tgt.npy', np.array(cleaned_tgt))
        
def main():
    clean_tte()

if __name__ == "__main__":
    main()
