import numpy as np


def generate_stats(src_path="src.npy", tgt_path="tgt.npy"):
    """
    Calculate:
    - percent censored vs uncensored 
    - mean time to censoring
    - mean time to death
    - mean num icd codes in censored
    - mean num icd codes in uncensored

    Params:
        src_path: string path of numpy source dataset file
        tgt_path: string path of numpy target dataset file
    """
    num_demographics = 2

    src = np.load(src_path)
    tgt = np.load(tgt_path)


    is_alive_count = 0
    is_dead_count = 0
    total_tte_alive = 0
    total_tte_dead = 0
    total_icd_alive = 0
    total_icd_dead = 0
    female_alive_count = 0
    female_dead_count = 0
    male_alive_count = 0
    male_dead_count = 0
    total_age_alive = 0
    total_age_dead = 0
    for src_row, tgt_row in zip(src, tgt):
        num_icd = len(src_row) - num_demographics
        is_female = src_row[0] == "F"  # Str M or F -> bool
        age = float(src_row[1])
        tte, is_alive = tgt_row

        # Calculate statistics on percent censored and mean time to censoring / mortality
        if is_alive:
            total_tte_alive += tte
            is_alive_count += 1
            total_icd_alive += num_icd
            female_alive_count += is_female
            male_alive_count += not is_female
            total_age_alive += age

        else:
            total_tte_dead += tte
            is_dead_count += 1
            total_icd_dead += num_icd
            female_dead_count += is_female
            male_dead_count += not is_female
            total_age_dead += age
        


    # Print final stats
    print("Number censored:", is_alive_count)
    print("Number uncensored:", is_dead_count)
    print("Percent censored:", 100 * is_alive_count / float(is_alive_count + is_dead_count))
    print("Mean time to censoring:", total_tte_alive / float(is_alive_count))
    print("Mean time to mortality:", total_tte_dead / float(is_dead_count))
    print("Mean num icd codes in censored:", total_icd_alive / float(is_alive_count))
    print("Mean num icd codes in uncensored:", total_icd_dead / float(is_dead_count))
    print("Mean age in censored:", total_age_alive / float(is_alive_count))
    print("Mean age in uncensored:", total_age_dead / float(is_dead_count))
    print("Percent female in censored:", 100 * female_alive_count / float(is_alive_count))
    print("Percent female in uncensored:", 100 * female_dead_count / float(is_dead_count))


def main():
    generate_stats()

if __name__ == "__main__":
    main()
