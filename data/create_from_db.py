import time
from collections import Counter
from datetime import timedelta

import numpy as np
import pandas as pd

admissions_table_path = '/deep/group/med/mimic-iii/ADMISSIONS.csv'
diagnoses_icd_table_path = '/deep/group/med/mimic-iii/DIAGNOSES_ICD.csv'
patients_table_path = '/deep/group/med/mimic-iii/PATIENTS.csv'

def get_icd():
    def _group_icd(row):
        print(row)
        return row

    icd_df = pd.read_csv(diagnoses_icd_table_path)
    icd_df_grouped = icd_df.groupby(["SUBJECT_ID", "HADM_ID"],
                                    as_index=False)
    icd_df_grouped = icd_df_grouped['ICD9_CODE'].aggregate(lambda x: list(x))

    icd_df_grouped = icd_df_grouped.sort_values(["SUBJECT_ID", "HADM_ID"])

    # Ensure to print out at least one patient who has >1 admissions, for debugging groupby
    print(icd_df_grouped.head(18))

    # TODO: Roll up past ICD codes for extra col in data
    # subject_id = None
    # for row in icd_df_grouped:
    #     if subject_id != row['SUBJECT_ID']:
    #         subject_id = row['SUBJECT_ID']

    np.save('src_master.npy', icd_df_grouped)
    icd_df_grouped.to_csv("src_master.csv", encoding="utf-8", index=False)

    select_icd_df_grouped = icd_df_grouped[['ICD9_CODE']]
    np.save('src.npy', select_icd_df_grouped)
    select_icd_df_grouped.to_csv("src.csv", encoding="utf-8", index=False)

    num_samples_list = [3, 20]
    for num_samples in num_samples_list:
        sample_icd_df_grouped = select_icd_df_grouped.head(num_samples)
        np.save('sample_src_' + str(num_samples) + '.npy', sample_icd_df_grouped)
        sample_icd_df_grouped.to_csv("sample_src_" + str(num_samples) + ".csv", encoding="utf-8", index=False)


def get_tte():
    # Get time to event (mortality), from time of ICD code which is discharge time
    def _tte(row):
        if pd.isna(row['DOD']):
            return 0
        return max(0, time.mktime(time.strptime(row['DOD'], "%Y-%m-%d %H:%M:%S")) -
                   time.mktime(time.strptime(row['DISCHTIME'], "%Y-%m-%d %H:%M:%S")))
    # Bool for alive/dead
    def _is_alive(row):
        return 1 if pd.isna(row['DOD']) else 0

    patients_df = pd.read_csv(patients_table_path)
    admissions_df = pd.read_csv(admissions_table_path)
    merged_df = pd.merge(patients_df, admissions_df, on='SUBJECT_ID', sort=True)

    merged_df['TTE'] = merged_df.apply(lambda row: _tte(row), axis=1)
    merged_df['IS_ALIVE'] = merged_df.apply(lambda row: _is_alive(row), axis=1)

    merged_df = merged_df[['TTE', 'IS_ALIVE', 'SUBJECT_ID', 'HADM_ID']]
    merged_df = merged_df.sort_values(["SUBJECT_ID", "HADM_ID"])
    print(merged_df.head())

    np.save('tgt_master.npy', merged_df)
    merged_df.to_csv("tgt_master.csv", encoding="utf-8", index=False)

    select_merged_df = merged_df[['TTE', 'IS_ALIVE']]
    np.save('tgt.npy', select_merged_df)
    select_merged_df.to_csv("tgt.csv", encoding="utf-8", index=False)

    num_samples_list = [3, 20]
    for num_samples in num_samples_list:
        sample_merged_df = select_merged_df.head(num_samples)
        np.save('sample_tgt_' + str(num_samples) + '.npy', sample_merged_df)
        sample_merged_df.to_csv("sample_tgt_" + str(num_samples) + ".csv", encoding="utf-8", index=False)


def main():
    get_icd()
    get_tte()


if __name__ == "__main__":
    main()
