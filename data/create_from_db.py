from collections import Counter

import time
from datetime import timedelta, datetime, date

import numpy as np
import pandas as pd

import csv

admissions_table_path = '/deep/group/med/mimic-iii/ADMISSIONS.csv'
diagnoses_icd_table_path = '/deep/group/med/mimic-iii/DIAGNOSES_ICD.csv'
patients_table_path = '/deep/group/med/mimic-iii/PATIENTS.csv'

def get_icd():
    icd_df = pd.read_csv(diagnoses_icd_table_path)
    icd_df_grouped = icd_df.groupby(["SUBJECT_ID", "HADM_ID"],
                                    as_index=False)
    icd_df_grouped = icd_df_grouped['ICD9_CODE'].aggregate(lambda x: list(x))

    icd_df_grouped = icd_df_grouped.sort_values(["SUBJECT_ID", "HADM_ID"])

    # Ensure to print out at least one patient who has >1 admissions, for debugging groupby
    print(icd_df_grouped.head(18))

    # Assign patient's gender
    patients_df = pd.read_csv(patients_table_path)
    admissions_df = pd.read_csv(admissions_table_path)

    """ Convert to python list, b/c pandas encounters a memory error, with below code:
            def _include_gender(row):
                patient = patients_df.loc[patients_df["SUBJECT_ID"] == row["SUBJECT_ID"]]
                # print("row in patients.csv is", patient)
                # print("returning gender", patient["GENDER"])
                return patient["GENDER"]
            icd_df_grouped.apply(_include_gender, axis=1)
            print(icd_df_grouped.head(18))
    """
    icd_codes_py = icd_df_grouped.values
    patients_py = patients_df.values
    admissions_py = admissions_df.values
    print('patients_py', patients_py[0])
    print('icd_codes_py', icd_codes_py[0])
    print('admissions_py', admissions_py[0])

    patients = {}
    for row in patients_py:
        subject_id = row[1]
        gender = row[2]
        dob = row[3]
        dod_hosp = row[5]
        patients[subject_id] = {"gender": gender, "dob": dob}
    print('patients', patients[2])
    print('patients gender', patients[2]['gender'])

    """ NB: implemented to make it easy to later add other demographics 
            in admissions.csv. 

            Incl marital status, ethnicity, language, 
            insurance, admission type/location, discharge location
    """
    admissions = {}
    for row in admissions_py:
        hadm_id = row[2]
        dischtime = row[4]
        admissions[hadm_id] = {"dischtime": dischtime}

    data_all = []
    data_all_ids = []
    obscured_age_count = 0
    # DOBs are obscured if patients are ever older than 89 in the system, and set back 300 years
    obscure_age_seconds = 300 * (365 * 24 * 60 * 60)
    for row in icd_codes_py:
        subject_id = row[0]
        hadm_id = row[1]

        # Age at which prediction was made: DISCHTIME (admissions.csv) - DOB (patients.csv)
        dob = patients[subject_id]['dob']
        dischtime = admissions[hadm_id]['dischtime']
        dob = datetime.strptime(dob, "%Y-%m-%d %H:%M:%S")
        dischtime = datetime.strptime(dischtime, "%Y-%m-%d %H:%M:%S")

        age_of_pred = dischtime - dob
        age_of_pred = age_of_pred.total_seconds()
        if age_of_pred > obscure_age_seconds:
            # print('age was obscured. birth', dob, 'dischtime', dischtime)
            obscured_age_count += 1
            # We set this as a flag in the db - can process downstream as 89 yo or just ignore val (most likely)
            age_of_pred = -1

        # Do not include hadm_ids or subject_ids
        data_individual = list(patients[subject_id]['gender']) + list([age_of_pred]) + list(row[-1])
        data_individual_ids = list(row[:-1]) + data_individual
        data_all.append(data_individual)
        data_all_ids.append(data_individual_ids)

    print(f'{obscured_age_count/len(patients_py)*100:.2f}% of patients had their dobs obscured. Absolute nums are: {obscured_age_count}/{len(patients_py)}')
    print('data_all', data_all[0])
    
    np.save('src_master.npy', data_all_ids)
    with open('src_master.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data_all_ids)

    np.save('src.npy', data_all)
    with open('src.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data_all)
        
    # TODO LATER: Roll up past ICD codes for extra col in data
    # subject_id = None
    # for row in icd_df_grouped:
    #     if subject_id != row['SUBJECT_ID']:
    #         subject_id = row['SUBJECT_ID']


    # Using pandas df to save to files

    # np.save('src_master.npy', icd_df_grouped)
    # icd_df_grouped.to_csv("src_master.csv", encoding="utf-8", index=False)

    # select_icd_df_grouped = icd_df_grouped[['ICD9_CODE']]
    # np.save('src.npy', select_icd_df_grouped)
    # select_icd_df_grouped.to_csv("src.csv", encoding="utf-8", index=False)

    # num_samples_list = [3, 20]
    # for num_samples in num_samples_list:
    #     sample_icd_df_grouped = select_icd_df_grouped.head(num_samples)
    #     np.save('sample_src_' + str(num_samples) + '.npy', sample_icd_df_grouped)
    #     sample_icd_df_grouped.to_csv("sample_src_" + str(num_samples) + ".csv", encoding="utf-8", index=False)


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
    # get_tte()


if __name__ == "__main__":
    main()
