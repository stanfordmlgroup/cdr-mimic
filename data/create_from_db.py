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

    """ Convert all df to python list, b/c pandas encounters a memory error, with below code:
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
    print('patients_py eg', patients_py[0])
    print('icd_codes_py eg', icd_codes_py[0])
    print('admissions_py eg', admissions_py[0])

    """ Information from the patients table that we want

        gender: str, 'M' or 'F'
        dob (date of birth): str, in format to convert to datetime

    """

    patients = {}
    for row in patients_py:
        subject_id = row[1]
        gender = row[2]
        dob = row[3]
        dod = row[4]
        is_dod_hosp = row[5]
        patients[subject_id] = {
                                    "gender": gender, 
                                    "dob": dob, 
                                    "dod": dod,
                                    "is_dod_hosp": False if str(is_dod_hosp) == 'nan' else True,
                                }
    print('patients eg', patients[2])
    print('patients gender eg', patients[2]['gender'])
    print('patients dob eg', patients[2]['dob'])
    print('patients dod eg', patients[2]['dod'])

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

    src = []
    src_ids = []
    obscured_age_count = 0
    dod_hosp_hadm_ids = []
    dod_hosp_count = 0
    # DOBs are obscured if patients are ever older than 89 in the system, and set back 300 years
    obscure_age_seconds = 300 * (365 * 24 * 60 * 60)
    for row in icd_codes_py:
        subject_id = row[0]
        hadm_id = row[1]

        # Remove hospital admissions where patients died in the hospital: if dischtime is after patient deathtime
        dod = patients[subject_id]['dod']
        dod = datetime.strptime(dod, "%Y-%m-%d %H:%M:%S")
        dischtime = admissions[hadm_id]['dischtime']
        dischtime = datetime.strptime(dischtime, "%Y-%m-%d %H:%M:%S")
        if dischtime > dod:
            dod_hosp_hadm_ids.append(hadm_id)
            dod_hosp_count += 1

        # Double check if patient died in hospital at this hadm (hospital admission)
        # by seeing if this hadm is their last && they have a is_dod_hosp flag
        # if patients[subject_id]['is_dod_hosp']:
            # TODO HERE: reverse the icd_codes_py list (then re-reverse back the src list) to figure out it it's the last hospital admission

        # Remove 

        # Age at which prediction was made: DISCHTIME (admissions.csv) - DOB (patients.csv)
        dob = patients[subject_id]['dob']
        dob = datetime.strptime(dob, "%Y-%m-%d %H:%M:%S")

        age_of_pred = dischtime - dob
        age_of_pred = age_of_pred.total_seconds()
        if age_of_pred > obscure_age_seconds:
            # print('age was obscured. birth', dob, 'dischtime', dischtime)
            obscured_age_count += 1
            # We set this as a flag in the db - can process downstream as 89 yo or just ignore val (most likely)
            age_of_pred = -1

        # Do not include hadm_ids or subject_ids
        data_individual = list(patients[subject_id]['gender']) + list([age_of_pred]) + list(row[-1])
        src.append(data_individual)

        # Include hadm_ids and subject_ids here in master csv file
        data_individual_ids = list(row[:-1]) + data_individual
        src_ids.append(data_individual_ids)

    print(f'{obscured_age_count/len(patients_py)*100:.2f}% of patients had their dobs obscured. Absolute nums are: {obscured_age_count}/{len(patients_py)}')
    print('src eg', src[0])
    
    # Save to files
    # np.save('src_master.npy', src_ids)
    # with open('src_master.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(src_ids)

    # np.save('src_disch_alive.npy', src)
    with open('src_disch_alive.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(src)
    
    # Return patient/subject ids that need to be removed from targets (tgt.csv) file(s)
    print(f'{dod_hosp_count} patients died in the hospital - to be removed from dataset')
    return dod_hosp_hadm_ids

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


def get_tte(dod_hosp_hadm_ids):
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

    merged_py = merged_df.values
    print('merged_py eg', merged_py[0])

    # Remove patients who died in the hospital
    tgt_disch_alive = []
    tgt_disch_alive_ids = []
    for row in merged_py:
        subject_id = row[2]
        if subject_id in dod_hosp_hadm_ids:
            continue

        # Do not include hadm_ids and subject_ids
        tgt_individual = list(row[:2])
        tgt_disch_alive.append(tgt_individual)
        
        # Include hadm_ids and subject_ids here in master csv file
        tgt_individual_ids = list(row)
        tgt_disch_alive_ids.append(tgt_individual_ids)
    print('tgt_disch_alive eg', tgt_disch_alive[0])
    
    # Save to files
    # np.save('tgt_master.npy', tgt_disch_alive_ids)
    # with open('tgt_master.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(tgt_disch_alive_ids)

    # np.save('tgt_disch_alive.npy', tgt_disch_alive)
    with open('tgt_disch_alive.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(tgt_disch_alive)
    
    return 
    # Using pandas df
    # np.save('tgt_master.npy', merged_df)
    # merged_df.to_csv("tgt_master.csv", encoding="utf-8", index=False)

    # select_merged_df = merged_df[['TTE', 'IS_ALIVE']]
    # np.save('tgt.npy', select_merged_df)
    # select_merged_df.to_csv("tgt.csv", encoding="utf-8", index=False)

    # num_samples_list = [3, 20]
    # for num_samples in num_samples_list:
    #     sample_merged_df = select_merged_df.head(num_samples)
    #     np.save('sample_tgt_' + str(num_samples) + '.npy', sample_merged_df)
    #     sample_merged_df.to_csv("sample_tgt_" + str(num_samples) + ".csv", encoding="utf-8", index=False)


def main():
    dod_hosp_hadm_ids = get_icd()
    get_tte(dod_hosp_hadm_ids)


if __name__ == "__main__":
    main()
