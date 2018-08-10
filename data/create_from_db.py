from collections import Counter, defaultdict

import time
from datetime import timedelta, datetime, date

import numpy as np
import pandas as pd

import csv

src_file_save = True
tgt_file_save = True

admissions_table_path = '/deep/group/med/mimic-iii/ADMISSIONS.csv'
diagnoses_icd_table_path = '/deep/group/med/mimic-iii/DIAGNOSES_ICD.csv'
patients_table_path = '/deep/group/med/mimic-iii/PATIENTS.csv'

SECONDS_PER_YEAR = 60 * 60 * 24 * 365.25
MANUAL_TTE_DELTA = 60 * 60 * 24 # seconds in one day

def get_icd(src_file_save=False, keep_only_hadm=True, keep_last_hadm=True):
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
                                    "is_dod": False if str(dod) == 'nan' else True,
                                    "is_dod_hosp": False if str(is_dod_hosp) == 'nan' else True,
                                }

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

    # DOBs are obscured if patients are ever older than 89 in the system, and set back 300 years
    obscure_age_seconds = 300 * (365 * 24 * 60 * 60)
    obscured_age_count = 0
    dod_hosp_count = 0

    # Stores the alive patients' last admission time { subject_id: last_dischtime }
    last_dischtimes = {}
    # Stores patients' admissions { subject_id: { hadm_id : dischtime }} 
    subjects_to_admissions = defaultdict(lambda: defaultdict(datetime))

    prelim_src = []
    prelim_src_ids = []
    remove_hadm_ids = []
    for row in icd_codes_py:
        subject_id = row[0]
        hadm_id = row[1]

        dischtime = admissions[hadm_id]['dischtime']
        dischtime = datetime.strptime(dischtime, "%Y-%m-%d %H:%M:%S")
        is_dod = patients[subject_id]['is_dod']
        if is_dod:
            # Dead patients: Remove hospital admissions where patients died in the hospital, ie. if dischtime is after patient deathtime
            dod = patients[subject_id]['dod']
            dod = datetime.strptime(dod, "%Y-%m-%d %H:%M:%S")
            if dischtime > dod:
                remove_hadm_ids.append(hadm_id)
                dod_hosp_count += 1
                continue
        else:
            # Add dichtime time to history
            subjects_to_admissions[subject_id][hadm_id] = dischtime

        # Age at which prediction was made: DISCHTIME (admissions.csv) - DOB (patients.csv)
        dob = patients[subject_id]['dob']
        dob = datetime.strptime(dob, "%Y-%m-%d %H:%M:%S")

        # Remove if dob was obscured (patient was 89+ yrs in db at some point)
        age_of_pred = dischtime - dob
        age_of_pred = age_of_pred.total_seconds()
        if age_of_pred > obscure_age_seconds:
            obscured_age_count += 1
            remove_hadm_ids.append(hadm_id)
            continue

        # Convert age of pred to years for numerical instability
        age_of_pred /= SECONDS_PER_YEAR

        # Do not include hadm_ids or subject_ids
        data_individual = list(patients[subject_id]['gender']) + [age_of_pred] + list(row[-1])
        prelim_src.append(data_individual)

        # Include hadm_ids and subject_ids here in master csv file
        data_individual_ids = [hadm_id, subject_id] + data_individual
        prelim_src_ids.append(data_individual_ids)

    """ Alive patients processing. Either we: 
        (a) Remove alive patient's last admission, even if it's their only one. We only want those with 2+ admissions. 
                Stored in remove_src_hadm_ids to be processed in src just below in this function, and also added here to remove_hadm_ids.
        (b) Pass these admissions for TTE calculation
                Stored in manual_tte_hadm_ids
        (c) Combination of (a) and (b) where we remove last admissions from patients with >1 visit, but keep those with only 1 admission
    """
    only_admission_count = 0
    last_admission_count = 0
    remove_src_hadm_ids = []
    manual_tte_hadm_ids = []
    for subject_id, hadm_to_admissions in subjects_to_admissions.items():
        hadm_ids = list(hadm_to_admissions.keys())
        admission_history = list(hadm_to_admissions.values())

        # Get last admission dischtime
        idx_last_hadm = np.argmax(admission_history)
        last_dischtimes[subject_id] = admission_history[idx_last_hadm]

        # Single admission patients (only admission)
        if len(hadm_ids) <= 1:
            only_admission_count += 1
            if keep_only_hadm:
                manual_tte_hadm_ids.append(hadm_ids[idx_last_hadm])
            else:
                remove_hadm_ids.append(hadm_ids[idx_last_hadm])
                remove_src_hadm_ids.append(hadm_ids[idx_last_hadm])
        # Last admissions from alive patients
        else:
            last_admission_count += 1
            if keep_last_hadm:
                manual_tte_hadm_ids.append(hadm_ids[idx_last_hadm])
            else:
                remove_hadm_ids.append(hadm_ids[idx_last_hadm])
                remove_src_hadm_ids.append(hadm_ids[idx_last_hadm])

    src = []
    src_ids = []
    # Save to final src vars, removing extra admissions (last or only) if applicable - these are flags we can toggle (keep_only_hadm and keep_last_hadm)
    if remove_src_hadm_ids:
        for i, row in enumerate(prelim_src_ids):
            hadm_id = row[0]
            if hadm_id in remove_src_hadm_ids:
                continue
            src_ids.append(row)
            src.append(prelim_src[i])
    else:
        src = prelim_src
        src_ids = prelim_src_ids
    
    # Save to files
    if src_file_save:
        np.save('src.npy', src)
        with open('src.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(src)

        np.save('src_master.npy', src_ids)
        with open('src_master.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['HADM_ID', 'SUBJECT_ID', 'GENDER', 'AGE_OF_PRED', 'ICD_CODES'])
            writer.writerows(src_ids)

    # Return hospital admission ids (hadm_ids) that need to be removed from targets (tgt.csv) file(s)
    counter_remove_hadm_ids = Counter(remove_hadm_ids)
    duplicate_count = sum(counter_remove_hadm_ids.values()) - len(counter_remove_hadm_ids)
    remove_hadm_ids_count = len(remove_hadm_ids) - duplicate_count
    n_admissions = len(admissions_py)
    print(f'{dod_hosp_count} ({dod_hosp_count/n_admissions*100:.2f}%) admissions removed b/c patients died in the hospital')
    print(f'{obscured_age_count} ({obscured_age_count/n_admissions*100:.2f}%) admissions removed b/c age was obscured')
    print(f'[Formerly] {only_admission_count} ({only_admission_count/n_admissions*100:.2f}%) admissions removed b/c were single admission patients who stayed alive')
    print(f'[Formerly] {last_admission_count} ({obscured_age_count/n_admissions*100:.2f}%) admissions removed b/c were alive patients last admissions')
    print(f'{remove_hadm_ids_count} ({remove_hadm_ids_count/n_admissions}%) admissions removed from dataset total {n_admissions} with {n_admissions - remove_hadm_ids_count} admissions remaining.')
    print(f'[Formerly] NB: {duplicate_count} of these admissions removed had 2 reasons for being removed')
    return remove_hadm_ids, last_dischtimes, manual_tte_hadm_ids


def get_tte(remove_hadm_ids, last_dischtimes, manual_tte_hadm_ids, tgt_file_save=False):
    patients_df = pd.read_csv(patients_table_path)
    admissions_df = pd.read_csv(admissions_table_path)

    admissions_df = admissions_df.sort_values(['SUBJECT_ID', 'HADM_ID'])
    
    patients_py = patients_df.values
    admissions_py = admissions_df.values

    # Get patient death info for is_alive col and computing tte col
    patients = {}
    for row in patients_py:
        subject_id = row[1]
        dod = row[4]

        patients[subject_id] = {
                                    "dod": dod,
                                    "is_alive": True if str(dod) == 'nan' else False,
                                }

    tgt = []
    tgt_ids = []
    for row in admissions_py:
        hadm_id = row[2]
        if hadm_id in remove_hadm_ids:
            continue

        subject_id = row[1]

        dischtime = row[4]
        dischtime = datetime.strptime(dischtime, "%Y-%m-%d %H:%M:%S")

        is_alive = patients[subject_id]['is_alive']

        # Calculate TTE (in years)
        if is_alive:
            # Alive: Get time to event (last encounter)
            event = last_dischtimes[subject_id]
            if hadm_id in manual_tte_hadm_ids:
                event += timedelta(seconds=MANUAL_TTE_DELTA)
        else:
            # Dead: Get time to event (mortality), from time of ICD code which is discharge time
            dod = patients[subject_id]['dod']
            dod = datetime.strptime(dod, "%Y-%m-%d %H:%M:%S")
            event = dod

        tte = event - dischtime
        tte = tte.total_seconds() / SECONDS_PER_YEAR
        if tte < 0:
            print(f'***ERROR TTE is negative\nEvent occurs at {event}, dischtime at {dischtime}\nis_alive {is_alive}\nhadm_id {hadm_id}, subject_id {subject_id}')
        
        # Do not include hadm_ids or subject_ids
        data_individual = [tte, int(is_alive)]
        tgt.append(data_individual)

        # Include hadm_ids and subject_ids here in master csv file
        data_individual_ids = [hadm_id, subject_id] + data_individual
        tgt_ids.append(data_individual_ids)

    # Save to files
    if tgt_file_save:
        np.save('tgt.npy', tgt)
        with open('tgt.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(tgt)

        np.save('tgt_master.npy', tgt_ids)
        with open('tgt_master.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['HADM_ID', 'SUBJECT_ID', 'TTE', 'IS_ALIVE'])
            writer.writerows(tgt_ids)
    
    return

def main():
    remove_hadm_ids, last_dischtimes, manual_tte_hadm_ids = get_icd(src_file_save)
    get_tte(remove_hadm_ids, last_dischtimes, manual_tte_hadm_ids, tgt_file_save)

if __name__ == "__main__":
    main()
