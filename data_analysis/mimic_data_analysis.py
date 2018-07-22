import numpy as np
from numpy import genfromtxt

from collections import Counter, defaultdict

import csv
import time
from datetime import timedelta

import pandas as pd

admissions_table_path = '/deep/group/med/mimic-iii/ADMISSIONS.csv'
chart_events_table_path = '/deep/group/med/mimic-iii/CHARTEVENTS.csv'
diagnoses_icd_table_path = '/deep/group/med/mimic-iii/DIAGNOSES_ICD.csv'
icustays_table_path = '/deep/group/med/mimic-iii/ICUSTAYS.csv'
patients_table_path = '/deep/group/med/mimic-iii/PATIENTS.csv'

def get_icd():
    icd_df = pd.read_csv(diagnoses_icd_table_path)
    print(icd_df.sort_values("HADM_ID").head())
    icd_df_grouped_admissions = icd_df.groupby(["HADM_ID", "SUBJECT_ID"])["ICD9_CODE"].apply(list)
    icd_df_grouped_subjects = icd_df.groupby(["SUBJECT_ID"])["ICD9_CODE"].apply(list)
    print(icd_df_grouped_admissions.head())
    print(icd_df_grouped_subjects.head(20))
    icd_df_grouped_subjects.to_csv("icd_subject_source.csv", encoding="utf-8", index=False)

def get_dods():
    patients_df = pd.read_csv(patients_table_path)
    patients_df = patients_df.sort_values("SUBJECT_ID")
    subject_ids = patients_df["SUBJECT_ID"]
    dobs = np.array(patients_df['DOB']).astype('datetime64[D]')
    print('dobs\n', dobs[:5])
   
    MAX_AGE_YEARS = 120*365
    max_ages = [np.timedelta64(MAX_AGE_YEARS, 'D')] * len(dobs)
    print(max_ages[:5])

    max_dods = dobs + max_ages
    # max_dods = max_dods.astype(str).astype(int)
    max_dods = [time.strftime(x) for x in max_dods.astype(str)]

    print('maxes\n', max_dods[:5])
    # print(max_dods.dtype)
    # print(max_dods.dtype)
    # print(patients_df['DOD'].dtypes)
    print('init dods\n', patients_df['DOD'].head())
    max_dods_dict = {k: v for k, v in enumerate(max_dods)}
    dods = patients_df['DOD'].astype('int32', errors='ignore').fillna(max_dods_dict)
    print('final dods\n', dods.head())
    dods.to_csv("icd_subject_target.csv", encoding="utf-8", index=False)
    # combined = pd.concat([subject_ids, dods], axis=1, keys=["SUBJECT_ID", "DOD"])
    # print(combined.head())
    # combined.to_csv("labels.csv", encoding="utf-8", index=False)

def admission(n_days_since_admission):
    # for each hospital admission, get datetime of 2nd day and use that for ICD diagnosis
    # store per admission (todo: aggregate per patient)
    admissions_df = pd.read_csv(admissions_table_path)
    admissions_df['ADMITTIME'] = pd.to_datetime(admissions_df['ADMITTIME'])

    admissions_df['NDAYSADMIT'] = admissions_df['ADMITTIME'] + timedelta(days=n_days_since_admission)

    diagnoses_df = pd.read_csv(diagnoses_icd_table_path)
    joined_df = pd.merge(admissions_df, diagnoses_df, on='HADM_ID', how='outer', suffixes=('_admissions', '_diagnoses'))
    joined_df = joined_df.groupby(['HADM_ID'])
    print(joined_df.head())


def dataset_datetime_range():
    admissions_df = pd.read_csv(admissions_table_path)
    # print(list(admissions_df.columns.values))
    # print(admissions_df.head())
    # print(admissions_df.head().sort_values(['ADMITTIME'], ascending=True))
    admissions_df['ADMITTIME'] = pd.to_datetime(admissions_df['ADMITTIME'])
    print(admissions_df['ADMITTIME'].max(), admissions_df['ADMITTIME'].min(), 
        'range of dates:', admissions_df['ADMITTIME'].max() - admissions_df['ADMITTIME'].min())
    # for index, row in admissions_df.iterrows():
    #     print(row['ADMITTIME'])
    # print(admissions_df[0], admissions_df[-1])

def chart_events_stats():
    start_time = time.time()
    df = pd.read_csv(chart_events_table_path)
    read_time = time.time() - start_time
    print("Time to read csv: {}".format(read_time))
    df.to_pickle('chartevents.pkl')
    # store = pd.HDFStore('chartevents.h5')
    # df = pd.read_csv(chart_events_table_path)
    # print("completed read")
    # store['df'] = df
    # store.close()
    print('done')


    # df_tl.to_hdf('store_tl.h5','table', append=True)
    # f = open(chart_events_table_path)
    # h = f.readline()
    # events = []
    # counter = defaultdict(int)
    # for i, l in enumerate(f):
    #     y = [str(x).strip('\n').strip('"') for x in l.split(',')]
    #     events.append(y[4])
    #     counter[y[4]] += 1
    #     if i % 1000000 == 0:
    #         print(sorted(counter.items(), key=lambda x: x[1])[-5:])
    # f.close()

    # chart_events_table = sparse_loadtxt(chart_events_table_path)
    # events = []
    # for row in chart_events_table:
    #   print(row[8])
    #   events.append(row[8])
    # events = [r[8] for r in chart_events_table]
    # print(len(events))
    # counter = Counter(events).most_common(3)
    # counter = Counter(events)
    # print("Chart events: {}\n Total number of events recorded: {}".format(counter, len(chart_events_table)))
    # print("Chart events (most common 3): {}\n Total number of events recorded: {}".format(counter, len(chart_events_table)))

def diagnoses_stats():
    diagnoses_icd_table = sparse_loadtxt(diagnoses_icd_table_path)
    icd_codes = [r[-1] for r in diagnoses_icd_table]
    counter = Counter(icd_codes)
    print("ICD codes: {}\n Total number of diagnoses recorded: {}".format(counter, len(diagnoses_icd_table)))

# How long are average lengths of stay? How much information do we have on each patient across this stay - are recorded events in the MIMIC db sparse?
def record_frequency_stats():
    icustays_table = sparse_loadtxt(icustays_table_path)

    stays = []
    for row in icustays_table:
        try:
            stays.append(float(row[-1]))
        except:
            print('Error: no LOS in db', row[-1])
    counter = Counter([int(s) for s in stays])
    print("Lengths of stay (days): {}. Average length of stay: {}".format(counter, np.mean(stays)))

    # TODO: how much info across db do we have on patient and how frequent is the interaction/recording (timestamping) of it? To determine if we can use RNN
    

# Should we look at the data at a certain interval, eg. each day in the EMR? Only if there's a lot of information on the same patients - which means # patients << # admissions
def patient_to_admissions_ratio():
    patients_table = sparse_loadtxt(patients_table_path)
    admissions_table = sparse_loadtxt(admissions_table_path)

    num_patients = len(patients_table)
    num_admissions = len(admissions_table)

    print("Ratio patients to admissions is %s : %s, or %.0f%%" % (num_patients, num_admissions, (num_patients / float(num_admissions) * 100)))

    # How many patients are readmitted (> 1 admission)?
    patients = []
    for row in admissions_table:
        # SUBJECT_ID is the name of the column in db
        patient_id = row[1]
        patients.append(patient_id)
    counter = Counter(patients)
    readmits = [k for k,v in counter.iteritems() if float(v) > 1]
    readmit_counts = Counter([v for k,v in counter.iteritems() if float(v) > 1])
    print("There are {} readmitted patients out of {} total patients, or {:.0f}%. Readmission counts are {}".format(len(readmits), num_patients, len(readmits) / float(num_patients) * 100, readmit_counts))

# Can we use mortality as our event? What makes it good/bad or comparable to other events like heart attack, etc.?
def mortality_stats():
    # Rows for date of death columns
    DOD = 4
    DOD_HOSP = 5
    DOD_SSN = 6

    # patients_table = genfromtxt(patients_table_path)
    patients_table = sparse_loadtxt(patients_table_path)
    counts = {"DOD": 0, "DOD_HOSP": 0, "DOD_SSN": 0}
    for row in patients_table:
        dod = row[DOD]
        dod_hosp = row[DOD_HOSP]
        dod_ssn = row[DOD_SSN]

        """ Data cleansing checks: 
                (1) DOD_HOSP and DOD_SSN exist if only if DOD exists (never without a DOD value and a DOD value either has DOD_HOSP, DOD_SSN, or both)
                (2) DOD overrides as DOD_HOSP if both DOD_HOSP and DOD_SSN exist -- there is 1 error -- based on the MIMIC dataset description online (see code on "incorrect override time" below to inspect error)
        """
        if dod:
            counts["DOD"] += 1
            if dod_hosp:
                counts["DOD_HOSP"] += 1
            # Counting specifically deaths outside of hospital (not incl in-hospital deaths)
            elif dod_ssn:
                counts["DOD_SSN"] += 1
            # Checking for error (1) -- none found here
            else:
                print("Error: dod recorded but no dod_hosp and/or dod_ssn?", dod, dod_hosp, dod_hosp)

            # Checking for error (2), incorrect override time -- shows 1 row's error in dataset
            if dod_ssn and dod_hosp and dod_hosp != dod_ssn and dod != dod_hosp:
                print("Error: incorrect override time", dod, dod_hosp, dod_ssn)

    print("Mortality in total is %s out of %s total patients (%.0f%%). %s are in-hospital (%.0f%% of deaths, %.0f%% of population). %s are out of hospital (%.0f%% of deaths, %.0f%% of population)." 
        % (counts["DOD"], len(patients_table), (counts["DOD"] / float(len(patients_table)) * 100), 
            counts["DOD_HOSP"], (counts["DOD_HOSP"] / float(counts["DOD"]) * 100), (counts["DOD_HOSP"] / float(len(patients_table)) * 100),
            counts["DOD_SSN"], (counts["DOD_SSN"] / float(counts["DOD"]) * 100), (counts["DOD_SSN"] / float(len(patients_table))) * 100))

# Load sparse csv's into matrices
def sparse_loadtxt(file):
    f = open(file)
    h = f.readline()
    ll = []
    for l in f:
        y = [str(x).strip('\n').strip('"') for x in l.split(',')]
        ll.append(y)
    # ll = np.array(ll)
    f.close()
    return ll

def main():
    # record_frequency_stats()
    # admission(2)
    get_icd()
    get_dods()

if __name__ == "__main__":
    main()

