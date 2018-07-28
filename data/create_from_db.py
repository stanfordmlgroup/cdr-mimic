import time
from collections import Counter
from datetime import timedelta

import numpy as np
import pandas as pd

admissions_table_path = '/deep/group/med/mimic-iii/ADMISSIONS.csv'
chart_events_table_path = '/deep/group/med/mimic-iii/CHARTEVENTS.csv'
diagnoses_icd_table_path = '/deep/group/med/mimic-iii/DIAGNOSES_ICD.csv'
icustays_table_path = '/deep/group/med/mimic-iii/ICUSTAYS.csv'
patients_table_path = '/deep/group/med/mimic-iii/PATIENTS.csv'


def get_icd():
    def _group_icd(row):
        print(row)
        return row

    icd_df = pd.read_csv(diagnoses_icd_table_path)
    icd_df_grouped = icd_df.groupby(["SUBJECT_ID", "HADM_ID"],
                                    as_index=False)
    icd_df_grouped = icd_df_grouped['ICD9_CODE'].aggregate(lambda x: list(x))

    # Ensure to print out at least one patient who has >1 admissions, for debugging groupby
    print(icd_df_grouped.head(18))

    # Roll up past ICD codes for extra col in data
    subject_id = None
    # for row in icd_df_grouped:
    #     if subject_id != row['SUBJECT_ID']:
    #         subject_id = row['SUBJECT_ID']

    # np.save('src.npy', icd_df_grouped)
    # icd_df_grouped.to_csv("src.csv", encoding="utf-8", index=False)


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
    print(merged_df.head())

    np.save('tgt.npy', merged_df)
    merged_df.to_csv("tgt.csv", encoding="utf-8", index=False)


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

    print("Ratio patients to admissions is %s : %s, or %.0f%%" % (
        num_patients, num_admissions, (num_patients / float(num_admissions) * 100)))

    # How many patients are readmitted (> 1 admission)?
    patients = []
    for row in admissions_table:
        # SUBJECT_ID is the name of the column in db
        patient_id = row[1]
        patients.append(patient_id)
    counter = Counter(patients)
    readmits = [k for k, v in counter.iteritems() if float(v) > 1]
    readmit_counts = Counter([v for k, v in counter.iteritems() if float(v) > 1])
    print("There are {} readmitted patients out of {} total patients, or {:.0f}%. Readmission counts are {}".format(
        len(readmits), num_patients, len(readmits) / float(num_patients) * 100, readmit_counts))


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

    print(
        "Mortality in total is %s out of %s total patients (%.0f%%). %s are in-hospital (%.0f%% of deaths, %.0f%% of population). %s are out of hospital (%.0f%% of deaths, %.0f%% of population)."
        % (counts["DOD"], len(patients_table), (counts["DOD"] / float(len(patients_table)) * 100),
           counts["DOD_HOSP"], (counts["DOD_HOSP"] / float(counts["DOD"]) * 100),
           (counts["DOD_HOSP"] / float(len(patients_table)) * 100),
           counts["DOD_SSN"], (counts["DOD_SSN"] / float(counts["DOD"]) * 100),
           (counts["DOD_SSN"] / float(len(patients_table))) * 100))


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
    # get_tte()


if __name__ == "__main__":
    main()
