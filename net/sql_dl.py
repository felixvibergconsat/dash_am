#!/usr/bin/env python3

import pyodbc
import numpy as np
import os
import datetime
import sklearn.preprocessing as pp
import env_file

def process(d):
    min = np.min(d)
    d = d - min
    max = np.max(d)
    d = d / max
    d = d - 0.5
    d = d * 2
    return np.reshape(d, (-1, 1)), min, max


def get_day_and_time_of_day(date_and_time):
    now = datetime.datetime.now()
    days_ago = [now - x for x in date_and_time]
    time_of_day = [x.time() for x in date_and_time]
    seconds = [(x.hour * 60 + x.minute) * 60 + x.second for x in time_of_day]
    seconds = np.array(seconds)
    seconds_normed = seconds / np.max(seconds)
    days_ago = np.array(days_ago)
    days_ago_normed = days_ago / np.max(days_ago)
    return np.reshape(days_ago_normed, (-1, 1)), np.reshape(seconds_normed, (-1, 1))


def get_onehot_encoding(station, LabelEncoder, OneHotEncoder):
    int_enc = LabelEncoder.fit_transform(station)
    int_enc = int_enc.reshape(len(int_enc), 1)
    ohe = OneHotEncoder.fit_transform(int_enc)
    return ohe, [LabelEncoder.inverse_transform(x) for x in OneHotEncoder.categories_]


def get_cdf_per_designation(designations):
    cdf = np.zeros((np.size(designations), 200))
    for n in range(np.size(designations)):
        if np.random.rand() < 0.95:
            cdf[n, -1] = 1
        else:
            cdf[n, -1] = 0
    return cdf


def discard_rows_with_few_designation_copies(data, limit):
    unique_designations = np.unique(data[:, 0])
    inds_set = [np.where(data[:, 0] == unique_designation) for unique_designation in unique_designations]
    inds_set = [x for x in inds_set if len(x[0]) > limit]
    inds_set = [item for sublist in inds_set for item in sublist[0]]
    return data[inds_set]


def main():
    LabelEncoder = pp.LabelEncoder()
    OneHotEncoder = pp.OneHotEncoder(sparse=False)

    reload_sql_data = 1

    if reload_sql_data:
        print('downloading sql data...', end=' ')
        env_file.load('../env')
        conn = pyodbc.connect(
            'DRIVER={};SERVER={};UID={};PWD={}'.format(os.environ['DRIVER'], os.environ['SERVER'],
                                                       os.environ['DB_USER'], os.environ['DB_PASSWORD']))
        cursor = conn.cursor()
        sql_result = []
        cursor.execute("EXEC [KEPServer].[dbo].[ML_Base]")
        for row in cursor.fetchall():
            sql_result.append(row)
        np.save('data/ML_Base', sql_result, allow_pickle=True)
        print('done')

    print('reading data...', end=' ')
    data = np.load('data/ML_Base.npy', allow_pickle=True)
    data = data[data[:, 1] < 3600]
    print('done')
    print('discarding rows...', end=' ')
    data = discard_rows_with_few_designation_copies(data, 10)
    print('done')
    print('normalizing fields...', end=' ')
    bcn81, min_bcn81, max_bcn81 = process(data[:, 2].astype('float64'))
    ord10, min_ord10, max_ord10 = process(data[:, 3].astype('float64'))
    ird01, min_ird01, max_ird01 = process(data[:, 4].astype('float64'))
    bcw30, min_bcw30, max_bcw30 = process(data[:, 5].astype('float64'))
    bcm01, min_bcm01, max_bcm01 = process(data[:, 8].astype('float64'))
    days_ago_normed, seconds_normed = get_day_and_time_of_day(data[:, 9])
    onehot_station, station_categories = get_onehot_encoding(data[:, 10], LabelEncoder, OneHotEncoder)
    onehot_design, design_categories = get_onehot_encoding(data[:, 6], LabelEncoder, OneHotEncoder)
    onehot_glapp, glapp_categories = get_onehot_encoding(data[:, 7], LabelEncoder, OneHotEncoder)
    ct, min_ct, max_ct = process(data[:, 1])
    print('done')
    print('calculating CDF...')
    CDF = get_cdf_per_designation(data[:, 0])
    print('done')

    print('formatting and saving...', end=' ')
    source = np.concatenate((bcn81, ord10, ird01, bcw30, bcm01, days_ago_normed, seconds_normed, onehot_station,
                             onehot_design, onehot_glapp), axis=1)
    source_valid = source[CDF[:, -1] == 0]
    source_train = source[CDF[:, -1] == 1]
    target_valid = ct[CDF[:, -1] == 0]
    target_train = ct[CDF[:, -1] == 1]

    helpers = np.array([[min_bcn81, max_bcn81], [min_ord10, max_ord10], [min_ird01, max_ird01], [min_bcw30, max_bcw30],
                        [min_bcm01, max_bcm01], [min_ct, max_ct], [station_categories], [design_categories],
                        [glapp_categories]])

    np.save('../app/data/helpers', helpers)

    np.save('data/source_valid', source_valid)
    np.save('data/source_train', source_train)
    np.save('data/target_valid', target_valid)
    np.save('data/target_train', target_train)
    print('done')


if __name__ == "__main__":
    main()
