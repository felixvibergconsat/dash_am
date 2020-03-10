import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
import pyodbc
import pandas as pd
import os
from app_functions import *

with open('../env') as env_data:
    for line in env_data:
        s = str(line).split('=')
        os.environ[s[0]] = s[1][:-1]

for station in ['AM1', 'AM2']:

    reload = True
    if reload:
        query = "SELECT * FROM [KEPServer].[arkiv].[PLC_Alarms] WHERE Machine = '{}' ORDER BY TIMESTAMP desc".format(station)
        data = exec_sql(os.environ, query)
        np.save('data/AM_Larm_{}'.format(station), data)

        query = "EXEC KEPServer.dbo.AM_BearingData_Dimensions {}".format(station)
        data = exec_sql(os.environ, query)
        np.save('data/AM_BearingData_Dimensions_{}'.format(station), data)

    bearing_data = np.load('data/AM_BearingData_Dimensions_{}.npy'.format(station), allow_pickle=True)
    six_months_ago = datetime.datetime.today() - datetime.timedelta(6*365/12)
    bearing_data = bearing_data[bearing_data[:, 0] > six_months_ago]
    alarms = np.load('data/AM_Larm_{}.npy'.format(station), allow_pickle=True)
    bearing_data = bearing_data[bearing_data[:, 0] != None]
    alarm_strings = np.unique(alarms[:, 1])
    alarm_strings = [s.replace('Assembly.{}-PLC.Alarms.'.format(station), '') for s in alarm_strings]
    alarm_strings = [s.replace('Assembly.{}-PLC.LarmFelvändRulle'.format(station), 'FV_Rulle') for s in alarm_strings]
    alarm_strings = [s for s in alarm_strings if 1 < len(s) < 10 and s != 'Generell' and s != 'SummaLarm']
    n_alarms_per_type = np.zeros(len(alarm_strings))
    table = []

    if reload:
        for i, recipe_number in enumerate(np.unique(bearing_data[:, 2])):
            bearing_data_ = bearing_data[bearing_data[:, 2] == recipe_number]
            if len(bearing_data_) > 1000:
                changes = bearing_data_[:-1, 1] - bearing_data_[1:, 1]
                changes = np.where(changes != 1)[0]
                starts = bearing_data_[changes, 0][1:]
                ends = bearing_data_[changes + 1, 0][:-1]
                blocks = np.concatenate((np.reshape(starts, (-1, 1)), np.reshape(ends, (-1, 1))), axis=1)
                alarms_in_blocks = []
                if len(blocks) > 0:
                    for block in blocks:
                        over = alarms[alarms[:, 3] > block[0]]
                        over_under = over[over[:, 3] < block[1]]
                        if len(over_under) > 0:
                            alarms_in_blocks.append(over_under[:, 1])
                    a = [item for sublist in alarms_in_blocks for item in sublist]
                    a = [s.replace('Assembly.{}-PLC.Alarms.'.format(station), '') for s in a]
                    a = [s.replace('Assembly.{}-PLC.LarmFelvändRulle'.format(station), 'FV_Rulle') for s in a]
                    a = [s for s in a if
                         1 < len(s) < 10 and s != 'Generell' and s != 'SummaLarm']
                    row = list(bearing_data_[0, [2, 4, 5, 6, 7, 8]])
                    for j, alarm_string in enumerate(alarm_strings):
                        row.append(np.sum(np.array(a) == alarm_string) / len(bearing_data_))
                        n_alarms_per_type[j] += np.sum(np.array(a) == alarm_string)
                    table.append(row)

        np.save('data/{}_correlation_base'.format(station), table)
        np.save('data/{}_n_alarms_per_type'.format(station), n_alarms_per_type)
    table = np.array(np.load('data/{}_correlation_base.npy'.format(station)))
    n_alarms_per_type = np.array(np.load('data/{}_n_alarms_per_type.npy'.format(station)))
    n_cols = np.size(table[0, :])
    R = np.zeros((n_cols, 6))
    for x in range(6):
        for y in range(x, n_cols):
            c = np.corrcoef(table[:, x], table[:, y])
            R[y, x] = c[0][1]
    df = pd.DataFrame(R, index=['recipe_number', 'Rollers', 'Diameter', 'Thickness', 'Width', 'Mass', *alarm_strings],
                      columns=['recipe_number', 'Rollers', 'Diameter', 'Thickness', 'Width', 'Mass'])
    df = df[6:].drop('recipe_number', axis=1)
    df = df.transpose()

    fig, ax = plt.subplots(figsize=(18, 5))
    d = ax.matshow(np.abs(df), cmap='coolwarm', vmin=-0.6, vmax=0.6, aspect='auto')
    ax.set_xticks(np.arange(len(df.columns.values)))
    ax.set_yticks(np.arange(len(df.index.values)))
    ax.set_xticklabels(df.columns.values)
    ax.set_yticklabels(df.index.values)
    plt.xticks(rotation=90)

    for (i, j), z in np.ndenumerate(df):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center', rotation=90)
    for i in range(len(n_alarms_per_type)):
        ax.text(i, 5, int(n_alarms_per_type[i]), ha='center', va='center', rotation=90)
        circle = plt.Circle((i, 5), n_alarms_per_type[i]/20000, color='r', clip_on=False)
        ax.add_artist(circle)
    plt.text(0, -1.2, '{}, Korrelation mellan dimensionsparameter och larmantal'.format(station), fontsize=14)
    plt.text(0, -1, 'Data fr.o.m. {}'.format(six_months_ago.date()), fontsize=8)
    plt.text(-1, 5.05, 'Number of events', ha='right')
    plt.tight_layout()
    plt.savefig('assets/corr_matrix_{}.png'.format(station))
    plt.clf()
