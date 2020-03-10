import numpy as np
import pyodbc
import os
import datetime

def get_minmax(param_limits):
    return param_limits[0], param_limits[0] + param_limits[1]


def process(d, limits):
    d -= limits[0]
    d /= limits[1]
    d -= 0.5
    d *= 2.0
    return d

def deprocess(d, limits):
    d /= 2.0
    d += 0.5
    d *= limits[1]
    d += limits[0]
    return d

def predict_last_week(data, model, helpers):
    n_rows = np.size(data[:, 0])
    rollers = np.transpose([process(data[:, 1].astype('float64'), helpers[0])])
    diameter = np.transpose([process(data[:, 2].astype('float64'), helpers[1])])
    thickness = np.transpose([process(data[:, 3].astype('float64'), helpers[2])])
    width = np.transpose([process(data[:, 4].astype('float64'), helpers[3])])
    mass = np.transpose([process(data[:, 8].astype('float64'), helpers[4])])

    station = np.zeros((n_rows, np.size(helpers[6])))
    design = np.zeros((n_rows, np.size(helpers[7])))
    clearance = np.zeros((n_rows, np.size(helpers[8])))

    for i in range(len(data)):
        design[i, np.where(helpers[7][0][0] == data[i,5])] = 1
        clearance[i, np.where(helpers[8][0][0] == data[i,6])] = 1
        station[i, np.where(helpers[6][0][0] == data[i,7])] = 1
    
    source = np.concatenate((rollers, 
                            diameter, 
                            thickness,
                            width,
                            mass,
                            np.zeros((n_rows, 1)), 
                            np.ones((n_rows, 1)), 
                            station, 
                            design, 
                            clearance), axis=1)
    output = model.eval(source)
    predictions = deprocess(output, helpers[5])
    return data[:, 9], 3600/predictions, data[:, 0]

def exec_sql(env, command):
    conn = pyodbc.connect(
            'DRIVER={};SERVER={};UID={};PWD={}'.format(
                'FreeTDS', 
                env['SERVER'], 
                env['DB_USER'], 
                env['DB_PASSWORD']))
    cursor = conn.cursor()
    sql_result = []
    cursor.execute(command)
    for row in cursor.fetchall():
        sql_result.append(row)
    return np.array(sql_result)

def calculate_mean(data):
    data = data[data[:, 0] > 30]
    data = data[data[:, 0] < 3600]
    data = data[data[:, 1] > 1]
    data = data[data[:, 1] < 20400101]
    ct = data[:, 0]
    dates = data[:, 1]
    dates = [d//100 for d in dates]
    unique_dates = np.unique(dates)

    ct_mean = []
    dates_vec = []

    for date in unique_dates:
        inds = np.where(dates == date)
        if len(inds[0]) > 50:
            ct_mean.append(3600/np.mean(ct[inds]))
            dates_vec.append(datetime.datetime.strptime(str(int(date)), '%Y%m'))

    return dates_vec, ct_mean

