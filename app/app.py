#!/usr/bin/env python3

from __future__ import print_function
from importlib import import_module
import torch
import pyodbc
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import os
import base64
import neural_network
from app_functions import *
from dash.dependencies import Input, Output

with open('env') as env_data:
    for line in env_data:
        s = str(line).split('=')
        os.environ[s[0]] = s[1][:-1]

def main():
    helpers = np.load('app/data/helpers.npy', allow_pickle=True)
    n_station = np.size(helpers[6])
    n_design = np.size(helpers[7])
    n_clearance = np.size(helpers[8])
    size = 5+2+n_station+n_design+n_clearance
    model = neural_network.Model(size)
    min_rollers, max_rollers  = get_minmax(helpers[0])
    min_diameter, max_diameter = get_minmax(helpers[1])
    min_thickness, max_thickness = get_minmax(helpers[2])
    min_width, max_width = get_minmax(helpers[3])
    min_mass, max_mass = get_minmax(helpers[4])
    station_options = [{'label': l, 'value': i} for i, l in enumerate(helpers[6][0][0])]
    design_options = [{'label': l, 'value': i} for i, l in enumerate(helpers[7][0][0])]
    clearance_options = [{'label': l, 'value': i} for i, l in enumerate(helpers[8][0][0])]
    types = exec_sql(os.environ, 'SELECT * FROM KEPServer.dbo.ML_Base_Table')
    bearing_options = [{'label': row[0], 'value': i} for i, row in enumerate(types)]


    prod_res_last_week_am2 = exec_sql(os.environ, 'EXEC KEPServer.dbo.ML_Base_Last_Week AM2, 30')
    last_week_outputs_am2, last_week_predictions_am2, last_week_designations_am2, last_week_amount_am2 = predict_last_week(prod_res_last_week_am2, model, helpers)
    last_week_designations_am2 = ['{}: {}'.format(d, int(round(a))) for d, a in zip(last_week_designations_am2, last_week_amount_am2.astype('float'))]
    production_time_output_am2 = last_week_amount_am2.astype('float')/last_week_outputs_am2.astype('float')
    production_time_predictions_am2 = last_week_amount_am2.astype('float')/last_week_predictions_am2.astype('float')
    diff_am2 = round(np.sum(production_time_output_am2)-np.sum(production_time_predictions_am2), 2)
    abstract_am2 = 'The Production Time was {} hours faster than expected.'.format(-diff_am2) if diff_am2<0 else 'The Production Time was {} hours slower than expected.'.format(diff_am2)

    app = dash.Dash(
            __name__,
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
            )
    # server = app.server
    app.config.suppress_callback_exceptions = False

    app.layout = html.Div(className='outer', children=[
        html.H1('Output prediction AM1/2'),
        html.Div(className='upper', children=[
            html.Div(className='bundlerleft', children=[
                html.Div(className='slider-row', children=[
                    html.Div(className='slider', children=[
                        html.Div(className='result', id='rollers_result'),
                        dcc.Slider(id='rollers_slider', min=min_rollers, max=max_rollers, 
                            updatemode='drag', value=min_rollers.astype('float64'))
                        ]),
                    html.Div(className='slider', children=[
                        html.Div(className='result', id='diameter_result'),
                        dcc.Slider(id='diameter_slider', min=min_diameter, max=max_diameter, 
                            updatemode='drag', value=min_diameter)
                        ]),
                    html.Div(className='slider', children=[
                        html.Div(className='result', id='thickness_result'),
                        dcc.Slider(id='thickness_slider', min=min_thickness, max=max_thickness, 
                            updatemode='drag', value=min_thickness)
                        ]),
                    html.Div(className='slider', children=[
                        html.Div(className='result', id='width_result'),
                        dcc.Slider(id='width_slider', min=min_width, max=max_width, 
                            updatemode='drag', value=min_width)
                        ]),
                    html.Div(className='slider', children=[
                        html.Div(className='result', id='mass_result'),
                        dcc.Slider(id='mass_slider', min=min_mass, max=max_mass, 
                            updatemode='drag', value=min_mass)
                        ]),
                    ]),
                html.Div(className='dropdown-row', children=[
                    html.Div(className='dropdown', children=[
                        html.Div(['Station']),
                        dcc.Dropdown(id='station', options=station_options, value=0),
                        ]),
                    html.Div(className='dropdown', children=[
                        html.Div(['Design']),
                        dcc.Dropdown(id='design', options=design_options, value=0),
                        ]),
                    html.Div(className='dropdown', children=[
                        html.Div(['Clearance']),
                        dcc.Dropdown(id='clearance', options=clearance_options, value=0),
                        ]),
                    ]),
                ]),
            html.Div(className='bundlerright', children=[
                html.Div(['Predicted Output: ']),
                html.Div(className='nn_result', id='nn_result'),
                html.Div(['bearings / hour']),
                ]),
            ]),
        html.Div(className='stats', children=[
            html.Div(className='dropdown_bearings', children=[
                html.Div(['Existing bearings']),
                dcc.Dropdown(id='bearings', options=bearing_options, value=0),
                ]),
            html.Div(className='stats_div', children=[
                dcc.Graph(id='stats_over_time'),
                ])
            ]),
        html.Div(className='graphs', children=[
            html.Div(className='graph', children=[
                html.Div([
                    html.Div(className='AM_last_days_title', children=['AM1, Last: ']),
                    html.Div(className='AM_last_days_title', children=[
                        dcc.Dropdown(id='AM1_dropdown', options=[{'label': '1 day', 'value': 1}, 
                            {'label': '3 days', 'value': 3},
                            {'label': '7 days', 'value': 7},
                            {'label': '30 days', 'value': 30}], value=7),
                        ]),
                    html.Div(className='AM_last_days_title', children=['large orders ']),
                    ]),
                dcc.Graph('AM1_last_days_fig')
                ]),
            html.Div(className='graph', children=[
                html.Div([
                    html.Div(className='AM_last_days_title', children=['AM2, Last: ']),
                    html.Div(className='AM_last_days_title', children=[
                        dcc.Dropdown(id='AM2_dropdown', options=[{'label': '1 day', 'value': 1}, 
                            {'label': '3 days', 'value': 3},
                            {'label': '7 days', 'value': 7},
                            {'label': '30 days', 'value': 30}], value=7),
                        ]),
                    html.Div(className='AM_last_days_title', children=['large orders ']),
                    ]),
                dcc.Graph('AM2_last_days_fig')
                ]),
            html.Div(className='graph', children=[
                html.Img(src=app.get_asset_url('corr_matrix_AM1.png'), style={'width': '90%'}),
                ]),
            html.Div(className='graph', children=[
                html.Img(src=app.get_asset_url('corr_matrix_AM2.png'), style={'width': '90%'}),
                ]),
            html.Div(className='graph', children=[
                html.Div([
                    html.Div(id='none', className='AM_last_days_title', children=['Future']),
                    ]),
                dcc.Graph('future_fig')
                ]),
            ]),
        ])

    @app.callback(Output('AM1_last_days_fig', 'figure'),
            [Input('AM1_dropdown', 'value')])
    def display_value(value):
        prod_res_last_week_am1 = exec_sql(os.environ, 'EXEC KEPServer.dbo.ML_Base_Last_Week AM1, {}'.format(value))
        last_week_outputs_am1, last_week_predictions_am1, last_week_designations_am1, last_week_amount_am1 = predict_last_week(prod_res_last_week_am1, model, helpers)
        last_week_designations_am1 = ['{}: {}'.format(d, int(round(a))) for d, a in zip(last_week_designations_am1, last_week_amount_am1.astype('float'))]
        production_time_output_am1 = last_week_amount_am1.astype('float')/last_week_outputs_am1.astype('float')
        production_time_predictions_am1 = last_week_amount_am1.astype('float')/last_week_predictions_am1.astype('float')
        diff_am1 = round(np.sum(production_time_output_am1)-np.sum(production_time_predictions_am1), 2)
        abstract_am1 = 'The Production Time was {} hours faster than expected.'.format(-diff_am1) if diff_am1<0 else 'The Production Time was {} hours slower than expected.'.format(diff_am1)

        return {'data': [{
                        'x': last_week_designations_am1,
                        'y': last_week_predictions_am1,
                        'type': 'bar', 
                        'name': 'Predictions'},
                        {  
                        'x': last_week_designations_am1, 
                        'y': last_week_outputs_am1, 
                        'type': 'bar', 
                        'name': 'Outputs'},],
                'layout': {
                    'title': {'text': '{}'.format(abstract_am1), 'y': 0.9},
                    'height': 500,
                    'margin': {'l': 120, 'b': 200, 't': 65, 'r': 120},
                    'legend': {'x': 0, 'y': 1.1},
                    'xaxis': {'tickangle': -45},}
                }
    
    @app.callback(Output('AM2_last_days_fig', 'figure'),
            [Input('AM2_dropdown', 'value')])
    def display_value(value):
        prod_res_last_week_am1 = exec_sql(os.environ, 'EXEC KEPServer.dbo.ML_Base_Last_Week AM2, {}'.format(value))
        last_week_outputs_am1, last_week_predictions_am1, last_week_designations_am1, last_week_amount_am1 = predict_last_week(prod_res_last_week_am1, model, helpers)
        last_week_designations_am1 = ['{}: {}'.format(d, int(round(a))) for d, a in zip(last_week_designations_am1, last_week_amount_am1.astype('float'))]
        production_time_output_am1 = last_week_amount_am1.astype('float')/last_week_outputs_am1.astype('float')
        production_time_predictions_am1 = last_week_amount_am1.astype('float')/last_week_predictions_am1.astype('float')
        diff_am1 = round(np.sum(production_time_output_am1)-np.sum(production_time_predictions_am1), 2)
        abstract_am1 = 'The Production Time was {} hours faster than expected.'.format(-diff_am1) if diff_am1<0 else 'The Production Time was {} hours slower than expected.'.format(diff_am1)

        return {'data': [{
                        'x': last_week_designations_am1,
                        'y': last_week_predictions_am1,
                        'type': 'bar', 
                        'name': 'Predictions'},
                        {  
                        'x': last_week_designations_am1, 
                        'y': last_week_outputs_am1, 
                        'type': 'bar', 
                        'name': 'Outputs'},],
                'layout': {
                    'title': {'text': '{}'.format(abstract_am1), 'y': 0.9},
                    'height': 500,
                    'margin': {'l': 120, 'b': 200, 't': 65, 'r': 120},
                    'legend': {'x': 0, 'y': 1.1},
                    'xaxis': {'tickangle': -45},}
                }
    
    
    @app.callback(Output('future_fig', 'figure'),
            [Input('none', 'children')])
    def display_value(none):
        future_prod = exec_sql(os.environ, 'SET ANSI_NULLS, ANSI_WARNINGS ON; EXEC KEPServer.dbo.ML_Base_Future_Orders')
        future_predictions, future_designations, future_amount = predict_future(future_prod, model, helpers)
        future_designations = ['{}: {}'.format(d, int(round(a))) for d, a in zip(future_designations, future_amount.astype('float'))]
        production_time_future = future_amount.astype('float')/future_predictions.astype('float')
        diff = round(np.sum(production_time_future), 2)
        abstract = 'The Effective Production Time is projected to be {} hours.'.format(diff)

        return {'data': [{
                        'x': future_designations,
                        'y': future_predictions,
                        'type': 'bar', 
                        'name': 'Predictions'},
                        ],
                'layout': {
                    'title': {'text': '{}'.format(abstract), 'y': 0.9},
                    'height': 500,
                    'margin': {'l': 120, 'b': 200, 't': 65, 'r': 120},
                    'legend': {'x': 0, 'y': 1.1},
                    'xaxis': {'tickangle': -45},}
                }
    
    
    @app.callback(Output('rollers_result', 'children'),
            [Input('rollers_slider', 'value')])
    def display_value(value):
        return 'Rollers: {}'.format(value)

    @app.callback(Output('diameter_result', 'children'),
            [Input('diameter_slider', 'value')])
    def display_value(value):
        return 'Diameter: {}'.format(value)

    @app.callback(Output('thickness_result', 'children'),
            [Input('thickness_slider', 'value')])
    def display_value(value):
        return 'Thickness: {}'.format(value)

    @app.callback(Output('width_result', 'children'),
            [Input('width_slider', 'value')])
    def display_value(value):
        return 'Width: {}'.format(value)

    @app.callback(Output('mass_result', 'children'),
            [Input('mass_slider', 'value')])
    def display_value(value):
        return 'Mass: {}'.format(round(value, 2))

    @app.callback(Output('nn_result', 'children'),
            [Input('rollers_slider', 'value'),
                Input('diameter_slider', 'value'),
                Input('thickness_slider', 'value'),
                Input('width_slider', 'value'),
                Input('mass_slider', 'value'),
                Input('station', 'value'),
                Input('design', 'value'),
                Input('clearance', 'value')])
    def update_nn(rollers, diameter, thickness, width, mass, station_index, design_index, clearance_index):
        rollers = process(rollers, helpers[0])
        diameter = process(diameter, helpers[1])
        thickness = process(thickness, helpers[2])
        width = process(width, helpers[3])
        mass = process(mass, helpers[4])
        station = np.zeros(np.size(helpers[6]))
        station[station_index] = 1
        design = np.zeros(np.size(helpers[7]))
        design[design_index] = 1
        clearance = np.zeros(np.size(helpers[8]))
        clearance[clearance_index] = 1

        source = np.concatenate(([rollers], 
            [diameter],
            [thickness],
            [width],
            [mass],
            [0],
            [1],
            station,
            design,
            clearance), axis=0)
        output = deprocess(model.eval(source), helpers[5])
        return round(3600/output, 2)

    @app.callback([Output('rollers_slider', 'value'),
        Output('diameter_slider', 'value'),
        Output('thickness_slider', 'value'),
        Output('width_slider', 'value'),
        Output('mass_slider', 'value'),
        Output('station', 'value'),
        Output('design', 'value'),
        Output('clearance', 'value'),
        Output('stats_over_time', 'figure')],
        [Input('bearings', 'value')])
    def select_bearing(value):
        design = np.where(helpers[7][0][0] == types[value][5])[0][0]
        clearance = np.where(helpers[8][0][0] == types[value][6])[0][0]
        station = np.where(helpers[6][0][0] == types[value][7])[0][0]

        designation = types[value][0][:-6]
        station_s = types[value][0][-4:-1]
        command = 'SELECT Cycletime, Bearing_MountDate FROM KEPServer.dbo.AM_BearingData WHERE ' \
                'Bearing_Designation = \'{}\' AND Station = \'{}\''.format(designation, station_s)
        ct_date = exec_sql(os.environ, command)
        x, y = calculate_mean(ct_date)


        if len(y) > 3:

            fig={ 'data':[{
                'x': x,
                'y': y,
                'type': 'bar',}],
                'layout': {
                    'title': {'text': 'Historic Production Rate for {}'.format(designation), 'y': 0.9},
                    'height': 180,
                    'margin': {'l': 40, 'b': 20, 't': 30, 'r': 20},
                    'yaxis': {'title': 'Bearings/hour'}}} 
        else:
            fig={ 'data':[{
                'x': [1],
                'y': [1],
                'text': 'Not enough data',
                'mode': 'text'}],
                'layout': {
                    'title': {'text': 'Historic Production Rate for {}'.format(designation), 'y': 0.9},
                    'height': 180,
                    'margin': {'l': 40, 'b': 20, 't': 30, 'r': 20},
                    'yaxis': {'title': 'Bearings/hour'}}} 



        return  round(float(types[value][1]), 2), round(float(types[value][2]), 2), round(float(types[value][3]), 2), \
                        round(float(types[value][4]), 2), round(float(types[value][8]), 2), station, design, clearance, fig
    
    
    app.run_server(host='0.0.0.0', port=8080)


if __name__ == '__main__':
    main()
