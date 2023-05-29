import datetime as DateTime
from dateutil.relativedelta import relativedelta
import csv
import json

from django.shortcuts import render, redirect
from django.templatetags.static import static

# stat-related
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..forms import *
from ..models import *
from ..utils_x.utils_lstm import *


def add_lstm(request, dataset):
    dataset_dates = "{0}_dataset_dates".format(dataset.lower())
    dataset_name = "{0}_dataset_data".format(dataset.lower())
    dataset_data = pd.DataFrame()

    dataset_data = pd.DataFrame({
        'Date' : request.session[dataset_dates],
        'Volume' : request.session[dataset_name]},)
    dataset_data['Volume'] = pd.to_numeric(request.session[dataset_name])
    dataset_data['Date'] = pd.to_datetime(request.session[dataset_dates])

    if request.method == "POST":
        form = LSTM_add_form(request.POST)

        if(form.is_valid()):

            model = form.save(False)

            # my_order = (int(request.POST["p_param"]),int(request.POST["d_param"]),int(request.POST["q_param"]))
            # my_seasonal_order = (int(request.POST["sp_param"]), int(request.POST["sd_param"]), int(request.POST["sq_param"]), int(request.POST["m_param"]))
            is_boxcox = request.POST.get('is_boxcox', False)
            lmbda = 0 if (request.POST["lmbda"] == "" or request.POST["lmbda"] == None) else float(request.POST["lmbda"])
            n_inputs = int(request.POST['n_inputs'])
            n_epochs = int(request.POST['n_epochs'])
            n_units = int(request.POST['n_units'])

            test_set_index = request.session['{0}_test_set_index'.format(dataset.lower())]
            result_model = model_lstm(dataset_data, dataset, test_set_index, n_inputs, n_epochs, n_units, is_boxcox, lmbda)

            model.dataset = dataset
            # model.graph = result_model["graph"]
            # model.graph = result_model["filename"]
            # model.bic = result_model["bic"]
            model.mse = result_model["mse"]
            model.rmse = result_model["rmse"]
            model.mape = result_model["mape"]
            model.mad = result_model["mad"]
            model.lmbda = result_model["lmbda"]

            forecasts_table = []
            # model_predictions = lstm_model["predictions"] + lstm_model["forecasts"]
            model_predictions = np.concatenate([result_model["predictions"], result_model["forecasts"]])

            curr_date = pd.to_datetime(result_model["test_set"]['Date'].values[0])
            for i in range(len(model_predictions)):
                year = curr_date.year
                quarter = curr_date.month // 3 + 1

                actual = result_model["test_set"]['Volume'].values[i] if i < len(result_model["test_set"]['Volume'].values) else 0
                error = actual - model_predictions[i] if actual != 0 else 0

                value_dict = {
                    'period' : "{0} Q{1}".format(year, quarter),
                    'actual' : actual,
                    'prediction' : model_predictions[i],
                    'error' : error,
                }
                forecasts_table.append(value_dict)
                curr_date += relativedelta(months=3)
            
            model.forecasts = forecasts_table
            # model.save()

            display_start = 1987
            display_end = 2025

            model_name = "LSTM U({0}) E({1}) W({2}) {3}".format(
                n_units,
                n_epochs,
                n_inputs,
                "BC" if is_boxcox else ""
            )

            # save into session
            model_details = {
                'id' : get_timestamp(),
                'model_name' : model_name,
                'model_type' : 'lstm',
                'n_inputs' : n_inputs,
                'n_epochs' : n_epochs,
                'n_units' : n_units,
                'is_boxcox' : is_boxcox,
                'lmbda' : lmbda,
                'dataset' : dataset,
                'mse' : result_model["mse"],
                'rmse' : result_model["rmse"],
                'mape' : result_model["mape"],
                'mad' : result_model["mad"],
                'lmbda' : result_model["lmbda"],
                'forecasts' : model_predictions.tolist(),
                'forecasts_table' : forecasts_table,
                'display_start' : display_start,
                'display_end' : display_end,
            }
            request.session['saved_lstm'].append(model_details)
            request.session.modified = True

            # model.save()

    return redirect('graphs_page', dataset)

# def delete_lstm(request, dataset, id):
#     model = LSTMModel.objects.get(id=id)
#     model.delete()

#     return redirect('graphs_page', dataset)