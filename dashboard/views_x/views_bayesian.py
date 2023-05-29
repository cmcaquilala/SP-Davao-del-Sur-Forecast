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
from ..utils_x.utils_bayesian import *

def add_bayesian(request, dataset):
    dataset_dates = "{0}_dataset_dates".format(dataset.lower())
    dataset_name = "{0}_dataset_data".format(dataset.lower())
    dataset_data = pd.DataFrame()

    dataset_data = pd.DataFrame({
        'Date' : request.session[dataset_dates],
        'Volume' : request.session[dataset_name]},)
    dataset_data['Volume'] = pd.to_numeric(request.session[dataset_name])
    dataset_data['Date'] = pd.to_datetime(request.session[dataset_dates])

    if request.method == "POST":
        form = SARIMA_add_form(request.POST)

        if(form.is_valid()):

            model = form.save(False)

            my_order = (int(request.POST["p_param"]),int(request.POST["d_param"]),int(request.POST["q_param"]))
            my_seasonal_order = (int(request.POST["sp_param"]), int(request.POST["sd_param"]), int(request.POST["sq_param"]), int(request.POST["m_param"]))
            is_boxcox = request.POST.get('is_boxcox', False)
            lmbda = 0 if (request.POST["lmbda"] == "" or request.POST["lmbda"] == None) else float(request.POST["lmbda"])

            test_set_index = request.session['{0}_test_set_index'.format(dataset.lower())]
            result_model = model_bayesian(dataset_data, dataset, test_set_index, my_order, my_seasonal_order, is_boxcox, lmbda)

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
            # model_predictions = result_model["predictions"] + result_model["forecasts"]
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

            # save into session
            model_details = {
                'id' : get_timestamp(),
                'model_name' : result_model['model_name'],
                'model_type' : 'bayesian',
                'is_boxcox' : is_boxcox,
                'lmbda' : lmbda,
                'dataset' : dataset,
                'p_param' : request.POST['p_param'],
                'd_param' : request.POST['d_param'],
                'q_param' : request.POST['q_param'],
                'sp_param' : request.POST['sp_param'],
                'sd_param' : request.POST['sd_param'],
                'sq_param' : request.POST['sq_param'],
                'm_param' : request.POST['m_param'],
                # 'bic' : result_model["bic"],
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
            request.session['saved_bayesian'].append(model_details)
            request.session.modified = True

    return redirect('graphs_page', dataset)

# def delete_bayesian(request, dataset, id):
#     model = BayesianSARIMAModel.objects.get(id=id)
#     model.delete()

#     return redirect('graphs_page', dataset)