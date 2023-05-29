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
from ..utils_x.utils_winters import *


def add_winters(request, dataset):
    dataset_dates = "{0}_dataset_dates".format(dataset.lower())
    dataset_name = "{0}_dataset_data".format(dataset.lower())
    dataset_data = pd.DataFrame()

    dataset_data = pd.DataFrame({
        'Date' : request.session[dataset_dates],
        'Volume' : request.session[dataset_name]},)
    dataset_data['Volume'] = pd.to_numeric(request.session[dataset_name])
    dataset_data['Date'] = pd.to_datetime(request.session[dataset_dates])

    if request.method == "POST":
        form = HoltWinters_add_form(request.POST)

        if(form.is_valid()):

            model = form.save(False)

            # my_order = (int(request.POST["p_param"]),int(request.POST["d_param"]),int(request.POST["q_param"]))
            # my_seasonal_order = (int(request.POST["sp_param"]), int(request.POST["sd_param"]), int(request.POST["sq_param"]), int(request.POST["m_param"]))
            is_boxcox = request.POST.get('is_boxcox', False)
            lmbda = 0 if (request.POST["lmbda"] == "" or request.POST["lmbda"] == None) else float(request.POST["lmbda"])
            trend = "mul" if (request.POST["trend"] == "mul") else "add"
            seasonal = "mul" if (request.POST["seasonal"] == "mul") else "add"
            damped = True if ("damped" in request.POST) else False

            test_set_index = request.session['{0}_test_set_index'.format(dataset.lower())]
            result_model = model_winters(dataset_data, dataset, test_set_index, trend, seasonal, damped, is_boxcox, lmbda)

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

            model_name = "Holt-Winters {0}{1}{2} {3}".format(
                trend[0].upper(),
                seasonal[0].upper(),
                "D" if damped else "N",
                "BC" if is_boxcox else ""
            )

            # save into session
            model_details = {
                'id' : get_timestamp(),
                'model_name' : model_name,
                'model_type' : 'winters',
                'is_boxcox' : is_boxcox,
                'lmbda' : lmbda,
                'dataset' : dataset,
                'mse' : result_model["mse"],
                'rmse' : result_model["rmse"],
                'trend' : "Additive" if trend=="add" else "Multiplicative",
                'seasonal' : "Additive" if seasonal=="add" else "Multiplicative",
                'damped' : str(damped),
                'mape' : result_model["mape"],
                'mad' : result_model["mad"],
                'lmbda' : result_model["lmbda"],
                'forecasts' : model_predictions.tolist(),
                'forecasts_table' : forecasts_table,
                'display_start' : display_start,
                'display_end' : display_end,
            }
            request.session['saved_winters'].append(model_details)
            request.session.modified = True

    return redirect('graphs_page', dataset)

# def delete_winters(request, dataset, id):
#     model = HoltWintersModel.objects.get(id=id)
#     model.delete()

#     return redirect('graphs_page', dataset)

# def delete_winters(request, dataset, id):
#     for model in request.session['saved_winters']:
#         if str(model['id']) == str(id):
#             request.session['saved_winters'].remove(model)
#             request.session.modified = True

#     return redirect('graphs_page', dataset)