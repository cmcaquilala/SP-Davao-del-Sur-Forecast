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
        form = BayesianSARIMA_add_form(request.POST)

        if(form.is_valid()):

            model = form.save(False)

            my_order = (int(request.POST["p_param"]),int(request.POST["d_param"]),int(request.POST["q_param"]))
            my_seasonal_order = (int(request.POST["sp_param"]), int(request.POST["sd_param"]), int(request.POST["sq_param"]), int(request.POST["m_param"]))
            is_boxcox = request.POST.get('is_boxcox', False)
            lmbda = 0 if (request.POST["lmbda"] == "" or request.POST["lmbda"] == None) else float(request.POST["lmbda"])

            bayesian_model = model_bayesian(dataset_data, dataset, my_order, my_seasonal_order, is_boxcox, lmbda)

            model.dataset = dataset
            # model.graph = bayesian_model["graph"]
            # model.graph = bayesian_model["filename"]
            # model.bic = bayesian_model["bic"]
            model.mse = bayesian_model["mse"]
            model.rmse = bayesian_model["rmse"]
            model.mape = bayesian_model["mape"]
            model.mad = bayesian_model["mad"]
            model.lmbda = bayesian_model["lmbda"]

            model_forecasts = []
            model_predictions = bayesian_model["predictions"] + bayesian_model["forecasts"]

            curr_date = pd.to_datetime(bayesian_model["test_set"]['Date'].values[0])
            for i in range(len(model_predictions)):
                year = curr_date.year
                quarter = curr_date.month // 3 + 1

                actual = bayesian_model["test_set"]['Volume'].values[i] if i < len(bayesian_model["test_set"]['Volume'].values) else 0
                error = actual - model_predictions[i] if actual != 0 else 0

                value_dict = {
                    'period' : "{0} Q{1}".format(year, quarter),
                    'actual' : actual,
                    'prediction' : model_predictions[i],
                    'error' : error,
                }
                model_forecasts.append(value_dict)
                curr_date += relativedelta(months=3)
            
            model.forecasts = model_forecasts
            model.save()

    return redirect('graphs_page', dataset)

def delete_bayesian(request, dataset, id):
    model = BayesianSARIMAModel.objects.get(id=id)
    model.delete()

    return redirect('graphs_page', dataset)