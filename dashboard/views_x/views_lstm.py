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
    filename = "static/{0} data.csv".format(str.lower(dataset))  
    with open(filename) as file:
        reader = csv.reader(file)
        readerlist = []
        next(reader)
        
        for row in reader:
            readerlist.append(row)

    dataset_data = pd.DataFrame(readerlist, columns=['Date','Volume'])
    dataset_data['Volume'] = pd.to_numeric(dataset_data['Volume'])
    dataset_data['Date'] = pd.to_datetime(dataset_data['Date'])

    if request.method == "POST":
        form = LSTM_add_form(request.POST)

        if(form.is_valid()):

            model = form.save(False)

            # my_order = (int(request.POST["p_param"]),int(request.POST["d_param"]),int(request.POST["q_param"]))
            # my_seasonal_order = (int(request.POST["sp_param"]), int(request.POST["sd_param"]), int(request.POST["sq_param"]), int(request.POST["m_param"]))
            is_boxcox = request.POST.get('is_boxcox', False)
            lmbda = 0 if (request.POST["lmbda"] == "" or request.POST["lmbda"] == None) else float(request.POST["lmbda"])

            lstm_model = model_lstm(filename, dataset_data, dataset, is_boxcox, lmbda)

            model.dataset = dataset
            # model.graph = lstm_model["graph"]
            model.graph = lstm_model["filename"]
            # model.bic = lstm_model["bic"]
            model.mse = lstm_model["mse"]
            model.rmse = lstm_model["rmse"]
            model.mape = lstm_model["mape"]
            model.mad = 0
            model.lmbda = lstm_model["lmbda"]

            model_forecasts = []
            # model_predictions = lstm_model["predictions"] + lstm_model["forecasts"]
            model_predictions = np.concatenate([lstm_model["predictions"], lstm_model["forecasts"].tolist()])

            curr_date = pd.to_datetime(lstm_model["test_set"]['Date'].values[0])
            for i in range(len(model_predictions)):
                year = curr_date.year
                quarter = curr_date.month // 3 + 1

                actual = lstm_model["test_set"]['Volume'].values[i] if i < len(lstm_model["test_set"]['Volume'].values) else 0
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

            j = lstm_model["predictions"]
            k = lstm_model["forecasts"]
            l = model_forecasts

            model.save()

    return redirect('graphs_page', dataset)

def delete_lstm(request, dataset, id):
    model = LSTMModel.objects.get(id=id)
    model.delete()

    return redirect('graphs_page', dataset)