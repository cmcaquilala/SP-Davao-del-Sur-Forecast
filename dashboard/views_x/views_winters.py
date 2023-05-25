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
        form = HoltWinters_add_form(request.POST)

        if(form.is_valid()):

            model = form.save(False)

            # my_order = (int(request.POST["p_param"]),int(request.POST["d_param"]),int(request.POST["q_param"]))
            # my_seasonal_order = (int(request.POST["sp_param"]), int(request.POST["sd_param"]), int(request.POST["sq_param"]), int(request.POST["m_param"]))
            is_boxcox = request.POST.get('is_boxcox', False)
            lmbda = 0 if (request.POST["lmbda"] == "" or request.POST["lmbda"] == None) else float(request.POST["lmbda"])

            winters_model = model_winters(filename, dataset_data, dataset, is_boxcox, lmbda)

            model.dataset = dataset
            # model.graph = winters_model["graph"]
            model.graph = winters_model["filename"]
            # model.bic = winters_model["bic"]
            model.mse = winters_model["mse"]
            model.rmse = winters_model["rmse"]
            model.mape = winters_model["mape"]
            model.mad = 0
            model.lmbda = winters_model["lmbda"]

            model_forecasts = []
            # model_predictions = winters_model["predictions"] + winters_model["forecasts"]
            model_predictions = np.concatenate([winters_model["predictions"], winters_model["forecasts"]])

            curr_date = pd.to_datetime(winters_model["test_set"]['Date'].values[0])
            for i in range(len(model_predictions)):
                year = curr_date.year
                quarter = curr_date.month // 3 + 1

                actual = winters_model["test_set"]['Volume'].values[i] if i < len(winters_model["test_set"]['Volume'].values) else 0
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

def delete_winters(request, dataset, id):
    model = HoltWintersModel.objects.get(id=id)
    model.delete()

    return redirect('graphs_page', dataset)

