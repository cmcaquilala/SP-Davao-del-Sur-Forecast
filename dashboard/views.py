from django.shortcuts import render, redirect
from .utils import *
from .models import *
from .forms import *
from django.templatetags.static import static

import csv
import json

# stat-related
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import datetime as DateTime
from dateutil.relativedelta import relativedelta

# Create your views here.
def index_page(request):
    return render(request, 'dashboard/index.html')

# Generalist
def add_sarima(request, dataset):
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
        form = SARIMA_add_form(request.POST)

        if(form.is_valid()):

            model = form.save(False)

            my_order = (int(request.POST["p_param"]),int(request.POST["d_param"]),int(request.POST["q_param"]))
            my_seasonal_order = (int(request.POST["sp_param"]), int(request.POST["sd_param"]), int(request.POST["sq_param"]), int(request.POST["m_param"]))
            is_box_cox = request.POST.get('is_boxcox', False)
            lmbda = 0 if (request.POST["lmbda"] == "" or request.POST["lmbda"] == None) else float(request.POST["lmbda"])

            sarima_model = model_sarima(dataset_data, dataset, my_order, my_seasonal_order, is_box_cox, lmbda)

            model.dataset = dataset
            model.graph = sarima_model["graph"]
            model.mse = sarima_model["mse"]
            model.rmse = sarima_model["rmse"]
            model.mape = sarima_model["mape"]
            model.mad = 0
            model.lmbda = sarima_model["lmbda"]

            forecasts = []
            predictions = np.ndarray.tolist(sarima_model["predictions"].values)
            out_of_sample = np.ndarray.tolist(sarima_model["forecasts"])
            predictions += out_of_sample

            curr_date = pd.to_datetime(sarima_model["test_set"]['Date'].values[0])
            for i in range(len(predictions)):
                curr_date += relativedelta(months=3)
                year = curr_date.year
                quarter = curr_date.month // 3 + 1

                actual = sarima_model["test_set"]['Volume'].values[i] if i < len(sarima_model["test_set"]['Volume'].values) else 0
                error = actual - predictions[i] if actual != 0 else 0

                value_dict = {
                    'period' : "{0} Q{1}".format(year, quarter),
                    'actual' : actual,
                    'prediction' : predictions[i],
                    'error' : error,
                }
                forecasts.append(value_dict)
            
            model.forecasts = forecasts
            model.save()

    return redirect('graphs_page', dataset)

def add_bayesian(request, dataset):
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
        form = Bayesian_ARMA_add_form(request.POST)

        if(form.is_valid()):

            model = form.save(False)

            my_order = (int(request.POST["p_param"]),int(request.POST["q_param"]))
            is_box_cox = False

            bayesian_arma_model = model_bayesian(dataset_data, dataset, my_order, is_box_cox)

            model.dataset = dataset
            model.graph = bayesian_arma_model["graph"]
            model.mse = bayesian_arma_model["mse"]
            model.rmse = bayesian_arma_model["rmse"]
            model.mape = bayesian_arma_model["mape"]
            model.mad = 0

            model.save()
    
    return redirect('graphs_page', dataset)

def delete_sarima(request, dataset, id):
    model = SARIMAModel.objects.get(id=id)
    model.delete()

    return redirect('graphs_page', dataset)

def delete_bayesian(request, dataset, id):
    model = BayesianARMAModel.objects.get(id=id)
    model.delete()

    return redirect('graphs_page', dataset)

def graphs_page(request, dataset):
    # Load dataset
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

    # Loads forms for modal
    sarima_form = SARIMA_add_form(request.POST)
    bayesian_form = Bayesian_ARMA_add_form(request.POST)

    # SARIMA Part
    sarima_models = []

    k = SARIMAModel.objects.filter(dataset=dataset)

    for x in SARIMAModel.objects.filter(dataset=dataset):
        sarima_models.append(x)

    # Bayesian ARMA Part
    bayesian_arma_models = []

    for x in BayesianARMAModel.objects.filter(dataset=dataset):
        bayesian_arma_models.append(x)

    # temporary date
    test_set = dataset_data[132:]

    merged_graphs = get_merged_graphs(sarima_models, bayesian_arma_models, test_set)

    context = {
        'dataset' : dataset,
        'merged_graphs' : merged_graphs,
        'sarima_models' : sarima_models,
        'bayesian_arma_models' : bayesian_arma_models,
        'sarima_form' : sarima_form,
        'bayesian_form' : bayesian_form,
    }

    return render(request, 'dashboard/graph_page.html', context)