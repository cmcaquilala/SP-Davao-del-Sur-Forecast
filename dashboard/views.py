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

# Create your views here.
def index_page(request):
    return render(request, 'dashboard/index.html')

# Generalist
def add_sarima(request, dataset):
    if dataset == "Rice":
        return rice_page_add_sarima(request)
    else:
        return corn_page_add_sarima(request)

def add_bayesian(request, dataset):
    if dataset == "Rice":
        return rice_page_add_bayesian(request)
    else:
        return corn_page_add_bayesian(request)

def delete_sarima(request, dataset, id):
    if dataset == "Rice":
        return rice_page_delete_sarima(request,id)
    else:
        return corn_page_delete_sarima(request,id)

def delete_bayesian(request, dataset, id):
    if dataset == "Rice":
        return rice_page_delete_bayesian(request, id)
    else:
        return corn_page_delete_bayesian(request, id)

# Rice
def rice_page_add_sarima(request):
    dataset_name = "Rice"

    with open('static/rice data.csv') as file:
        reader = csv.reader(file)
        readerlist = []
        next(reader)
        
        for row in reader:
            readerlist.append(row)

    rice_data = pd.DataFrame(readerlist, columns=['Date','Volume'])
    rice_data['Volume'] = pd.to_numeric(rice_data['Volume'])
    rice_data['Date'] = pd.to_datetime(rice_data['Date'])

    if request.method == "POST":
        form = SARIMA_add_form(request.POST)

        if(form.is_valid()):

            model = form.save(False)

            my_order = (int(request.POST["p_param"]),int(request.POST["d_param"]),int(request.POST["q_param"]))
            my_seasonal_order = (int(request.POST["sp_param"]), int(request.POST["sd_param"]), int(request.POST["sq_param"]), int(request.POST["m_param"]))

            sarima_model = model_sarima(rice_data, "Rice", my_order, my_seasonal_order)

            model.dataset = dataset_name
            model.graph = sarima_model["graph"]
            model.mse = sarima_model["mse"]
            model.rmse = sarima_model["rmse"]
            model.mape = sarima_model["mape"]
            model.mad = 0

            model.save()
    
    return redirect('rice_page')

def rice_page_delete_sarima(request, id):
    model = SARIMAModel.objects.get(id=id)
    model.delete()

    return redirect('rice_page')

def rice_page_add_bayesian(request):
    dataset_name = "Rice"

    with open('static/rice data.csv') as file:
        reader = csv.reader(file)
        readerlist = []
        next(reader)
        
        for row in reader:
            readerlist.append(row)

    rice_data = pd.DataFrame(readerlist, columns=['Date','Volume'])
    rice_data['Volume'] = pd.to_numeric(rice_data['Volume'])
    rice_data['Date'] = pd.to_datetime(rice_data['Date'])

    if request.method == "POST":
        form = Bayesian_ARMA_add_form(request.POST)

        if(form.is_valid()):

            model = form.save(False)

            my_order = (int(request.POST["p_param"]),int(request.POST["q_param"]))
            bayesian_arma_model = model_bayesian(rice_data, "Rice", my_order)

            model.dataset = dataset_name
            model.graph = bayesian_arma_model["graph"]
            model.mse = bayesian_arma_model["mse"]
            model.rmse = bayesian_arma_model["rmse"]
            model.mape = bayesian_arma_model["mape"]
            model.mad = 0

            model.save()
    
    return redirect('rice_page')

def rice_page_delete_bayesian(request, id):
    model = BayesianARMAModel.objects.get(id=id)
    model.delete()

    return redirect('rice_page')

def rice_page(request):

    # Start of others
    dataset_name = "Rice"

    with open('static/rice data.csv') as file:
        reader = csv.reader(file)
        readerlist = []
        next(reader)
        
        for row in reader:
            readerlist.append(row)

    rice_data = pd.DataFrame(readerlist, columns=['Date','Volume'])
    rice_data['Volume'] = pd.to_numeric(rice_data['Volume'])
    rice_data['Date'] = pd.to_datetime(rice_data['Date'])

    # Loads forms for modal
    sarima_form = SARIMA_add_form(request.POST)
    bayesian_form = Bayesian_ARMA_add_form(request.POST)

    # SARIMA Part
    sarima_models = []

    k = SARIMAModel.objects.filter(dataset='Rice')

    for x in SARIMAModel.objects.filter(dataset='Rice'):
        sarima_models.append(x)

    # Bayesian ARMA Part
    bayesian_arma_models = []

    for x in BayesianARMAModel.objects.filter(dataset='Rice'):
        bayesian_arma_models.append(x)

    # End of Bayesian ARMA

    context = {
        'dataset' : dataset_name,
        'sarima_models' : sarima_models,
        'bayesian_arma_models' : bayesian_arma_models,
        'sarima_form' : sarima_form,
        'bayesian_form' : bayesian_form,
    }

    return render(request, 'dashboard/graph_page.html', context)

# Corn
def corn_page_add_sarima(request):
    dataset_name = "Corn"

    with open('static/corn data.csv') as file:
        reader = csv.reader(file)
        readerlist = []
        next(reader)
        
        for row in reader:
            readerlist.append(row)

    corn_data = pd.DataFrame(readerlist, columns=['Date','Volume'])
    corn_data['Volume'] = pd.to_numeric(corn_data['Volume'])
    corn_data['Date'] = pd.to_datetime(corn_data['Date'])

    if request.method == "POST":
        form = SARIMA_add_form(request.POST)

        if(form.is_valid()):

            model = form.save(False)

            my_order = (int(request.POST["p_param"]),int(request.POST["d_param"]),int(request.POST["q_param"]))
            my_seasonal_order = (int(request.POST["sp_param"]), int(request.POST["sd_param"]), int(request.POST["sq_param"]), int(request.POST["m_param"]))

            sarima_model = model_sarima(corn_data, "Corn", my_order, my_seasonal_order)

            model.dataset = dataset_name
            model.graph = sarima_model["graph"]
            model.mse = sarima_model["mse"]
            model.rmse = sarima_model["rmse"]
            model.mape = sarima_model["mape"]
            model.mad = 0

            model.save()
    
    return redirect('corn_page')

def corn_page_delete_sarima(request, id):
    model = SARIMAModel.objects.get(id=id)
    model.delete()

    return redirect('corn_page')

def corn_page_add_bayesian(request):
    dataset_name = "Corn"

    with open('static/corn data.csv') as file:
        reader = csv.reader(file)
        readerlist = []
        next(reader)
        
        for row in reader:
            readerlist.append(row)

    corn_data = pd.DataFrame(readerlist, columns=['Date','Volume'])
    corn_data['Volume'] = pd.to_numeric(corn_data['Volume'])
    corn_data['Date'] = pd.to_datetime(corn_data['Date'])

    if request.method == "POST":
        form = Bayesian_ARMA_add_form(request.POST)

        if(form.is_valid()):

            model = form.save(False)

            my_order = (int(request.POST["p_param"]),int(request.POST["q_param"]))
            bayesian_arma_model = model_bayesian(corn_data, "Corn", my_order)

            model.dataset = dataset_name
            model.graph = bayesian_arma_model["graph"]
            model.mse = bayesian_arma_model["mse"]
            model.rmse = bayesian_arma_model["rmse"]
            model.mape = bayesian_arma_model["mape"]
            model.mad = 0

            model.save()
    
    return redirect('corn_page')

def corn_page_delete_bayesian(request, id):
    model = BayesianARMAModel.objects.get(id=id)
    model.delete()

    return redirect('corn_page')

def corn_page(request):

    # Start of others
    dataset_name = "Corn"

    with open('static/corn data.csv') as file:
        reader = csv.reader(file)
        readerlist = []
        next(reader)
        
        for row in reader:
            readerlist.append(row)

    corn_data = pd.DataFrame(readerlist, columns=['Date','Volume'])
    corn_data['Volume'] = pd.to_numeric(corn_data['Volume'])
    corn_data['Date'] = pd.to_datetime(corn_data['Date'])

    # Loads forms for modal
    sarima_form = SARIMA_add_form(request.POST)
    bayesian_form = Bayesian_ARMA_add_form(request.POST)

    # SARIMA Part
    sarima_models = []

    for x in SARIMAModel.objects.filter(dataset='Corn'):
        sarima_models.append(x)

    # Bayesian ARMA Part
    bayesian_arma_models = []

    for x in BayesianARMAModel.objects.filter(dataset='Corn'):
        bayesian_arma_models.append(x)

    # End of Bayesian ARMA

    context = {
        'dataset' : dataset_name,
        'sarima_models' : sarima_models,
        'bayesian_arma_models' : bayesian_arma_models,
        'sarima_form' : sarima_form,
        'bayesian_form' : bayesian_form,
    }

    return render(request, 'dashboard/graph_page.html', context)
