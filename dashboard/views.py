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
def indexPage(request):
    return render(request, 'dashboard/index.html')

def ricePage(request):

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

    if request.method == "POST":
        form = SARIMA_add_form(request.POST)

        if(form.is_valid()):

            model = form.save(False)

            my_order = (int(request.POST["p_param"]),int(request.POST["d_param"]),int(request.POST["q_param"]))
            my_seasonal_order = (int(request.POST["sp_param"]), int(request.POST["sd_param"]), int(request.POST["sq_param"]), int(request.POST["m_param"]))

            sarima_model = model_sarima(rice_data, "Rice", my_order, my_seasonal_order)

            model.graph = sarima_model["graph"]
            model.mse = sarima_model["mse"]
            model.rmse = sarima_model["rmse"]
            model.mape = sarima_model["mape"]
            model.mad = 0

            model.save()

        return redirect('ricePage')

    # SARIMA Part
    sarima_models = []

    for x in SARIMAModel.objects.all():

        # sarima_model = model_sarima(rice_data, "Rice", my_order, my_seasonal_order)

        sarima_models.append({
            'model' : x,
        })

    # for x in SARIMAModel.objects.all():
    #     my_order = (x.p_param,x.d_param,x.q_param)
    #     my_seasonal_order = (x.sp_param, x.sd_param, x.sq_param, x.m_param)

    #     sarima_model = model_sarima(rice_data, "Rice", my_order, my_seasonal_order)

    #     x.mse = sarima_model["mse"]
    #     x.rmse = sarima_model["rmse"]
    #     x.mape = sarima_model["mape"]
    #     x.mad = 0

    #     x.graph = sarima_model["graph"]

    #     x.save()

    #     sarima_models.append({
    #         'model' : x,
    #     })

    # Bayesian ARMA Part
    bayesian_arma_models = []

    for x in BayesianARMAModel.objects.all():
        my_order = (x.p_param,x.q_param)

        bayesian_arma_model = model_bayesian(rice_data, "Rice", my_order)

        x.mse = bayesian_arma_model["mse"]
        x.rmse = bayesian_arma_model["rmse"]
        x.mape = bayesian_arma_model["mape"]
        x.mad = 0

        x.save()

        # x.update(mse = sarima_model["mse"])
        # x.update(rmse = sarima_model["rmse"])
        # x.update(mape = sarima_model["mape"])
        # x.update(mad = 0)

        # sarima_summary = {
        #     'graph' : sarima_model["graph"],
        #     'order' : my_order,
        #     'seasonal_order' : my_seasonal_order,
        #     'mse' : '{0:.2f}'.format(sarima_model["mse"]),
        #     'rmse' : '{0:.4f}'.format(sarima_model["rmse"]),
        #     'mape' : '{0:.4f}'.format(sarima_model["mape"]),
        # }

        bayesian_arma_models.append({
            'model' : x,
            'graph' : bayesian_arma_model["graph"]
        })


    # End of Bayesian ARMA

    context = {
        'dataset' : dataset_name,
        'sarima_models' : sarima_models,
        'bayesian_arma_models' : bayesian_arma_models,
        'sarima_form' : sarima_form,
    }

    return render(request, 'dashboard/graph_page.html', context)

def riceAddSARIMA(request):
    form = SARIMA_add_form(request.POST)

    if request.method == "POST":
        form = SARIMA_add_form(request.POST)



        if(form.is_valid()):
            form.save()

        return redirect('ricePage')

    context = {'form' : form}
    return(render(request, context))

def cornPage(request):

    # rice_data = pd.read_csv("rice data.csv")
    # corn_data = pd.read_csv("corn data.csv")

    with open('static/corn data.csv') as file:
        reader = csv.reader(file)
        readerlist = []
        next(reader)
        
        for row in reader:
            readerlist.append(row)

    corn_data = pd.DataFrame(readerlist, columns=['Date','Volume'])
    corn_data['Volume'] = pd.to_numeric(corn_data['Volume'])
    corn_data['Date'] = pd.to_datetime(corn_data['Date'])
    # corn_data['Volume'] = corn_data['Volume'].astype(float)
    # corn_data['Date'] = corn_data['Date'].astype(datetime)

    # json_records = corn_data.reset_index().to_json(orient ='records')
    # arr = []
    # arr = json.loads(json_records)

    my_order = (1,0,0)
    my_seasonal_order = (1, 0, 1, 4)

    sarima_model = model_sarima(corn_data, "Corn", my_order, my_seasonal_order)

    sarima_summary = {
        'graph' : sarima_model["graph"],
        'order' : my_order,
        'seasonal_order' : my_seasonal_order,
        'mse' : '{0:.2f}'.format(sarima_model["mse"]),
        'rmse' : '{0:.4f}'.format(sarima_model["rmse"]),
        'mape' : '{0:.4f}'.format(sarima_model["mape"]),
    }

    context = {
        'sarima' : sarima_summary,
    }

    return render(request, 'dashboard/graph_page.html', context)