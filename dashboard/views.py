from django.shortcuts import render
from .utils import *
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
    # rice_data['Volume'] = rice_data['Volume'].astype(float)
    # rice_data['Date'] = rice_data['Date'].astype(datetime)

    # json_records = rice_data.reset_index().to_json(orient ='records')
    # arr = []
    # arr = json.loads(json_records)

    my_order = (1,0,0)
    my_seasonal_order = (1, 0, 1, 4)

    sarima_model = model_sarima(rice_data, "Rice", my_order, my_seasonal_order)

    sarima_summary = {
        'graph' : sarima_model["graph"],
        'order' : my_order,
        'seasonal_order' : my_seasonal_order,
        'mse' : '{0:.2f}'.format(sarima_model["mse"]),
        'rmse' : '{0:.4f}'.format(sarima_model["rmse"]),
        'mape' : '{0:.4f}'.format(sarima_model["mape"]),
    }

    context = {
        'dataset' : dataset_name,
        'sarima' : sarima_summary,
    }

    return render(request, 'dashboard/graph_page.html', context)



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