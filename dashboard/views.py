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

    # rice_data = pd.read_csv("rice data.csv")
    # corn_data = pd.read_csv("corn data.csv")

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

    sarima_chart = model_sarima(rice_data, my_order, my_seasonal_order)

    context = {
        # 'rice_data' : arr,
        'sarima_chart' : sarima_chart,
    }

    return render(request, 'dashboard/index.html', context)



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

    sarima_chart = model_sarima(corn_data, my_order, my_seasonal_order)

    context = {
        # 'corn_data' : arr,
        'sarima_chart' : sarima_chart,
    }

    return render(request, 'dashboard/index.html', context)