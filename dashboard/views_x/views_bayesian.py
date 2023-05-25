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
        form = BayesianARMA_add_form(request.POST)

        if(form.is_valid()):

            model = form.save(False)

            my_order = (int(request.POST["p_param"]),int(request.POST["q_param"]))
            is_boxcox = False

            bayesian_arma_model = model_bayesian(dataset_data, dataset, my_order, is_boxcox)

            model.dataset = dataset
            model.graph = bayesian_arma_model["graph"]
            model.mse = bayesian_arma_model["mse"]
            model.rmse = bayesian_arma_model["rmse"]
            model.mape = bayesian_arma_model["mape"]
            model.mad = 0

            model.save()
    
    return redirect('graphs_page', dataset)

def delete_bayesian(request, dataset, id):
    model = BayesianARMAModel.objects.get(id=id)
    model.delete()

    return redirect('graphs_page', dataset)
