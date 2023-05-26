import datetime as DateTime
from dateutil.relativedelta import relativedelta
import csv
import json

from django.shortcuts import render, redirect
from django.templatetags.static import static
from .models import *
from .forms import *

from .utils_x.utils_sarima import *
from .utils_x.utils_bayesian import *
from .utils_x.utils_winters import *
from .utils_x.utils_lstm import *
from .utils_x.utils import *

from .views_x.views_sarima import *
from .views_x.views_bayesian import *
from .views_x.views_winters import *
from .views_x.views_lstm import *

# stat-related
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create your views here.
def index_page(request):
    return render(request, 'dashboard/index.html')

def change_summary_year(request, dataset):
    request.session['summary_end_year'] = int(request.POST['summary_end_year'])
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
    bayesian_form = BayesianSARIMA_add_form(request.POST)
    winters_form = HoltWinters_add_form(request.POST)
    lstm_form = LSTM_add_form(request.POST)

    # Load all models
    sarima_models = []
    bayesian_models = []
    winters_models = []
    lstm_models = []

    for x in SARIMAModel.objects.filter(dataset=dataset):
        sarima_models.append(x)
    for x in BayesianSARIMAModel.objects.filter(dataset=dataset):
        bayesian_models.append(x)
    for x in HoltWintersModel.objects.filter(dataset=dataset):
        winters_models.append(x)
    for x in LSTMModel.objects.filter(dataset=dataset):
        lstm_models.append(x)

    # temporary date
    test_set = dataset_data[132:]

    summary_end_year = 2025
    if 'summary_end_year' in request.session:
        summary_end_year = int(request.session['summary_end_year'])
    else:
        request.session['summary_end_year'] = summary_end_year

    merged_graphs = get_merged_graphs(sarima_models, bayesian_models, winters_models, lstm_models, test_set, summary_end_year)

    context = {
        'dataset' : dataset,
        'merged_graphs' : merged_graphs,
        'sarima_models' : sarima_models,
        'bayesian_models' : bayesian_models,
        'winters_models' : winters_models,
        'lstm_models' : lstm_models,
        'sarima_form' : sarima_form,
        'bayesian_form' : bayesian_form,
        'winters_form' : winters_form,
        'lstm_form' : winters_form,
    }

    return render(request, 'dashboard/graph_page.html', context)