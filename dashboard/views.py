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
    if request.POST['summary_end_year'] in (None, "") or int(request.POST['summary_end_year']) < 2022:
        request.session['summary_end_year'] = 2023
    else:
        request.session['summary_end_year'] = int(request.POST['summary_end_year'])
    return redirect('graphs_page', dataset)

def change_model_year(request, dataset, id):
    current_model = None
    for model in request.session['saved_sarima']:
        if str(model['id']) == str(id):
            current_model = model
    for model in request.session['saved_bayesian']:
        if str(model['id']) == str(id):
            current_model = model
    for model in request.session['saved_winters']:
        if str(model['id']) == str(id):
            current_model = model
    for model in request.session['saved_lstm']:
        if str(model['id']) == str(id):
            current_model = model
    
    if current_model == None:
        redirect('graphs_page', dataset)

    display_start = int(request.POST['display_start'])
    display_end = int(request.POST['display_end'])

    if(display_start in (None, "")
       or display_start < 1987):
        display_start = 1987
    if(display_end in (None, "")
       or display_end > 2050):
        display_end = 2025
    if(display_end <= display_start):
        display_start = 1987
        display_end = 2025

    current_model['display_start'] = str(display_start)
    current_model['display_end'] = str(display_end)
    request.session.modified = True

    return redirect('graphs_page', dataset)

def delete_model(request, dataset, id):
    current_model = None
    current_type = ""
    for model in request.session['saved_sarima']:
        if str(model['id']) == str(id):
            current_model = model
            current_type = "sarima"
    for model in request.session['saved_bayesian']:
        if str(model['id']) == str(id):
            current_model = model
            current_type = "bayesian"
    for model in request.session['saved_winters']:
        if str(model['id']) == str(id):
            current_model = model
            current_type = "winters"
    for model in request.session['saved_lstm']:
        if str(model['id']) == str(id):
            current_model = model
            current_type = "lstm"
    
    if current_model == None:
        redirect('graphs_page', dataset)

    request.session["saved_{0}".format(current_type)].remove(model)
    request.session.modified = True

    return redirect('graphs_page', dataset)

def graphs_page(request, dataset):

    # Load dataset
    dataset_dates = "{0}_dataset_dates".format(dataset.lower())
    dataset_name = "{0}_dataset_data".format(dataset.lower())
    dataset_data = pd.DataFrame()

    if dataset_name not in request.session:
        request.session[dataset_dates] = []
        request.session[dataset_name] = []

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

        request.session[dataset_dates] = dataset_data['Date'].astype(str).tolist()
        request.session[dataset_name] = dataset_data['Volume'].tolist()

        request.session.modified = True

    else:
        dataset_data = pd.DataFrame({
            'Date' : request.session[dataset_dates],
            'Volume' : request.session[dataset_name]},)
        dataset_data['Volume'] = pd.to_numeric(request.session[dataset_name])
        dataset_data['Date'] = pd.to_datetime(request.session[dataset_dates])

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

    # for x in SARIMAModel.objects.filter(dataset=dataset):
    #     sarima_models.append(x)
    # for x in BayesianSARIMAModel.objects.filter(dataset=dataset):
    #     bayesian_models.append(x)
    # for x in HoltWintersModel.objects.filter(dataset=dataset):
    #     winters_models.append(x)
    # for x in LSTMModel.objects.filter(dataset=dataset):
    #     lstm_models.append(x)

    # collect all models saved in session
    if 'saved_sarima' not in request.session:
        request.session['saved_sarima'] = []
    else:
        for model in request.session['saved_sarima']:
            if model['dataset'] == dataset:
                model['graph'] = plot_model(dataset_data, model)
                sarima_models.append(model)

    if 'saved_bayesian' not in request.session:
        request.session['saved_bayesian'] = []
    else:
        for model in request.session['saved_bayesian']:
            if model['dataset'] == dataset:
                model['graph'] = plot_model(dataset_data, model)
                bayesian_models.append(model)

    if 'saved_winters' not in request.session:
        request.session['saved_winters'] = []
    else:
        for model in request.session['saved_winters']:
            if model['dataset'] == dataset:
                model['graph'] = plot_model(dataset_data, model)
                winters_models.append(model)

    if 'saved_lstm' not in request.session:
        request.session['saved_lstm'] = []
    else:
        for model in request.session['saved_lstm']:
            if model['dataset'] == dataset:
                model['graph'] = plot_model(dataset_data, model)
                lstm_models.append(model)

    # collect summary
    summary_end_year = 2025
    if 'summary_end_year' in request.session:
        summary_end_year = int(request.session['summary_end_year'])
    else:
        request.session['summary_end_year'] = summary_end_year

    # error handling for end_year
    if summary_end_year < 2023 or request.session['summary_end_year'] < 2023:
        summary_end_year = 2023
        request.session['summary_end_year'] = summary_end_year
    elif summary_end_year > 2050 or request.session['summary_end_year'] > 2050:
        summary_end_year = 2050
        request.session['summary_end_year'] = summary_end_year

    test_set = dataset_data[132:]
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
        'lstm_form' : lstm_form,
    }

    return render(request, 'dashboard/graph_page.html', context)