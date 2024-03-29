import datetime as DateTime
from dateutil.relativedelta import relativedelta
import csv
import json

from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseNotFound, HttpResponseServerError
from django.templatetags.static import static
from django.core.files.uploadedfile import SimpleUploadedFile
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

    request.session["saved_{0}".format(current_type)].remove(current_model)
    request.session.modified = True

    return redirect('graphs_page', dataset)


def download_results(request, dataset, id):
    model_types = ["sarima", "bayesian", "winters", "lstm"]
    model_to_save = {}

    for model_type in model_types:
        for model in request.session["saved_{0}".format(model_type)]:
            if model['id'] == id:
                model_to_save = model

    download_file = json.dumps(model_to_save, indent=4)
    response = HttpResponse(download_file, content_type='application/json')
    response['Content-Disposition'] = 'attachment; filename={0}'.format("saved_model.json")

    # with open("model {0}".format(id), "w") as outfile:
    #     outfile.write(download_file)

    return response


def upload_results(request, dataset):
    a = request.FILES
    
    json_file = request.FILES['json_file']
    model_results = json.load(json_file)
    save_json_to_session(request, model_results)

    return redirect('graphs_page', dataset)


def change_test_set(request, dataset):
    dataset_dates = "{0}_dataset_dates".format(dataset.lower())
    list_of_dates = request.session[dataset_dates]
    old_year = int(request.session['{0}_test_set_date'.format(dataset.lower())][:4])
    input_year = int(request.POST['test_set_year'])

    # input checking
    lbound = int(list_of_dates[4][:4])
    ubound = int(list_of_dates[-4][:4]) - 1

    if input_year < lbound or input_year > ubound:
        return redirect('edit_dataset_page', dataset)

    clear_all_models(request, dataset)
    request.session.modified = True

    request.session['{0}_test_set_date'.format(dataset.lower())] = "{0}-01-01".format(input_year)
    request.session['{0}_test_set_index'.format(dataset.lower())] -= (old_year - input_year) * 4

    k = request.session['{0}_test_set_date'.format(dataset.lower())]
    l = request.session['{0}_test_set_index'.format(dataset.lower())]

    return redirect('edit_dataset_page', dataset)


def add_datapoint(request, dataset):
    # Check input
    # user_input = request.POST['new_volume']
    user_input = request.POST['new_volume']

    try:
        user_input = float(user_input)

        if user_input < 0:
            return redirect('edit_dataset_page', dataset)
    except:
        return redirect('edit_dataset_page', dataset)

    clear_all_models(request, dataset)
    request.session.modified = True

    # Look for the last date
    dataset_name = "{0}_dataset_data".format(dataset.lower())
    dataset_dates = "{0}_dataset_dates".format(dataset.lower())
    last_date = datetime.strptime(request.session[dataset_dates][-1], '%Y-%m-%d')

    # Add 3 to last date, add volume to array
    request.session[dataset_dates].append(
        (last_date + relativedelta(months=3)).strftime('%Y-%m-%d')
        )
    request.session[dataset_name].append(user_input)
    request.session.modified = True

    return redirect('edit_dataset_page', dataset)


def edit_datapoint(request, dataset, date):
    # Check input
    # user_input = request.POST['new_volume']
    user_input = request.POST['new_volume']

    try:
        user_input = float(user_input)

        if user_input < 0:
            return redirect('edit_dataset_page', dataset)
    except:
        return redirect('edit_dataset_page', dataset)

    clear_all_models(request, dataset)
    request.session.modified = True

    # Search for cell
    dataset_name = "{0}_dataset_data".format(dataset.lower())
    dataset_dates = "{0}_dataset_dates".format(dataset.lower())
    datapoint_index = request.session[dataset_dates].index(date)

    # Replace it
    request.session[dataset_name][datapoint_index] = str(user_input)
    request.session.modified = True

    return redirect('edit_dataset_page', dataset)


def delete_datapoint(request, dataset):
    clear_all_models(request, dataset)
    request.session.modified = True

    dataset_name = "{0}_dataset_data".format(dataset.lower())
    dataset_dates = "{0}_dataset_dates".format(dataset.lower())

    # Replace it
    request.session[dataset_dates].pop()
    request.session[dataset_name].pop()
    request.session.modified = True

    return redirect('edit_dataset_page', dataset)

def reload_models_page(request, dataset):
    reload_dataset(request, dataset)
    return redirect('graphs_page', dataset)

def reload_dataset_page(request, dataset):
    reload_dataset(request, dataset)
    return redirect('edit_dataset_page', dataset)


def edit_dataset_page(request, dataset):
    # Load dataset
    dataset_dates = "{0}_dataset_dates".format(dataset.lower())
    dataset_name = "{0}_dataset_data".format(dataset.lower())

    dataset_dates = request.session[dataset_dates]
    dataset_data = request.session[dataset_name]

    test_set_index = request.session["{0}_test_set_index".format(dataset.lower())]
    test_set_date = request.session["{0}_test_set_date".format(dataset.lower())]
    test_set_year = int(test_set_date[:4])

    dataset_table = []
    curr_date = datetime.strptime(dataset_dates[0], '%Y-%m-%d')

    for i in range(len(dataset_data)):
        year = curr_date.year
        quarter = curr_date.month // 3 + 1

        date = curr_date
        volume = float(dataset_data[i])

        value_dict = {
            'period' : "{0} Q{1}".format(year, quarter),
            'date' : date.strftime('%Y-%m-%d'),
            'volume' : volume,
        }

        dataset_table.append(value_dict)
        curr_date += relativedelta(months=3)

    context = {
        'dataset_table' : dataset_table,
        'dataset' : dataset,
        'test_set_index' : test_set_index,
        'test_set_date' : test_set_date,
        'test_set_year' : test_set_year,
    }

    return render(request, 'dashboard/edit_dataset_page.html', context)

def graphs_page(request, dataset):

    # Load session data
    dataset_dates = "{0}_dataset_dates".format(dataset.lower())
    dataset_name = "{0}_dataset_data".format(dataset.lower())
    dataset_data = pd.DataFrame()

    # Basically if session just started
    if dataset_name not in request.session:
        request.session['saved_sarima'] = []
        request.session['saved_bayesian'] = []
        request.session['saved_winters'] = []
        request.session['saved_lstm'] = []
        request.session['rice_test_set_index'] = 132
        request.session['rice_test_set_date'] = '2020-01-01'
        request.session['corn_test_set_index'] = 132
        request.session['corn_test_set_date'] = '2020-01-01'
        dataset_data = reload_dataset(request, 'corn')
        dataset_data = reload_dataset(request, 'rice')
    else:
        dataset_data = pd.DataFrame({
            'Date' : request.session[dataset_dates],
            'Volume' : request.session[dataset_name],
        })
        dataset_data['Volume'] = pd.to_numeric(dataset_data['Volume'])
        dataset_data['Date'] = pd.to_datetime(dataset_data['Date'])

    # Load test set info
    test_set_index = request.session['{0}_test_set_index'.format(dataset.lower())]
    test_set = dataset_data[test_set_index:]

    # test_set_date = pd.to_datetime(request.session['test_set_date'])
    # test_set_index = dataset_data.index[dataset_data['Date'] == test_set_date][0]
    # request.session['test_set_index'] = test_set_index

    # Loads forms for modal
    sarima_form = SARIMA_add_form(request.POST)
    bayesian_form = BayesianSARIMA_add_form(request.POST)
    winters_form = HoltWinters_add_form(request.POST)
    lstm_form = LSTM_add_form(request.POST)
    upload_form = JSON_upload_form(request.POST)

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
                model['graph'] = plot_model(dataset_data, test_set_index, model)
                sarima_models.append(model)

    if 'saved_bayesian' not in request.session:
        request.session['saved_bayesian'] = []
    else:
        for model in request.session['saved_bayesian']:
            if model['dataset'] == dataset:
                model['graph'] = plot_model(dataset_data, test_set_index, model)
                bayesian_models.append(model)

    if 'saved_winters' not in request.session:
        request.session['saved_winters'] = []
    else:
        for model in request.session['saved_winters']:
            if model['dataset'] == dataset:
                model['graph'] = plot_model(dataset_data, test_set_index, model)
                winters_models.append(model)

    if 'saved_lstm' not in request.session:
        request.session['saved_lstm'] = []
    else:
        for model in request.session['saved_lstm']:
            if model['dataset'] == dataset:
                model['graph'] = plot_model(dataset_data, test_set_index, model)
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
        'upload_form' : upload_form,
    }

    return render(request, 'dashboard/graph_page.html', context)

