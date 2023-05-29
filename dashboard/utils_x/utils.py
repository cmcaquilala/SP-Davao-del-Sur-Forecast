import base64
from io import BytesIO
from datetime import datetime
import math

# stat-related
# from statsmodels.graphics.tsaplots import plot_pacf
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.stattools import adfuller
# from scipy import stats, special
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd

# import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# from statsmodels.tsa.arima_model import ARIMA

# # Using rpy
# import rpy2
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr, data
from rpy2.robjects.packages import importr

# r_base = importr('base')
# r_utils = importr('utils')
# r_generics = importr('generics')

# r_utils.chooseCRANmirror(ind=1)
# r_utils.install_packages('stats')
# r_utils.install_packages('forecast')

# r_stats = importr('stats')
# r_forecast = importr('forecast')

def get_r_package(pkg_name):
	# Docker's Copy
	lib_dir1 = '/usr/local/lib/R/site-library'
	lib_dir2 = '/usr/lib/R/site-library'
	lib_dir3 = '/usr/lib/R/library'

	# Cedric's Copy
	# lib_dir1 = 'C:/Users/Cedric/AppData/Local/R/win-library/4.3'
	# lib_dir2 = 'C:/Program Files/R/R-4.3.0/library'
	# lib_dir3 = None

	try:
		return importr(pkg_name, suppress_messages=False, lib_loc=lib_dir1)
	except:
		try:
			return importr(pkg_name, suppress_messages=False, lib_loc=lib_dir2)
		except:
			return importr(pkg_name, suppress_messages=False, lib_loc=lib_dir3)

def get_graph():
	buffer = BytesIO()
	plt.savefig(buffer, format='png')
	buffer.seek(0)
	image_png = buffer.getvalue()
	graph = base64.b64encode(image_png)
	graph = graph.decode('utf-8')
	buffer.close()
	return graph

def get_plot(x,y):
	# ...
	graph = get_graph()
	return graph

def plot_model(dataset_data, test_set_index, model):
	test_set_date = dataset_data.iloc[test_set_index]['Date']

	no_of_periods = (2050 - test_set_date.year + 1) * 4
	forecast_dates = pd.date_range(start=test_set_date, periods=no_of_periods, freq="QS")

	predictions = model['forecasts']

	# Plotting
	plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
	plt.plot(dataset_data['Date'], dataset_data['Volume'])
	plt.plot(forecast_dates, predictions)
	# plt.plot(test_set['Date'], predictions)
	plt.xlim(datetime(year=int(model['display_start']) - 1, month=1, day = 1),
	  		datetime(year=int(model['display_end']), month=10, day = 1))
	plot_title = 'Quarterly ' + model['dataset'] + ' Production Volume of Davao del Sur Using ' + model['model_name']
	plt.title(plot_title)
	plt.ylabel('Volume in Tons')
	plt.xlabel('Date')
	plt.xticks(rotation=45)
	plt.grid(True)

	return get_graph()

def get_merged_graphs(sarima_models, bayesian_models, winters_models, lstm_models, test_set, end_year):
	plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
	plot_title = 'Quarterly Predictions of Production Volume of Davao del Sur'
	plt.title(plot_title)
	plt.ylabel('Volume in Tons')
	plt.xlabel('Date')
	plt.xticks(rotation=45)
	plt.grid(True)

	plt.plot(test_set['Date'], test_set['Volume'],
	  linewidth=4, label="Test Set")
	plt.legend()
    
	date_start = test_set['Date'][test_set.index.start]
	no_of_periods = (end_year - date_start.year + 1) * 4

	if (len(sarima_models) + len(bayesian_models) + len(winters_models) + len(lstm_models) < 1):
		return get_graph()

	for model in sarima_models:
		# predict_plot = pd.concat([predictions_df, forecasts_df], ignore_index=True)
		# no_of_periods = len(model.forecasts)
		forecast_dates = pd.date_range(start=date_start, periods=no_of_periods, freq="QS")

		predictions = []
		for i in range(no_of_periods):
			predictions.append(model['forecasts'][i])

		plt.plot(forecast_dates, predictions,label="{0} {1}".format(
			str(model['model_name']),
			"BC " + str(model['lmbda']) if model['is_boxcox'] else ""))
		plt.legend()

	for model in bayesian_models:
		# predict_plot = pd.concat([predictions_df, forecasts_df], ignore_index=True)
		# no_of_periods = len(model.forecasts)
		forecast_dates = pd.date_range(start=date_start, periods=no_of_periods, freq="QS")

		predictions = []
		for i in range(no_of_periods):
			predictions.append(model['forecasts'][i])

		plt.plot(forecast_dates, predictions,label="{0} {1}".format(
			str(model['model_name']),
			"BC " + str(model['lmbda']) if model['is_boxcox'] else ""))
		plt.legend()

	for model in winters_models:
		# predict_plot = pd.concat([predictions_df, forecasts_df], ignore_index=True)
		# no_of_periods = len(model.forecasts)
		forecast_dates = pd.date_range(start=date_start, periods=no_of_periods, freq="QS")

		predictions = []
		for i in range(no_of_periods):
			predictions.append(model['forecasts'][i])

		plt.plot(forecast_dates, predictions,label="{0} {1}".format(
			str(model['model_name']),
			"BC " + str(model['lmbda']) if model['is_boxcox'] else ""))
		plt.legend()

	for model in lstm_models:
		# predict_plot = pd.concat([predictions_df, forecasts_df], ignore_index=True)
		# no_of_periods = len(model.forecasts)
		forecast_dates = pd.date_range(start=date_start, periods=no_of_periods, freq="QS")

		predictions = []
		for i in range(no_of_periods):
			predictions.append(model['forecasts'][i])

		plt.plot(forecast_dates, predictions,label="{0} {1}".format(
			str(model['model_name']),
			"BC " + str(model['lmbda']) if model['is_boxcox'] else ""))
		plt.legend()


	return get_graph()

def get_MSE(actual, predictions):
	total = 0

	for i in range(actual.size):
		total += (actual[i] - predictions[i])**2

	return total / (actual.size - 1)

def get_RMSE(actual, predictions):
	return math.sqrt(get_MSE(actual, predictions))

def get_MAD(actual,predictions):
	total = 0

	for i in range(actual.size):
		total += math.fabs(actual[i] - predictions[i])

	return total / (actual.size)

def get_MAPE(actual, predictions):
	total = 0

	for i in range(actual.size):
		total += math.fabs((actual[i] - predictions[i]) / actual[i])

	return (total / (actual.size)) * 100

def get_timestamp():
	return str((datetime.now() - datetime.utcfromtimestamp(0)).total_seconds() * 1000.0)