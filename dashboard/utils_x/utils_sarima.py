import base64
from io import BytesIO
from datetime import datetime
import math
from .utils import *

# stat-related
# from statsmodels.graphics.tsaplots import plot_pacf
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.tsa.stattools import adfuller
# from scipy import stats, special
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pymc as pm
# import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
# from statsmodels.tsa.arima_model import ARIMA

# Using rpy
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data

r_base = importr('base')
r_utils = importr('utils')
r_generics = importr('generics')

r_utils.chooseCRANmirror(ind=1)
r_utils.install_packages('stats')
r_utils.install_packages('forecast')

r_stats = importr('stats')
r_forecast = importr('forecast')


def model_sarima(filename, dataset_data, dataset_name, my_order, my_seasonal_order, is_boxcox, lmbda):

	# Initialization
	# 28*4 forecasts = up to 2050
	no_of_forecasts = 28 * 4
	train_set_size = 132

	train_set = dataset_data[0:train_set_size]
	test_set = dataset_data[train_set_size:]

	# ----------------------
	# Using rpy2:
	# Convert inputs
	r_order = robjects.FloatVector([my_order[0],my_order[1],my_order[2]])
	r_seasonal_order = robjects.FloatVector([my_seasonal_order[0],my_seasonal_order[1],my_seasonal_order[2]])
	r_null = robjects.r['as.null']()

	# import data
	r_dataset_data = r_utils.read_csv(filename)
	r_dataset_data_ts = r_stats.ts(data = r_dataset_data[1], frequency = 4, start = [1987,1])

	# train-test split
	r_train_set = r_stats.ts(r_dataset_data_ts[0:train_set_size], frequency = 4, start = [1987,1])
	r_test_set = r_stats.ts(r_dataset_data_ts[train_set_size:len(r_dataset_data_ts)], frequency = 4, start = [2020,1])

	# get boxcox lambda
	if is_boxcox and lmbda == 0:
		r_lambda = r_forecast.BoxCox_lambda(r_train_set)
		lmbda = r_lambda[0]
	elif is_boxcox:
		r_lambda = lmbda
	else:
		r_lambda = r_null

	# # create model
	r_model = r_forecast.Arima(r_train_set, r_order, r_seasonal_order, r_null, True, False, False, r_lambda)

	# fitting w/ test set
	# r_predictions = r_stats.ts(r_test_set,start = [2020,1],frequency = 4)

	# creating one-step-ahead values
	r_new_model = r_forecast.Arima(r_dataset_data_ts, r_order, r_seasonal_order, r_null, True, False, False,
								r_lambda, False, "CSS-ML", r_model)
	r_one_step_forecasts = r_stats.fitted(r_new_model)[len(r_train_set):len(r_dataset_data_ts)]

	# metrics
	r_metrics = r_generics.accuracy(r_one_step_forecasts,r_test_set)

	# forecasts
	r_forecasts = r_generics.forecast(r_dataset_data_ts,model=r_model,h=no_of_forecasts)

	# End of rpy2
	# ----------------------

	predictions = []
	forecasts = []

	for x in r_one_step_forecasts:
		predictions.append(x)

	for x in r_forecasts[3]:
		forecasts.append(x)

	# Model Evaluation
	model_MSE = r_metrics[1]*r_metrics[1]
	model_RMSE = r_metrics[1]
	model_MAPE = r_metrics[4]
	model_MAD = get_MAD(test_set['Volume'].values,predictions.values)
	model_BIC = r_model[15][0]
	# model_aic = r_model[5][0]
	# model_aicc = r_model[14][0]

	forecast_dates = pd.date_range(test_set['Date'][test_set.index.stop-1], periods=no_of_forecasts, freq="QS")
	predictions_df = pd.DataFrame(
		{'Date': test_set['Date'],
		'Volume': predictions})
	forecasts_df = pd.DataFrame(
		{'Date': forecast_dates,
		'Volume': forecasts})

	predict_plot = pd.concat([predictions_df, forecasts_df], ignore_index=True)

	# Plotting
	plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
	plt.plot(dataset_data['Date'], dataset_data['Volume'])
	plt.plot(predict_plot['Date'], predict_plot['Volume'])
	# plt.plot(test_set['Date'], predictions)
	plot_title = 'Quarterly ' + dataset_name + ' Production Volume of Davao del Sur Using SARIMA' + str(my_order) + str(my_seasonal_order)
	plt.title(plot_title)
	plt.ylabel('Volume in Tons')
	plt.xlabel('Date')
	plt.xticks(rotation=45)
	plt.grid(True)

	filename = "models/{0} {1}{2}{3} {4} {5}.png".format(
		dataset_name,
		"SARIMA",
		my_order,
		my_seasonal_order,
		"BC" + str(lmbda) if is_boxcox else "",
		get_timestamp(),
	)
	plt.savefig("static/images/" + filename, format = "png")
	graph = get_graph()

	return {
		"graph" : graph,
		"filename" : filename,
		"predictions" : predictions,
		"forecasts" : forecasts,
		"test_set" : test_set,
		"bic" : model_BIC,
		"mse" : model_MSE,
		"rmse" : model_RMSE,
		"mape" : model_MAPE,
		"mad" : model_MAD,
		"lmbda" : lmbda,
	}