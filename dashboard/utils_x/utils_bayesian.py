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

# r_base = importr('base')
# r_utils = importr('utils')
# r_generics = importr('generics')

# r_utils.chooseCRANmirror(ind=1)
# r_utils.install_packages('stats')
# r_utils.install_packages('forecast')
# r_utils.install_packages("rstan")
# r_utils.install_packages('bayesforecast')

# r_stats = importr('stats')
# r_forecast = importr('forecast')
# r_bayesforecast = importr('bayesforecast')

def model_bayesian(filename, dataset_data, dataset_name, my_order, my_seasonal_order, is_boxcox, lmbda):

	# Initialization
	# 28*4 forecasts = up to 2050
	no_of_forecasts = 28 * 4
	train_set_size = 132
	no_of_iterations = 5000

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

	r_data_transf = r_train_set

	# get boxcox lambda
	if is_boxcox and lmbda == 0:
		r_lambda = r_forecast.BoxCox_lambda(r_train_set)

	# transform if needed
	if is_boxcox:
		r_data_transf = r_forecast.BoxCox(r_train_set, r_lambda)

	# create model
	print("where error?")
	r_model = r_bayesforecast.stan_sarima(ts=r_data_transf)
	print("miderror?")
	r_model = r_bayesforecast.stan_sarima(ts=r_data_transf, order=r_order, seasonal=r_seasonal_order,
										prior_ar=r_bayesforecast.normal(0,1), prior_ma=r_bayesforecast.normal(0,1), prior_sigma0=r_bayesforecast.inverse_gamma(0.01,0.01),
										iter = no_of_iterations)
	print("past error")

	# Getting fitted values
	r_data_fitted = r_generics.forecast(r_data_transf, model=r_model,h=no_of_forecasts)

	# --if boxcox, tranform back
	if is_boxcox:
		r_data_fitted = r_forecast.InvBoxCox(r_data_fitted[1], r_lambda)

	# metrics
	r_metrics = r_generics.accuracy(r_data_fitted,r_test_set)

	# forecasts
	r_forecasts = r_generics.forecast(r_dataset_data_ts,model=r_model,h=no_of_forecasts)

	# End of rpy2
	# ----------------------

	predictions = []
	forecasts = []

	for x in r_data_fitted:
		predictions.append(x)

	for x in r_forecasts[1]:
		forecasts.append(x)

	# Model Evaluation
	model_MSE = r_metrics[1]*r_metrics[1]
	model_RMSE = r_metrics[1]
	model_MAPE = r_metrics[4]
	model_MAD = get_MAD(test_set['Volume'].values,predictions.values)
	# model_bic = r_model[15]
	# model_aic = r_model[5]
	# model_aicc = r_model[14]

	# # Graph Plotting
	# points_to_display = 100

	forecast_start = test_set['Date'][test_set.index.stop-1] + pd.DateOffset(months=3)
	forecast_dates = pd.date_range(forecast_start, periods=no_of_forecasts, freq="QS")
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
	plot_title = 'Quarterly ' + dataset_name + ' Production Volume of Davao del Sur Using Bayesian SARIMA' + str(my_order) + str(my_seasonal_order)
	plt.title(plot_title)
	plt.ylabel('Volume in Tons')
	plt.xlabel('Date')
	plt.xticks(rotation=45)
	plt.grid(True)

	filename = "models/{0} {1}{2}{3} {4} {5}.png".format(
		dataset_name,
		"Bayesian SARIMA",
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
		# "bic" : model_BIC,
		"mse" : model_MSE,
		"rmse" : model_RMSE,
		"mape" : model_MAPE,
		"mad" : model_MAD,
		"lmbda" : lmbda,
	}