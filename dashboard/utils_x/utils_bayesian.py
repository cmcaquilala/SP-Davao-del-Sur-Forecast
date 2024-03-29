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


def model_bayesian(dataset_data, dataset_name, train_set_idx, my_order, my_seasonal_order, is_boxcox, lmbda):
	r_base = get_r_package('base')
	r_utils = get_r_package('utils')
	r_generics = get_r_package('generics')

	r_stats = get_r_package('stats')
	r_forecast = get_r_package('forecast')
	r_bayesforecast = get_r_package('bayesforecast')

	# Initialization
	# 28*4 forecasts = up to 2050
	test_set_date = dataset_data.iloc[-1]['Date']
	no_of_forecasts = (2050 - (test_set_date.year + 1) + 1) * 4
	forecast_dates = pd.date_range(start=test_set_date, periods=no_of_forecasts, freq="QS")

	train_set_size = train_set_idx
	train_set = dataset_data[0:train_set_size]
	test_set = dataset_data[train_set_size:]

	no_of_iterations = 5000

	# ----------------------
	# Using rpy2:
	# Convert inputs
	r_order = robjects.FloatVector([my_order[0],my_order[1],my_order[2]])
	r_seasonal_order = robjects.FloatVector([my_seasonal_order[0],my_seasonal_order[1],my_seasonal_order[2]])
	r_null = robjects.r['as.null']()

	# import data
	r_dataset_data = [float(i) for i in dataset_data['Volume'].values.tolist()]
	r_dataset_data_ts = r_stats.ts(data = r_base.as_numeric(r_dataset_data), frequency = 4, start = [1987,1])

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

	# transform if needed
	if is_boxcox:
		r_data_transf = r_forecast.BoxCox(r_train_set, r_lambda)

	# create model
	r_model = r_bayesforecast.stan_sarima(ts=r_data_transf, order=r_order, seasonal=r_seasonal_order,
										prior_ar=r_bayesforecast.normal(0,1), prior_ma=r_bayesforecast.normal(0,1), prior_sigma0=r_bayesforecast.inverse_gamma(0.01,0.01),
										iter = no_of_iterations)

	# Getting fitted values
	r_data_fitted = r_generics.forecast(r_data_transf, model=r_model,h=len(test_set))
	
	# --if boxcox, tranform back
	if is_boxcox:
		r_data_fitted = r_forecast.InvBoxCox(r_data_fitted[1], r_lambda)
	else:
		r_data_fitted = r_data_fitted[1]

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
	model_MAD = get_MAD(test_set['Volume'].values,predictions)
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
	# plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
	# plt.plot(dataset_data['Date'], dataset_data['Volume'])
	# plt.plot(predict_plot['Date'], predict_plot['Volume'])
	# plot_title = 'Quarterly ' + dataset_name + ' Production Volume of Davao del Sur Using SARIMA' + str(my_order) + str(my_seasonal_order)
	# plt.title(plot_title)
	# plt.ylabel('Volume in Tons')
	# plt.xlabel('Date')
	# plt.xticks(rotation=45)
	# plt.grid(True)

	# filename = "models/{0} {1}{2}{3} {4} {5}.png".format(
	# 	dataset_name,
	# 	"SARIMA",
	# 	my_order,
	# 	my_seasonal_order,
	# 	"BC" + str(lmbda) if is_boxcox else "",
	# 	get_timestamp(),
	# )
	# plt.savefig("static/images/" + filename, format = "png")
	graph = get_graph()

	return {
		"graph" : graph,
		# "filename" : filename,
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