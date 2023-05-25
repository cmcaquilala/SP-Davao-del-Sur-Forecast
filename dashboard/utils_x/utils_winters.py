import base64
from io import BytesIO
from datetime import datetime
import math
from .utils import *

# stat-related
# from statsmodels.graphics.tsaplots import plot_pacf
# from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.stattools import adfuller
from scipy import stats, special
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pymc as pm
# import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
# from statsmodels.tsa.arima_model import ARIMA


def model_winters(filename, dataset_data, dataset_name, is_boxcox, lmbda):
	# Initialization
	no_of_forecasts = 12
	train_set_size = 132

	train_set = dataset_data[0:train_set_size]
	test_set = dataset_data[train_set_size:]

	# Creating Holt-Winters Model

    # Transforming
	if is_boxcox:
		lmbda = stats.boxcox(dataset_data["Volume"])[1]
		df_data = stats.boxcox(train_set['Volume'], lmbda=lmbda)
	else:
		df_data = train_set['Volume']

    # Creating Model
	model = ExponentialSmoothing(df_data, seasonal_periods=4, 
								trend='additive', seasonal='additive')
	model_fit = model.fit()

    # Fitting with test set
	predictions = model_fit.forecast(len(test_set))
	if is_boxcox:
		predictions = special.inv_boxcox(predictions, lmbda)
	predictions = pd.Series(predictions, index=test_set.index)

    # Predicting future values
	no_of_forecasts = 12
	forecasts = model_fit.forecast(no_of_forecasts)
	if is_boxcox:
		forecasts = special.inv_boxcox(forecasts, lmbda)

    # Model Evaluation
	model_MSE = get_MSE(test_set['Volume'].values,predictions.values)
	model_RMSE = get_RMSE(test_set['Volume'].values,predictions.values)
	model_MAPE = get_MAPE(test_set['Volume'].values,predictions.values)

	# Model Evaluation
	# model_MSE = r_metrics[1]*r_metrics[1]
	# model_RMSE = r_metrics[1]
	# model_MAPE = r_metrics[4]
	# model_BIC = r_model[15][0]
	# model_aic = r_model[5][0]
	# model_aicc = r_model[14][0]

	# # Graph Plotting
	# points_to_display = 100

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
	plot_title = 'Quarterly ' + dataset_name + ' Production Volume of Davao del Sur Using Holt-Winters'
	plt.title(plot_title)
	plt.ylabel('Volume in Tons')
	plt.xlabel('Date')
	plt.xticks(rotation=45)
	plt.grid(True)

	filename = "models/{0} {1} {2} {3}.png".format(
		dataset_name,
		"Holt-Winters",
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
		"mse" : model_MSE,
		"rmse" : model_RMSE,
		"mape" : model_MAPE,
		"lmbda" : lmbda,
	}