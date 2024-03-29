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


def model_winters(dataset_data, dataset_name, train_set_idx, trend, seasonal, damped, is_boxcox, lmbda):
	# Initialization
	# 28*4 forecasts = up to 2050
	test_set_date = dataset_data.iloc[-1]['Date']
	no_of_forecasts = (2050 - (test_set_date.year + 1) + 1) * 4
	forecast_dates = pd.date_range(start=test_set_date, periods=no_of_forecasts, freq="QS")

	train_set_size = train_set_idx
	train_set = dataset_data[0:train_set_size]
	test_set = dataset_data[train_set_size:]

	# checking inputs
	if trend.lower() in ("mul","multiplicative"):
		trend = "mul"
	else:
		trend = "add"

	if seasonal.lower() in ("mul","multiplicative"):
		seasonal = "mul"
	else:
		seasonal = "add"

	# Creating Holt-Winters Model
    # Transforming
	if is_boxcox:
		if lmbda == 0:
			lmbda = stats.boxcox(dataset_data["Volume"])[1]
		df_data = stats.boxcox(train_set['Volume'], lmbda=lmbda)
	else:
		df_data = train_set['Volume']

    # Creating Model
	model = ExponentialSmoothing(df_data, seasonal_periods=4, trend=trend, damped_trend=damped, seasonal=seasonal)
	model_fit = model.fit()

    # Fitting with test set
	predictions = model_fit.forecast(len(test_set))
	if is_boxcox:
		predictions = special.inv_boxcox(predictions, lmbda)
	predictions = pd.Series(predictions, index=test_set.index)

    # Predicting future values
	forecasts = model_fit.forecast(no_of_forecasts)
	if is_boxcox:
		forecasts = special.inv_boxcox(forecasts, lmbda)

    # Model Evaluation
	model_MSE = get_MSE(test_set['Volume'].values,predictions.values)
	model_RMSE = get_RMSE(test_set['Volume'].values,predictions.values)
	model_MAPE = get_MAPE(test_set['Volume'].values,predictions.values)
	model_MAD = get_MAD(test_set['Volume'].values,predictions.values)

	forecast_dates = pd.date_range(test_set['Date'][test_set.index.stop-1], periods=no_of_forecasts, freq="QS")
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
	# plot_title = 'Quarterly ' + dataset_name + ' Production Volume of Davao del Sur Using Holt-Winters'
	# plt.title(plot_title)
	# plt.ylabel('Volume in Tons')
	# plt.xlabel('Date')
	# plt.xticks(rotation=45)
	# plt.grid(True)

	# filename = "models/{0} {1} {2} {3}.png".format(
	# 	dataset_name,
	# 	"Holt-Winters",
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
		"mse" : model_MSE,
		"rmse" : model_RMSE,
		"mape" : model_MAPE,
		"mad" : model_MAD,
		"lmbda" : lmbda,
	}