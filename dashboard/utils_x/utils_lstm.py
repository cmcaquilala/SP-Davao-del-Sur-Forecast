import base64
from io import BytesIO
from datetime import datetime
import math
from .utils import *

# stat-related
# from statsmodels.graphics.tsaplots import plot_pacf
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.tsa.stattools import adfuller
from scipy import stats, special
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pymc as pm
# import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# reproducibility
import os
tf.keras.utils.set_random_seed(5)
os.environ['PYTHONHASHSEED']=str(5)

def model_lstm(dataset_data, dataset_name, train_set_idx, n_inputs, n_epochs, n_units, is_boxcox, lmbda):
	# Initialization
	# 28*4 forecasts = up to 2050
	activation = 'relu'
	test_set_date = dataset_data.iloc[-1]['Date']
	no_of_forecasts = (2050 - (test_set_date.year + 1) + 1) * 4
	forecast_dates = pd.date_range(start=test_set_date, periods=no_of_forecasts, freq="QS")

	train_set_size = train_set_idx
	train_set = dataset_data[0:train_set_size]
	test_set = dataset_data[train_set_size:]

	# Creating LSTM Model
    # Transforming
	lmbda = stats.boxcox(dataset_data["Volume"])[1]

	if is_boxcox:
		if lmbda == 0:
			lmbda = stats.boxcox(dataset_data["Volume"])[1]
		transf_volume = stats.boxcox(train_set['Volume'], lmbda=lmbda)
		df_data = pd.DataFrame({
			'Date' : train_set['Date'],
			'Volume' : transf_volume,
		})
	else:
		df_data = train_set

	scaler = MinMaxScaler()
	scaler.fit(df_data.set_index('Date'))
	scaled_train = scaler.transform(df_data.set_index('Date'))
	scaled_test = scaler.transform(df_data.set_index('Date'))
	df_data = scaled_train

	# Creating Model
	generator = TimeseriesGenerator(df_data, df_data, length=n_inputs, batch_size=1)
	model = Sequential()
	model.add(LSTM(n_units, activation=activation, input_shape=(n_inputs, 1)))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')

	early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
	model.fit(generator,epochs=n_epochs,callbacks=early_stop)

	# Fitting with test set
	transf_prediction_results = []
	first_eval_batch = df_data[-n_inputs:]
	print(type(first_eval_batch))
	current_batch = first_eval_batch.reshape((1, n_inputs, 1))

	for i in range(len(test_set)):
		# get the prediction value for the first batch
		current_pred = model.predict(current_batch)[0]
		
		# append the prediction into the array
		transf_prediction_results.append(current_pred) 
		
		# use the prediction to update the batch and remove the first value
		current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

	prediction_results = scaler.inverse_transform(transf_prediction_results)
	prediction_results = [item[0] for item in prediction_results]

	predictions = pd.Series(prediction_results, index = test_set['Date'])
	if is_boxcox:
		predictions = special.inv_boxcox(predictions, lmbda)
		predictions = pd.Series(predictions)

	# Predicting future values
	forecast_results = []
	first_eval_batch = np.array(transf_prediction_results[-n_inputs:])
	print(type(first_eval_batch))
	current_batch = first_eval_batch.reshape((1, n_inputs, 1))

	for i in range(no_of_forecasts):
		# get the prediction value for the first batch
		current_pred = model.predict(current_batch)[0]
		
		# append the prediction into the array
		forecast_results.append(current_pred) 
		
		# use the prediction to update the batch and remove the first value
		current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

	forecast_results = scaler.inverse_transform(forecast_results)
	forecast_results = [item[0] for item in forecast_results]

	forecasts = pd.Series(forecast_results)
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
		'Volume': predictions.values})
	forecasts_df = pd.DataFrame(
		{'Date': forecast_dates,
		'Volume': forecasts})

	predict_plot = pd.concat([predictions_df, forecasts_df], ignore_index=True)

	# Plotting
	# plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
	# plt.plot(dataset_data['Date'], dataset_data['Volume'])
	# plt.plot(predict_plot['Date'], predict_plot['Volume'])
	# # plt.plot(test_set['Date'], predictions)
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

	predictions = predictions.values.tolist()
	forecasts = forecasts.values.tolist()

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