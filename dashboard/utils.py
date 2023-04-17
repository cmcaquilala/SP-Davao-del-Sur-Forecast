import base64
from io import BytesIO
from datetime import datetime
import math

# stat-related
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from scipy import stats, special
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pymc as pm
# import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
# from statsmodels.tsa.arima_model import ARIMA

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

def get_merged_graphs(sarima_models, bayesian_arma_models, test_set):
	plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
	plot_title = 'Quarterly Predictions of Production Volume of Davao del Sur'
	plt.title(plot_title)
	plt.ylabel('Volume in Tons')
	plt.xlabel('Date')
	plt.xticks(rotation=45)
	plt.grid(True)

	plt.plot(test_set['Date'], test_set['Volume'],
	  linewidth=4, label="Dataset")
	plt.legend()
    
	date_start = test_set['Date'][test_set.index.start]

	if (len(sarima_models) + len(bayesian_arma_models) < 1):
		return get_graph()

	for model in sarima_models:
		# predict_plot = pd.concat([predictions_df, forecasts_df], ignore_index=True)
		no_of_periods = len(model.forecasts)
		forecast_dates = pd.date_range(start=date_start, periods=no_of_periods, freq="QS")

		predictions = []
		for item in model.forecasts:
			predictions.append(item['prediction'])

		plt.plot(forecast_dates, predictions,label="{0} {1}".format(
			model.get_shorthand_str(),
			"BC " + str(model.lmbda) if model.is_boxcox else ""))
		plt.legend()


	return get_graph()


def model_sarima(df, dataset_name, my_order, my_seasonal_order, is_boxcox, lmbda):

	# Train-Test Split
	train_set = df[0:132]
	test_set = df[132:]

	# ACF / PCF Plot
	# plot_pacf(train_set['Volume'],lags=60)
	# plot_acf(train_set['Volume'],lags=60)

	# Transformation
	if is_boxcox:
		lmbda = stats.boxcox(train_set['Volume'])[1] if lmbda == 0 else lmbda
		df_data = stats.boxcox(train_set['Volume'], lmbda=lmbda)
		# df_data = stats.boxcox(train_set['Volume'])[0]
	else:
		df_data = train_set['Volume']

	# transf_df_data['Volume'] = np.log(train_set['Volume'])
	# transf_df_data['Volume'] = transf_df_data['Volume'].diff()
	# transf_df_data = transf_df_data.drop(transf_df_data.index[0])

	# Model Creation
	model = SARIMAX(df_data, order=my_order, seasonal_order=my_seasonal_order)
	model_fit = model.fit()

	# print(model_fit.summary())

	# Test Set Fitting
	predictions = model_fit.forecast(len(test_set))
	if is_boxcox:
		predictions = special.inv_boxcox(predictions, lmbda)
	predictions = pd.Series(predictions, index=test_set.index)
	# predictions = pd.Series(predictions, index=test_set.index)
	# residuals = test_set['Volume'] - predictions

	predictions_df = pd.DataFrame(
		{'Date': test_set['Date'],
		'Volume': predictions})

	# Forecasts
	no_of_forecasts = 9
	forecasts = model_fit.forecast(no_of_forecasts)
	if is_boxcox:
		forecasts = special.inv_boxcox(forecasts, lmbda)

	forecast_dates = pd.date_range(test_set['Date'][test_set.index.stop-1], periods=no_of_forecasts, freq="QS")

	forecasts_df = pd.DataFrame(
		{'Date': forecast_dates,
		'Volume': forecasts})

	# Model Evaluation
	model_MSE = get_MSE(test_set['Volume'].values,predictions.values)
	model_RMSE = get_RMSE(test_set['Volume'].values,predictions.values)
	model_MAPE = get_MAPE(test_set['Volume'].values,predictions.values)

	# Graph Plotting
	points_to_display = 100

	predict_plot = pd.concat([predictions_df, forecasts_df], ignore_index=True)

	plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
	# plt.xlim([points_to_display,df.size-points_to_display])
	# plt.xlim([df['Date'][0],df['Date'][df['Date'].size-1]])
	plt.plot(df['Date'], df['Volume'])
	# plt.plot(test_set['Date'], predictions)
	plt.plot(predict_plot['Date'], predict_plot['Volume'])
	plot_title = 'Quarterly ' + dataset_name + ' Production Volume of Davao del Sur Using SARIMA' + str(my_order) + str(my_seasonal_order)
	plt.title(plot_title)
	plt.ylabel('Volume in Tons')
	plt.xlabel('Date')
	plt.xticks(rotation=45)
	plt.grid(True)

	# filename = "static/models/SARIMA({0})({1}){2}.png".format(str(my_order), str(my_seasonal_order),str((datetime.now() - datetime.utcfromtimestamp(0)).total_seconds() * 1000.0))
	# plt.savefig(filename, format = "png")
	filename = "static/models/{0} {1}{2}{3} {4}.png".format(
		dataset_name,
		"SARIMA",
		my_order,
		my_seasonal_order,
		"BC" + str(lmbda) if is_boxcox else "",
	)
	plt.savefig(filename, format = "png")
	graph = get_graph()

	return {
		"graph" : graph,
		"model" : model,
		"predictions" : predictions,
		"forecasts" : forecasts,
		"test_set" : test_set,
		"mse" : model_MSE,
		"rmse" : model_RMSE,
		"mape" : model_MAPE,
		"lmbda" : lmbda,
	}

def model_bayesian(df, dataset_name, my_order, is_box_cox):
	train_set = df[0:132]
	test_set = df[132:]

	num_samples = 10000
	forecasted_vals = []
	num_periods = test_set['Volume'].size

	with pm.Model() as bayes_model:
		#priors
		phi = pm.Normal("phi", mu=0, sigma=1, shape=my_order[0])
		# delta = pm.Normal("delta", mu=0, sigma=1, shape=q_param)
		sigma = pm.InverseGamma("sigma", alpha=0.01, beta=0.01)
		#Likelihood
		likelihood = pm.AR("x", phi, sigma, observed=train_set['Volume'])
		#posterior
		trace = pm.sample(1000, cores=2)

	phi_vals = []
	for i in range(trace.posterior.phi[0][0].size):
		phi_vals.append(trace.posterior.phi[0][:,i])

	# phi1_vals = trace.posterior.phi[0][:,0]
	# phi2_vals = trace.posterior.phi[0][:,1]
	sigma_vals = trace.posterior.sigma[0]

    # print(sigma_vals)

	for _ in range(num_samples):
		curr_vals = list(train_set['Volume'].copy())
        
		phi_val = []
		for i in range(len(phi_vals)):
			phi_val.append(np.random.choice(phi_vals[i]))

		# phi1_val = np.random.choice(phi1_vals)
		# phi2_val = np.random.choice(phi2_vals)
		sigma_val = np.random.choice(sigma_vals)
        
		# my_value = np.random.normal(0, sigma_val)
		for _ in range(num_periods):
			my_value = np.random.normal(0, sigma_val)
			for i in range(len(phi_val)):
				my_value += curr_vals[-i]*phi_val[i]
			curr_vals.append(my_value)
			# curr_vals.append(curr_vals[-1]*phi1_val + curr_vals[-2]*phi2_val + np.random.normal(0, sigma_val))
		forecasted_vals.append(curr_vals[-num_periods:]) 
	forecasted_vals = np.array(forecasted_vals)

	obtained_means = []
	for i in range(num_periods):
		plt.figure(figsize=(10,4))
		vals = forecasted_vals[:,i]
		mu, dev = round(vals.mean(), 3), round(vals.std(), 3)
		sns.distplot(vals)
		# p1 = plt.axvline(forecast[0][i], color='k')
		p2 = plt.axvline(vals.mean(), color='b')
		obtained_means.append(vals.mean())
		# plt.legend((p1,p2), ('MLE', 'Posterior Mean'), fontsize=20)
		# plt.legend(p2, 'Posterior Mean', fontsize=20)
		# plt.title('Forecasted t+%s\nPosterior Mean: %s\nMLE: %s\nSD Bayes: %s\nSD MLE: %s'%((i+1), mu, round(forecast[0][i],3), dev, round(forecast[1][i],3)), fontsize=20)

	# Diagnostics

	# Test Set Fitting
	predictions = obtained_means
	predictions = pd.Series(predictions, index=test_set.index)
	# residuals = test_set['Volume'] - predictions

	# Model Evaluation
	model_MSE = get_MSE(test_set['Volume'].values,predictions.values)
	model_RMSE = get_RMSE(test_set['Volume'].values,predictions.values)
	model_MAPE = get_MAPE(test_set['Volume'].values,predictions.values)

	# Graph Plotting
	points_to_display = 100

	plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
	# plt.xlim([points_to_display,df.size-points_to_display])
	plt.xlim([df['Date'][0],df['Date'][df['Date'].size-1]])
	plt.plot(df['Date'], df['Volume'])
	plt.plot(test_set['Date'], predictions)
	plot_title = 'Quarterly ' + dataset_name + ' Production Volume of Davao del Sur Using Bayesian ARMA' + str(my_order)
	plt.title(plot_title)
	plt.ylabel('Volume in Tons')
	plt.xlabel('Date')
	plt.xticks(rotation=45)
	plt.grid(True)
	graph = get_graph()

	return {
		"graph" : graph,
		# "model" : model,
		"predictions" : predictions,
		"test_set" : test_set,
		"mse" : model_MSE,
		"rmse" : model_RMSE,
		"mape" : model_MAPE,
	}

def get_MSE(actual, predictions):
	total = 0

	for i in range(actual.size):
		total += (actual[i] - predictions[i])**2

	return total / (actual.size - 1)

def get_RMSE(actual, predictions):
	total = 0

	for i in range(actual.size):
		total += (actual[i] - predictions[i])**2

	return math.sqrt(total / (actual.size - 1))

def get_MAPE(actual, predictions):
	total = 0

	for i in range(actual.size):
		total += math.fabs((actual[i] - predictions[i]) / actual[i])

	return (total / (actual.size)) * 100