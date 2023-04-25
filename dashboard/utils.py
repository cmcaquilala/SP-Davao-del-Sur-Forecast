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


def model_sarima(filename, dataset_data, dataset_name, my_order, my_seasonal_order, is_boxcox, lmbda):

	# Initialization
	no_of_forecasts = 12
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


	# Model Evaluation
	model_MSE = r_metrics[1]*r_metrics[1]
	model_RMSE = r_metrics[1]
	model_MAPE = r_metrics[4]
	model_BIC = r_model[15][0]
	# model_aic = r_model[5][0]
	# model_aicc = r_model[14][0]

	# # Graph Plotting
	# points_to_display = 100

	predictions = []
	forecasts = []

	for x in r_one_step_forecasts:
		predictions.append(x)

	for x in r_forecasts[3]:
		forecasts.append(x)

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

	return total / (actual.size)

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

def get_timestamp():
	return str((datetime.now() - datetime.utcfromtimestamp(0)).total_seconds() * 1000.0)