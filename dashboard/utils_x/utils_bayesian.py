import base64
from io import BytesIO
from datetime import datetime
import math
from .utils import *

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