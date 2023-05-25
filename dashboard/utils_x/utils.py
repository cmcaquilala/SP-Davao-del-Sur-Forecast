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
# import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# from statsmodels.tsa.arima_model import ARIMA

# # Using rpy
# import rpy2
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr, data

# r_base = importr('base')
# r_utils = importr('utils')
# r_generics = importr('generics')

# r_utils.chooseCRANmirror(ind=1)
# r_utils.install_packages('stats')
# r_utils.install_packages('forecast')

# r_stats = importr('stats')
# r_forecast = importr('forecast')

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