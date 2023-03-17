import base64
from io import BytesIO
import datetime as DateTime

# stat-related
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

def model_sarima(df, my_order, my_seasonal_order):

	train_set = df[1:132]
	test_set = df[132:]

	# plot_pacf(train_set['Volume'],lags=60)
	# plot_acf(train_set['Volume'],lags=60)

	transf_df_data = train_set.copy()

	transf_df_data['Volume'] = np.log(train_set['Volume'])
	transf_df_data['Volume'] = transf_df_data['Volume'].diff()
	transf_df_data = transf_df_data.drop(transf_df_data.index[0])

	# my_order = (1,0,0)
	# my_seasonal_order = (1, 0, 1, 4)

	model = SARIMAX(train_set['Volume'], order=my_order, seasonal_order=my_seasonal_order)
	model_fit = model.fit()

	# print(model_fit.summary())

	predictions = model_fit.forecast(len(test_set))
	predictions = pd.Series(predictions, index=test_set.index)
	# residuals = test_set['Volume'] - predictions

	points_to_display = 100

	plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
	# plt.xlim([points_to_display,df.size-points_to_display])
	plt.xlim([df['Date'][0],df['Date'][df['Date'].size-1]])
	plt.plot(df['Date'], df['Volume'])
	plt.plot(test_set['Date'], predictions)
	plt.title('Quarterly Rice Production Volume of Davao del Sur')
	plt.ylabel('Volume in Tons')
	plt.xlabel('Date')
	plt.xticks(rotation=45)
	plt.grid(True)
	
	graph = get_graph()
	return graph