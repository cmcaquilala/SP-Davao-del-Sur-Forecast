from django import forms
from django.forms import ModelForm
from .models import *

class SARIMA_add_form(ModelForm):

    class Meta:
        model = SARIMAModel
        fields = ['p_param','d_param','q_param','sp_param','sd_param','sq_param','m_param','is_boxcox','lmbda']
        labels = {
            'p_param' : 'P',
            'd_param' : 'D',
            'q_param' : 'Q',
            'sp_param' : 'Seasonal P',
            'sd_param' : 'Seasonal D',
            'sq_param' : 'Seasonal Q',
            'm_param' : 'Season M',
            'is_boxcox' : 'Use Box Cox Transformation',
            'lmbda' : 'Lambda (default: auto)',
        }
        
class BayesianSARIMA_add_form(ModelForm):

    class Meta:
        model = BayesianSARIMAModel
        fields = ['p_param','d_param','q_param','sp_param','sd_param','sq_param','m_param','is_boxcox','lmbda']
        labels = {
            'p_param' : 'P',
            'd_param' : 'D',
            'q_param' : 'Q',
            'sp_param' : 'Seasonal P',
            'sd_param' : 'Seasonal D',
            'sq_param' : 'Seasonal Q',
            'm_param' : 'Season M',
            'is_boxcox' : 'Use Box Cox Transformation',
            'lmbda' : 'Lambda (default: auto)',
        }

class HoltWinters_add_form(ModelForm):

    class Meta:
        model = HoltWintersModel
        fields = ['is_boxcox','lmbda','trend','seasonal','damped']
        labels = {
            'is_boxcox' : 'Use Box Cox Transformation',
            'lmbda' : 'Lambda (default: auto)',
            'trend' : 'Trend Method',
            'seasonal' : 'Seasonal Method',
            'damped' : 'Use Damping',
        }

class LSTM_add_form(ModelForm):

    class Meta:
        model = LSTMModel
        fields = ['is_boxcox','lmbda']
        labels = {
            'is_boxcox' : 'Use Box Cox Transformation',
            'lmbda' : 'Lambda (default: auto)',
        }