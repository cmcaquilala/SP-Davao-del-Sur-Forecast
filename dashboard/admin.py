from django.contrib import admin
from .models import *

# Register your models here.
admin.site.register(SARIMAModel)
admin.site.register(BayesianSARIMAModel)
admin.site.register(HoltWintersModel)
admin.site.register(LSTMModel)