from django.db import models
from django.core.validators import MinValueValidator

class SARIMAModel(models.Model):
    dataset = models.TextField(null=True)
    p_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    d_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    q_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    sp_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    sd_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    sq_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    m_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    bic = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    rmse = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mse = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mape = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mad = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    is_boxcox = models.BooleanField(default=False)
    lmbda = models.DecimalField(max_digits=20, decimal_places=6, blank=True, null=True, default=0)
    
    order = models.SmallIntegerField(validators=[MinValueValidator(0)],default=0)
    forecasts = models.JSONField(null=True)
    graph = models.ImageField(null=True)

    def get_order_str(self):
        return "(" + str(self.p_param) + "," + str(self.d_param) + "," + str(self.q_param) + ")"

    def get_seasonal_str(self):
        return "(" +  str(self.sp_param) + "," +  str(self.sd_param) + "," +  str(self.sq_param) + ")"

    def get_shorthand_str(self):
        shorthand = "SARIMA" + self.get_order_str() + self.get_seasonal_str()
        return shorthand

    def __str__(self):
        order = self.get_order_str()
        seasonal = self.get_seasonal_str() + str(self.m_param)
        
        addons = ""
        addons += "BC " if self.is_boxcox else ""
        addons += str(self.lmbda) if self.is_boxcox else ""

        return self.dataset + " SARIMA" + order + seasonal + addons

class BayesianSARIMAModel(models.Model):
    dataset = models.TextField(null=True)
    p_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    d_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    q_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    sp_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    sd_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    sq_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    m_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    bic = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    rmse = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mse = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mape = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mad = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    is_boxcox = models.BooleanField(default=False)
    lmbda = models.DecimalField(max_digits=20, decimal_places=6, blank=True, null=True, default=0)
    
    order = models.SmallIntegerField(validators=[MinValueValidator(0)],default=0)
    forecasts = models.JSONField(null=True)
    graph = models.ImageField(null=True)

    def get_order_str(self):
        return "(" + str(self.p_param) + "," + str(self.d_param) + "," + str(self.q_param) + ")"

    def get_seasonal_str(self):
        return "(" +  str(self.sp_param) + "," +  str(self.sd_param) + "," +  str(self.sq_param) + ")"

    def get_shorthand_str(self):
        shorthand = "Bayesian SARIMA" + self.get_order_str() + self.get_seasonal_str()
        return shorthand

    def __str__(self):
        order = self.get_order_str()
        seasonal = self.get_seasonal_str() + str(self.m_param)
        
        addons = ""
        addons += "BC " if self.is_boxcox else ""
        addons += str(self.lmbda) if self.is_boxcox else ""

        return self.dataset + " Bayesian SARIMA" + order + seasonal + addons
    
class HoltWintersModel(models.Model):
    ADDITIVE = "add"
    MULTIPLICATIVE = "mul"
    METHOD_CHOICE = [
        (ADDITIVE, "Additive"),
        (MULTIPLICATIVE, "Multiplicative"),
    ]

    dataset = models.TextField(null=True)
    trend = models.CharField(null=False,choices=METHOD_CHOICE,default=ADDITIVE,max_length=3)
    seasonal = models.CharField(null=False,choices=METHOD_CHOICE,default=ADDITIVE,max_length=3)
    damped = models.BooleanField(default=False)
    rmse = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mse = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mape = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mad = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    is_boxcox = models.BooleanField(default=False)
    lmbda = models.DecimalField(max_digits=20, decimal_places=6, blank=True, null=True, default=0)

    forecasts = models.JSONField(null=True)
    graph = models.ImageField(null=True)

    def get_shorthand_str(self):
        shorthand = "Holt-Winters Model"
        return shorthand

    def __str__(self):
        formatted = self.dataset + " Holt-Winters Model" 
        return formatted
    
class LSTMModel(models.Model):
    dataset = models.TextField(null=True)
    n_inputs = models.SmallIntegerField(validators=[MinValueValidator(0)])
    n_epochs = models.SmallIntegerField(validators=[MinValueValidator(0)])
    n_units = models.SmallIntegerField(validators=[MinValueValidator(0)])
    rmse = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mse = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mape = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mad = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    is_boxcox = models.BooleanField(default=False)
    lmbda = models.DecimalField(max_digits=20, decimal_places=6, blank=True, null=True, default=0)

    forecasts = models.JSONField(null=True)
    graph = models.ImageField(null=True)

    def get_shorthand_str(self):
        shorthand = "LSTM Model"
        return shorthand

    def __str__(self):
        formatted = self.dataset + " LSTM Model" 
        return formatted