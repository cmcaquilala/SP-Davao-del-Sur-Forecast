from django.db import models
from django.core.validators import MinValueValidator

class SARIMAModel(models.Model):
    p_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    d_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    q_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    sp_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    sd_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    sq_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    m_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    rmse = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mse = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mape = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mad = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    order = models.SmallIntegerField(validators=[MinValueValidator(0)],default=0)
    graph = models.ImageField(null=True)

    def __str__(self):
        formatted = "SARIMA(" + str(self.p_param) + ", " + str(self.d_param) + ", " + str(self.q_param) + ")(" +  str(self.sp_param) + ", " +  str(self.sd_param) + ", " +  str(self.sq_param) + ")" + str(self.m_param)
        return formatted

class BayesianARMAModel(models.Model):
    p_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    q_param = models.SmallIntegerField(validators=[MinValueValidator(0)])
    rmse = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mse = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mape = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    mad = models.DecimalField(max_digits=20, decimal_places=4,default=0)
    order = models.SmallIntegerField(validators=[MinValueValidator(0)],default=0)
    graph = models.ImageField(null=True)

    def __str__(self):
        formatted = "Bayesian ARMA(" + str(self.p_param) + ',' + str(self.q_param) + ")" 
        return formatted