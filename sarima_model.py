
#import all the neccesary libraries
import pandas as pd 
import numpy as np 
import sys
import warnings
import itertools
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot as plt
from pandas.plotting import lag_plot
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas import datetime
import calendar
import seaborn as sns
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def parser(x):
    """
        this method parses the string into DateTime format d-m-Y
    """
    return datetime.strptime(x, '%d-%m-%Y')


#read validation data
test = pd.read_csv('validation_data.csv', header=0, parse_dates=[1], index_col=1,date_parser=parser)

#read trenddata for training
series = pd.read_csv('trenddata.csv', header=0, parse_dates=[1], index_col=1,date_parser=parser)

#select the product range
prange = series[series.Productcat=='ProdCat2'].shape[0]
pvrange = test[test.Productcat == 'ProdCat2'].shape[0]

#remove names as now not needed
series=series[['Sales']]
test=test[['Sales']]

print('range of product',prange)

p1series=series[169:prange+168]
pv1test=test[21:pvrange+20]

#model = ARIMA(np.asarray(p1series), order=(5,1,0))
#this gives RSME about 1200 for prodCat1 so eliminated as SARIMA performs much better

#use SARIMA Model for forecasting the product data Sales as it has very low RMSE than simple ARIMA

#find the best combination of p q d and seasonal parameters sp,sq,sd,se to feed to model with minimum AIC

p = range(0, 2)
d = range(0, 2)
q = range(0, 2)

pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[1], 12) for x in list(itertools.product(p, d, q))]
best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None
temp_model = None
ok =True
for param in pdq:   
    for param_seasonal in seasonal_pdq: 
        temp_model = SARIMAX(p1series,order=param,seasonal_order = param_seasonal,enforce_invertibility=False,
                            enforce_stationarity=False)
        results = temp_model.fit(disp=False)
        if results.aic < best_aic :
            best_aic = results.aic
            best_pdq = param
            best_seasonal_pdq = param_seasonal
print("Best ARIMA {} x {} model - AIC:{}".format(best_pdq,best_seasonal_pdq,best_aic))

#feed the best parameters obtained to sarima
p1, d1, q1 = best_pdq
a1,b1,c1,d11 = best_seasonal_pdq
model = SARIMAX(p1series,order=(p1,d1,q1),seasonal_order=(a1,b1,c1,d11),enforce_invertibility=False,
                    enforce_stationarity=False)    
model_fit = model.fit()
print(model_fit.summary())

# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()


#predict on validation data
predictions = model_fit.predict(start=len(p1series), end=len(p1series)+len(pv1test)-1, dynamic=False)

#calculate rmse

rmse = np.sqrt(mean_squared_error(pv1test['Sales'], predictions))
rmse = round(rmse, 3)
abs_error = np.abs(pv1test['Sales']-predictions)

tempResultsDf = pd.DataFrame({'Method':['Seasonal Autoregressive Integrated Moving Average'], 'RMSE': [rmse]})

#print RMSE and MAPE
print(tempResultsDf)


#plot training data and forcast for validation
#validate prediction
plt.plot(pv1test.index,pv1test.Sales,'b')
plt.plot(pv1test.index,predictions,'r--')
plt.show();
