import pandas as pd 
import numpy as np 
import sys
import warnings
import itertools
warnings.filterwarnings("ignore")
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
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

def parser(x):
    return datetime.strptime(x, '%d-%m-%Y')


loaded = SARIMAXResults.load('model1.pkl')
test = pd.read_csv('validation_data.csv', header=0, parse_dates=[1], index_col=1,date_parser=parser)
series = pd.read_csv('trenddata.csv', header=0, parse_dates=[1], index_col=1,date_parser=parser)
print(test.index)
test=test[['Sales']]
test = test[:22]
print(test)
train = series
predictions = loaded.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
print(predictions.shape)
print(test.shape)
print(test.shape)
rmse = np.sqrt(mean_squared_error(test['Sales'], predictions))
rmse = round(rmse, 3)
tempResultsDf = pd.DataFrame({'Method':['Seasonal Autoregressive Integrated Moving Average'], 'RMSE': [rmse] })
print(tempResultsDf)
plt.plot(test,color = 'blue')
plt.savefig('test_prod_test'+mm+'.jpg')
plt.show()
plt.plot(predictions, color='red')
plt.savefig('test_prod_pred'+mm+'.jpg')
plt.show()
