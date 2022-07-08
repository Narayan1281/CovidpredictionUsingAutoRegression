# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 00:09:58 2021
@author: rn
"""
## Name: Rustam Narayan  ;  Roll No.: B20128
## Phone No.: 8603861159
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg as AR
from sklearn.metrics import mean_squared_error
import math

covid_df = pd.read_csv('daily_covid_cases.csv')

#################################################
###################### Q1 #######################

## Part(a)
fig = plt.figure()
ax1 = plt.subplot(1,1,1)
xticks = np.arange(16,617,60)
xlabel = ['Feb-20', 'Apr-20', 'Jun-20', 'Aug-20', 'Oct-20', 'Dec-20', 'Feb-21', 'Apr-21', 'Jun-21', 'Aug-21', 'Oct-21']
plt.plot(covid_df['new_cases'])
plt.xticks(xticks,xlabel)
plt.xlabel('Month-Year',fontsize=13)
plt.ylabel('New confirmed cases',fontsize=13)
plt.title('Lineplot--Q1a',fontsize=15)
plt.show()

## Part(b)
print('\nQ1:part(b)\n')
## Generating time sequence with a one-day lag
lag_1 = covid_df['new_cases'].shift(1)
print(f"The Pearson correlation (autocorrelation) coefficient between the generated one-day lag time sequence and the given time sequence = {covid_df['new_cases'].corr(lag_1) :.4f}")

## Part(c)
## scatter plot between the given time sequence and one-day lagged
fig = plt.figure()
ax2 = plt.subplot(1,1,1)
plt.scatter(covid_df['new_cases'][1:],lag_1[1:])
plt.xlabel('Original time sequence',fontsize=13)
plt.ylabel('One day lagged time sequence',fontsize=13)
plt.title('One day lagged sequence vs Original',fontsize=15)
plt.show()

## Part(d)
print('\nQ1:part(d)\n')

def correlationWithLag(series1,lag):
    '''takes original time series and lag-value as input and gives lagged time sequence'''
    lagged = series1.shift(lag)
    corr = series1[lag: ].corr(lagged[lag: ])
    print(f'Lag-{lag} correlation = {corr :.4f}')
    return corr

lag_value = [1,2,3,4,5,6]
corr_y = []
for i in lag_value:
    corr_y.append(correlationWithLag(covid_df['new_cases'],i))

fig = plt.figure()
ax3 = plt.subplot(1,1,1)
plt.plot(lag_value,corr_y)
plt.xlabel('Lagged-Value',fontsize=13)
plt.ylabel('Correlation Coefficient',fontsize=13)
plt.title('Correlation coefficient vs lag value',fontsize=15)
plt.show()    

## Part(e)

sm.graphics.tsa.plot_acf(covid_df['new_cases'], lags=lag_value)
plt.xlabel('Lagged-Value',fontsize=13)
plt.ylabel('Correlation Coefficient',fontsize=13)
plt.title('Correlation coefficient vs lag value',fontsize=15)
plt.show()    

#################################################
###################### Q2 #######################

## Part(a)
print('\nQ2:part(a)\n')

# Train test split
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size) ## math.ceil gives the smallest integer greater than or equal to the passed arguments
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

window = 5 # The lag=5
model = AR(train, lags=window,old_names=True) 
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model 
print(f'The coeficients of trained AR model is \n{coef}')
#using these coefficients walk forward over time steps in test, one step each time

## Part(b)
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1] # Add other values
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.

## Part(b)-(i)
## scatter plot between actual and predicted values
fig = plt.figure()
ax4 = plt.subplot(1,1,1)
plt.scatter(test,predictions)
plt.xlabel('Actual Value',fontsize=13)
plt.ylabel('Predicted Value',fontsize=13)
plt.title('Actual vs Predicted Values',fontsize=15)
plt.show()

## Part(b)-(ii)
## line plot showing actual and predicted test values
fig = plt.figure()
ax5 = plt.subplot(1,1,1)
plt.plot(test,label='Actual Value')
plt.plot(predictions,label='Predicted Value')
plt.xlabel('Days',fontsize=13)
plt.ylabel('New covid cases',fontsize=13)
plt.title('Actual vs Predicted Values',fontsize=15)
plt.legend()
plt.show()


## Part(b)-(iii)
print('\nQ2:part(b)-(iii)\n')
## Compute RMSE (%) and MAPE between actual and predicted test data
rmse = (np.sqrt(mean_squared_error(test,predictions))/np.mean(test))*100
mape = (np.sum(abs(test-predictions)/test)/len(test))*100
print(f'RMSE(%) b/w actual and predicted values is {rmse:.4f}')
print(f'MAPE b/w actual and predicted values is {mape:.4f}')


#################################################
###################### Q3 #######################
print('\nQ3:\n')

lag_list = [1, 5, 10, 15, 25]
rmse_list = []
mape_list = []
for i in lag_list:
    window = i # The lag=i
    model = AR(train, lags=window,old_names=True) 
    model_fit = model.fit() # fit/train the model
    coef = model_fit.params # Get the coefficients of AR model 
    # print(f'The coeficients of trained AR model is \n{coef}')
    #using these coefficients walk forward over time steps in test, one step each time
    
    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list() # List to hold the predictions, 1 step at a time
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1] # Add other values
        obs = test[t]
        predictions.append(yhat) #Append predictions to compute RMSE later
        history.append(obs) # Append actual test value to history, to be used in next step.
    rmse_i = (np.sqrt(mean_squared_error(test,predictions))/np.mean(test))*100
    mape_i = (np.sum(abs(test-predictions)/test)/len(test))*100
    rmse_list.append(rmse_i)
    mape_list.append(mape_i)

print(f'RMSE(%) of repective lagged series is \n{mape_list}')
print(f'MAPE of repective lagged series is \n{rmse_list}')
## bar chart showing RMSE (%) on the y-axis and lagged values on the x-axis
fig = plt.figure()
ax6 = plt.subplot(1,1,1)
plt.bar([1,2,3,4,5],rmse_list)
plt.xticks([1,2,3,4,5],lag_list)
plt.xlabel('Lag-value',fontsize=13)
plt.ylabel('RMSE(%)',fontsize=13)
plt.title('Lag-values vs RMSE(%)',fontsize=15)
plt.show()

## a bar chart showing MAPE on the y-axis and lagged values on the x-axis
fig = plt.figure()
ax7 = plt.subplot(1,1,1)
plt.bar([1,2,3,4,5],mape_list)
plt.xticks([1,2,3,4,5],lag_list)
plt.xlabel('Lag-value',fontsize=13)
plt.ylabel('MAPE',fontsize=13)
plt.title('Lag-values vs MAPE',fontsize=15)
plt.show()

#################################################
###################### Q4 #######################
print('\nQ4:\n')
for p in range(1,len(train)):
    corr = np.corrcoef(train[p:].ravel(), train[:len(train)-p].ravel())[0,1]
    if (abs(corr) <= 2/np.sqrt(len(train[p:]))):
      print('The heuristic value for the optimal number of lags is',p-1)
      break

heuristic_value = p - 1
window = heuristic_value # The lag
model = AR(train, lags=window,old_names=True) 
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model 
# print(f'The coeficients of trained AR model is \n{coef}')
#using these coefficients walk forward over time steps in test, one step each time

history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1] # Add other values
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.
rmse_q4 = (np.sqrt(mean_squared_error(test,predictions))/np.mean(test))*100
mape_q4 = (np.sum(abs(test-predictions)/test)/len(test))*100

print(f'\nRMSE(%) b/w actual and predicted values is {rmse_q4:.4f}')
print(f'MAPE b/w actual and predicted values is {mape_q4:.4f}\n\n')




