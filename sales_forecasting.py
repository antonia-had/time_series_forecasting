'''This code was written for instructional purposes by Antonia Hadjimichael 
during Fall 2018 for CEE5930 at Cornell University. '''

import numpy as np #Package we'll use for numerical calculations
import matplotlib.pyplot as plt #From matplotlib package we import pyplot for plots
import pandas #Package to data manipulation
import scipy.optimize #Package we'll use to optimize
plt.style.use('seaborn-colorblind') #This is a pyplot style (optional)

'''Load the data into a dataframe (DF) with the name wine_sales'''
time_series = pandas.Series.from_csv("wine_sales.csv", header=0)

P=12 #number of seasonal periods in a cycle

'''In this example, the functions are written as they apply to a continous time
series. They are the same when applied to one's training data series. For the 
purposes of validation, you should divide data to a training set and a validation 
set. Then, should you choose to apply the functions listed here, you should 
apply the functions for the training set, extract forecasts and then use those 
to initialize your validation period. To divide the data, you would do something
like this:
training = time_series[0:108] # Up to December '88
validation = time_series[108:] # From January '89 until end
'''

#==============================================================================
'''MODEL DEVELOPMENT'''
#==============================================================================
''' Below, we are defining all models we talked about during the forecasting 
chapter. The Naive, Simple Moving Average, and Weighted Moving Average models 
are defined first for demonstration purposes but we won't be using them in this 
example. The models are then defined in this order: Simple Exponential Smoothing,
Double Exponential Smoothing, Additive Seasonal, Multiplicative Seasonal, 
Additive Holt Winters, and Multiplicative Holt Winters.'''

'''NAIVE MODEL 
Using this model, y_hat(t+1)=y(t) (i.e., the predicted next value is equal to 
the last observed value). We first need to create an array in our forecasts DF 
to store our forecast values and initialize the first value.'''
def naive():
    y_hat=pandas.Series().reindex_like(time_series)
    y_hat[0]= time_series[0] # Initialize forecasting array with first observation
    ''' Loop through every month using the model to forecast y'''
    #This sets a range for the index to loop through
    for t in range(len(y_hat)-1): 
        y_hat[t+1]= time_series[t] # Apply model to forecast time i+1
    return y_hat

'''SIMPLE MOVING AVERAGE
Using this model, y_hat(t+1)=(y(t)+y(t-1)...+y(t-k+1))/k (i.e., the predicted 
next value is equal to the average of the last k observed values). We first 
need to create an array to store our forecast values.'''
def SMA(params):
    k=int(np.array(params))
    y_hat=pandas.Series().reindex_like(time_series)
    y_hat[0:k]=time_series[0:k]
    ''' Loop through every month using the model to forecast y.
    Be careful with Python indexing!'''
    for t in range(k-1,len(y_hat)-1): #This sets a range for the index to loop through
        y_hat[t+1]= np.sum(time_series[t-k+1:t+1])/k # Apply model to forecast time i+1
    return y_hat

'''WEIGHTED MOVING AVERAGE
Using this model, y_hat(t+1)=w(1)*y(t)+w(2)*y(t-1)...+w(k)*y(t-k+1) (i.e., the 
predicted next value is equal to the weighted average of the last k observed 
values). We first need to create an array to store our forecast values.'''
def WMA(params):
    weights = np.array(params)
    k=len(weights)
    y_hat=pandas.Series().reindex_like(time_series)
    y_hat[0:k]=time_series[0:k] # Initialize values
    ''' Loop through every month using the model to forecast y.
    Be careful with Python indexing!'''
    for t in range(k-1,len(y_hat)-1): #This sets a range for the index to loop through
        y_hat[t+1]= np.sum(time_series[t-k+1:t+1].multiply(weights)) # Apply model to forecast time i+1
    return y_hat
'''This model includes the constraint that all our weights should sum to one. 
To include this in our optimization, we need to define it as a function of our 
weights.'''
def WMAcon(params):
    weights = np.array(params)
    return np.sum(weights)-1

'''SINGLE EXPONENTIAL SMOOTHING
Using this model, y_hat(t+1)=y_hat(t)+a*(y(t)-y_hat(t))(i.e., the 
predicted next value is equal to the weighted average of the last forecasted value and its
difference from the observed). We first need to create an array to store our forecast values.'''
def SES(params):
    a = np.array(params)
    y_hat=pandas.Series().reindex_like(time_series)
    y_hat[0]=time_series[0] # Initialize values
    ''' Loop through every month using the model to forecast y.
    Be careful with Python indexing!'''
    for t in range(len(y_hat)-1): #This sets a range for the index to loop through
        y_hat[t+1]=  y_hat[t]+a*(time_series[t]-y_hat[t])# Apply model to forecast time i+1
    return y_hat

'''DOUBLE EXPONENTIAL SMOOTHING (Holts Method)
Using this model, y_hat(t+1)=E(t)+T(t) (i.e., the 
predicted next value is equal to the expected level of the time series plus the 
trend). We first need to create an array to store our forecast values.'''
def DES(params):
    a,b = np.array(params)
    y_hat=pandas.Series().reindex_like(time_series)
    '''We need to create series to store our E and T values.'''
    E = pandas.Series().reindex_like(time_series)
    T = pandas.Series().reindex_like(time_series)
    y_hat[0]=E[0]=time_series[0] # Initialize values
    T[0]=0
    ''' Loop through every month using the model to forecast y.
    Be careful with Python indexing!'''
    for t in range(len(y_hat)-1): #This sets a range for the index to loop through
        E[t+1] = a*time_series[t]+(1-a)*(E[t]+T[t])
        T[t+1] = b*(E[t+1]-E[t])+(1-b)*T[t]
        y_hat[t+1] = E[t] + T[t] # Apply model to forecast time i+1
    return y_hat

'''ADDITIVE SEASONAL
Using this model, y_hat(t+1)=E(t)+S(t-p) (i.e., the 
predicted next value is equal to the expected level of the time series plus the 
appropriate seasonal factor). We first need to create an array to store our 
forecast values.'''
def ASM(params):
    a,b = np.array(params)[:2]
    p = int(params[2])
    y_hat=pandas.Series().reindex_like(time_series)
    '''We need to create series to store our E and S values.'''
    E = pandas.Series().reindex_like(time_series)
    S = pandas.Series().reindex_like(time_series)
    y_hat[:p]=time_series[0] # Initialize values
    '''We need to initialize the first p number of E and S values'''
    E[:p] = np.sum(time_series[:p])/p
    S[:p] = time_series[:p]-E[:p]
    ''' Loop through every month using the model to forecast y.
    Be careful with Python indexing!'''
    for t in range(p-1, len(y_hat)-1): #This sets a range for the index to loop through
        E[t+1] = a*(time_series[t]-S[t+1-p])+(1-a)*E[t]
        S[t+1] = b*(time_series[t]-E[t])+(1-b)*S[t+1-p]
        y_hat[t+1] = E[t] + S[t+1-p] # Apply model to forecast time i+1
    return y_hat

'''MULTIPLICATIVE SEASONAL
Using this model, y_hat(t+1)=E(t)*S(t-p) (i.e., the 
predicted next value is equal to the expected level of the time series times 
the appropriate seasonal factor). We first need to create an array to store our 
forecast values.'''
def MSM(params):
    a,b = np.array(params)[:2]
    p = int(params[2])
    y_hat=pandas.Series().reindex_like(time_series)
    '''We need to create series to store our E and S values.'''
    E = pandas.Series().reindex_like(time_series)
    S = pandas.Series().reindex_like(time_series)
    y_hat[:p]=time_series[0] # Initialize values
    '''We need to initialize the first p number of E and S values'''
    E[:p] = np.sum(time_series[:p])/p
    S[:p] = time_series[:p]/E[:p]
    ''' Loop through every month using the model to forecast y.
    Be careful with Python indexing!'''
    for t in range(p-1, len(y_hat)-1): #This sets a range for the index to loop through
        E[t+1] = a*(time_series[t]/S[t+1-p])+(1-a)*E[t]
        S[t+1] = b*(time_series[t]/E[t])+(1-b)*S[t+1-p]
        y_hat[t+1] = E[t]*S[t+1-p] # Apply model to forecast time i+1
    return y_hat

'''ADDITIVE HOLT-WINTERS METHOD
Using this model, y_hat(t+1)=(E(t)+T(t))*S(t-p) (i.e., the 
predicted next value is equal to the expected level of the time series plus the 
trend, times the appropriate seasonal factor). We first need to create an array 
to store our forecast values.'''
def AHW(params):
    a, b, g = np.array(params)[:3]
    p = int(params[3])
    y_hat=pandas.Series().reindex_like(time_series)
    '''We need to create series to store our E and S values.'''
    E = pandas.Series().reindex_like(time_series)
    S = pandas.Series().reindex_like(time_series)
    T = pandas.Series().reindex_like(time_series)
    y_hat[:p]=time_series[0] # Initialize values
    '''We need to initialize the first p number of E and S values'''
    E[:p] = np.sum(time_series[:p])/p
    S[:p] = time_series[:p]-E[:p]
    T[:p] = 0
    ''' Loop through every month using the model to forecast y.
    Be careful with Python indexing!'''
    for t in range(p-1, len(y_hat)-1): #This sets a range for the index to loop through
        E[t+1] = a*(time_series[t]-S[t+1-p])+(1-a)*(E[t]+T[t])
        T[t+1] = b*(E[t+1]-E[t])+(1-b)*T[t]
        S[t+1] = g*(time_series[t]-E[t])+(1-g)*S[t+1-p]
        y_hat[t+1] = E[t]+T[t]+S[t+1-p] # Apply model to forecast time i+1
    return y_hat

'''MUTLIPLICATIVE HOLT-WINTERS METHOD
Using this model, y_hat(t+1)=(E(t)+T(t))*S(t-p) (i.e., the 
predicted next value is equal to the expected level of the time series plus the 
trend, times the appropriate seasonal factor). We first need to create an array 
to store our forecast values.'''
def MHW(params):
    a, b, g = np.array(params)[:3]
    p = int(params[3])
    y_hat=pandas.Series().reindex_like(time_series)
    '''We need to create series to store our E and S values.'''
    E = pandas.Series().reindex_like(time_series)
    S = pandas.Series().reindex_like(time_series)
    T = pandas.Series().reindex_like(time_series)
    y_hat[:p]=time_series[0] # Initialize values
    '''We need to initialize the first p number of E and S values'''
    S[:p] = time_series[:p]/(np.sum(time_series[:p])/p)
    E[:p] = time_series[:p]/S[:p]
    T[:p] = 0
    ''' Loop through every month using the model to forecast y.
    Be careful with Python indexing!'''
    for t in range(p-1, len(y_hat)-1): #This sets a range for the index to loop through
        E[t+1] = a*(time_series[t]/S[t+1-p])+(1-a)*(E[t]+T[t])
        T[t+1] = b*(E[t+1]-E[t])+(1-b)*T[t]
        S[t+1] = g*(time_series[t]/E[t])+(1-g)*S[t+1-p]
        y_hat[t+1] = (E[t]+T[t])*S[t+1-p] # Apply model to forecast time i+1
    return y_hat

'''FORECAST EVALUATION'''
'''Define the error metric(s) that are going to be used as functions. In this 
example I will use MSE to evaluate my models.'''
def MSE(params, args):
    model, = args
    t_error = np.zeros(len(time_series))
    forecast = model(params)
    for t in range(len(time_series)):
        t_error[t] = time_series[t]-forecast[t]
    MSE = np.mean(np.square(t_error))
    return MSE

#==============================================================================
'''OPTIMIZATION'''
#==============================================================================
''' List of all the models we will be optimizing'''
models = [SES, DES, ASM, MSM, AHW, MHW]
''' This is a list of all the default parameters for the models we will be 
optimizing. P is the number of periods in a cycle.'''
                      #SES,  DES,     ASM
default_parameters = [[0.5],[0.5,0.5],[0.5,0.5,P],
                      #MSM,        AHW,            MHW
                      [0.5,0.5,P],[0.5,0.5,0.5,P],[0.5,0.5,0.5,P]]
''' This is a list of all the bounds for the default parameters we will be 
optimizing. All the a,b,g's are weights between 0 and 1. We don't need to 
optimize our P parameter (number of periods) so its bounds are the value 
itself.'''
bounds = [[(0,1)],[(0,1)]*2, [(0,1)]*2 + [(P,P)],
           [(0,1)]*2 + [(P,P)],[(0,1)]*3 + [(P,P)],[(0,1)]*3 + [(P,P)]]

''' We can now write a loop that will go through all of the possible models and 
attempt to minimize their MSE. To store the minimized MSE values as well as the 
parameter values that produce them, we can create an array to store the MSEs 
and a list to store the parameter values for each model.'''
min_MSEs = np.zeros(len(models)) # Array to store minimized MSEs
opt_params = [None]*len(models) # Empty list to store optim. parameters
for i in range(len(models)):
    '''This is the scipy optimization function we will be using. It produces a 
    distionary item that contains the minimized MSE value (under 'fun'), the 
    parameters that produce it (under 'x') and other information.'''
    res = scipy.optimize.minimize(MSE, # Function we're minimizing (MSE in this case)
                                  default_parameters[i], # Default parameters to use
                                  # Additional arguments that the optimizer 
                                  #won't be changing (model and data type)
                                  args=[models[i]], 
                                  method='L-BFGS-B', # Optimization method to use
                                  bounds=bounds[i]) # Parameter bounds
    min_MSEs[i] = res['fun'] #Store minimized MSE value
    opt_params[i] = res['x'] #Store parameter values identified by optimizer   
'''Note: For the WMA model, the weights should sum to 1 and this should be 
input to our optimization as a constraint. To do so, we need to define the 
constraint function as a dictionary and include the following in our 
minimization call: constraints=[{'type':'eq','fun': WMAcon}]. The number of 
periods to consider cannot be optimized by this type of optimizer.'''
    
#==============================================================================
'''REPORT RESULTS'''
#==============================================================================
''' We will first create a figure with our observations and all our forecasts.'''
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) # Create figure
ax.set_title("Australian wine sales (kilolitres)") # Set figure title
l1 = ax.plot(time_series, color='black', linewidth=3.0, label='Observations') # Plot observations
for i in range(len(models)):
    ax.plot(time_series.index,models[i](opt_params[i]), label = models[i].__name__)
ax.legend() # Activate figure legend
plt.show()
'''We can also produce an MSE report for all the models.'''
print('The estimated MSE for all the models are:')
for i in range(len(models)):
    print(models[i].__name__ +': '+str(min_MSEs[i]))
