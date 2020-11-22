# %%
# Imports
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from skopt import BayesSearchCV

# %%


########## SETTING UP ##########



# %%
# Retrieve pre-processed dataset
data_df = pd.read_csv('Ex3_Preprocessing.csv', index_col= 0, parse_dates=[0])

# %%
# Split into only UD1/UD2, and paired
data_df_UD1 = data_df[data_df['payment_class'] == 1] #UD1
data_df_UD2 = data_df[data_df['payment_class'] == 2] #UD2
data_df_UD12 = data_df[(data_df['payment_class'] == 1) | (data_df['payment_class'] == 2)] #UD1 and UD2

# Fix class imbalance
data_df_UD1 = data_df_UD1.sample(n=int(len(data_df_UD2)/4), replace=True) # dangerously much oversampling, almost 10x to get to 1:4 split to UD2
data_df_UD12 = pd.concat([data_df_UD1, data_df_UD2])

# %%
# Pick dataset to use (can be put into a function if i feel like it later)
used_dataset = data_df_UD12

# Split into X and y
y = used_dataset['Case duration']
X = used_dataset.drop(['Case duration', 'Complete Timestamp'], axis=1)

# %%
# Split into train-test sets
## Can't use future cases to predict past cases, so cuts of time-ordered matrix are used instead of random sampling for splitting
train_size = 0.7 #size of train split

X_train = X.iloc[:int(len(X)*train_size)]
X_test = X.iloc[int(len(X)*train_size):]
y_train = y.iloc[:int(len(y)*train_size)]
y_test = y.iloc[int(len(y)*train_size):]

# %%


########## MODELS ##########



# %%
# Linear regression
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_test, y_test) #-0.008610657493041352 (SUPER CRAP)

# %%
# Polynomial linear regression
## Data prep
poly_degree = 3
poly_features = PolynomialFeatures(degree=poly_degree)
X_transf = poly_features.fit_transform(X_train)

## Define/train model
linreg = LinearRegression().fit(X_transf, y_train)

## Calc bias/variance
X_transf_test = poly_features.fit_transform(X_test)
y_hat = linreg.predict(X_transf_test)

rmse = np.sqrt(mean_squared_error(y_test, y_hat))
r2 = r2_score(y_test, y_hat)

print('RMSE: ', rmse) 
print('R2: ', r2)
# On Train set
# degree = 2: RMSE:  151.88032911446774, R2:  0.07431454449433705
# degree = 3: RMSE:  141.74281273540916, R2:  0.19376346556372537

# On Test set
# degree = 2: RMSE:  79048.30736670438, R2:  -249862.58715404916
# degree = 3: RMSE:  RMSE:  463542.07221424393, R2:  -8592035.958376516

# %%
# Random forest
## Define/train model
rand_forest = RandomForestRegressor(n_estimators=1000, random_state=0)
rand_forest.fit(X_train, y_train)

## Predictions
y_hat = rand_forest.predict(X_train)

rmse = np.sqrt(mean_squared_error(y_train, y_hat))
r2 = r2_score(y_train, y_hat)

print('RMSE: ', rmse) 
print('R2: ', r2)
# On Train set
# n=100: RMSE:  53.1157291602075, R2:  0.8753868889133495
# n=200: RMSE:  52.49448684165802, R2:  0.8782847956602003
# n=500: RMSE:  52.25716299019348, R2:  0.8793828395581254
# n=1000: RMSE:  52.01332488035817, R2:  0.8805058413053213

# On Test set
# n=100: RMSE:  165.85959787264403, R2:  -0.1000160879831089
# n=200: RMSE:  165.40026973575056, R2:  -0.0939318009490524
# n=500: RMSE:  164.837295107636, R2:  -0.08649762019097884
# n=1000: RMSE:  164.9340060555953, R2:  -0.08777290245197711

# Look at estimations for UD1 and UD2 seperately
# y_hat[blabla = 1] for UD1

# %%
# ADAboost regression
## Define/train model
ada_boost = AdaBoostRegressor(random_state=0, n_estimators=200)
ada_boost.fit(X_train, y_train)

## Predictions
y_hat = ada_boost.predict(X_train)

rmse = np.sqrt(mean_squared_error(y_train, y_hat))
r2 = r2_score(y_train, y_hat)

print('RMSE: ', rmse) 
print('R2: ', r2)
# On Train set
# n=100: RMSE:  157.0380917970301, R2:  -0.0892497328243611
# n=200: RMSE:  157.0380917970301, R2:  -0.0892497328243611

# On Test set
# n=100: RMSE:  167.23514438334834, R2:  -0.11833758169385544
# n=200: RMSE:  167.23514438334834, R2:  -0.11833758169385544


# %%
# Bagging regressor
## Define/train model
bagg_regr = BaggingRegressor(n_estimators=200, random_state=0)
bagg_regr.fit(X_train, y_train)

# Predictions
y_hat = bagg_regr.predict(X_train)

rmse = np.sqrt(mean_squared_error(y_train, y_hat))
r2 = r2_score(y_train, y_hat)

print('RMSE: ', rmse) 
print('R2: ', r2)
# On Train set
# n = 10: RMSE:  63.25071855095771, R2:  0.8232951903742393
# n = 100: RMSE:  53.12355810994833, R2:  0.8753501517096134
# n = 200: RMSE:  52.54485102807364, R2:  0.8780511319644563

# On Test set
# n = 10: RMSE:  170.82834497702, R2:  -0.16691087536150695
# n = 100: RMSE:  165.85154036170636, R2:  -0.09990921233067707
# n = 200: RMSE:  165.4933971927544, R2:  -0.09516400893693411

# %%
# Histogram-based Gradient Boosting Regression Tree
## Define/train model
hist_boost = HistGradientBoostingRegressor(max_iter = 10000, learning_rate = 0.00001, loss = 'least_squares')
hist_boost.fit(X_train, y_train)

# Predictions
y_hat = hist_boost.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_hat))
r2 = r2_score(y_test, y_hat)

print('RMSE: ', rmse) 
print('R2: ', r2)
# On Test set
# max_iter=100, lr=0.1: RMSE:  164.95847973452217, R2:  -0.08809574406033427
# max_iter=500, lr=0.1: RMSE:  175.5928128791024, R2:  -0.23290976591255608
# max_iter=100, lr=0.01: RMSE:  160.164493919987, R2:  -0.025770753165551108
# max_iter=500, lr=0.01: RMSE:  162.1181697268406, R2:  -0.05094794352731302
# max_iter = 1000, lr=0.0001: RMSE:  158.41835764485302, R2:  -0.0035264732961632905
# max_iter = 10000, lr=0.00001: RMSE:  158.41827324121243, R2:  -0.003525403959673934

# %%
# Kernal Ridge Regression
## Define/train model
kernel_ridge = KernelRidge(alpha=1.0, kernel='linear')
kernel_ridge.fit(X_train, y_train)

# Predictions
y_hat = kernel_ridge.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_hat))
r2 = r2_score(y_test, y_hat)

print('RMSE: ', rmse) 
print('R2: ', r2)
# On Test set
# kernel='linear': RMSE:  158.84130463530624, R2:  -0.008892077298348067
# kernel='additive_chi2': RMSE:  160.81389508624162, R2:  -0.03410577377260782
# kernel='poly': RMSE:  61783.46907040451, R2:  -152636.9062818268
# kernel='rbf': RMSE:  370.4147348694965, R2:  -4.486486681739983

# %%
# Kernal Ridge Regression (with hyperparameter tuning)
# define search space
params = dict()
params['alpha'] = (1e-6, 100.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['degree'] = (1,3)
params['kernel'] = ['linear', 'rbf', 'sigmoid']

# define evaluation
cv = KFold(n_splits=10, random_state=1)

# define the search
search = BayesSearchCV(estimator=KernelRidge(), search_spaces=params, n_jobs=-1, cv=cv)

# perform the search
search.fit(X_train, y_train)

# report the best result
print(search.best_score_)
print(search.best_params_)


# %%
# Support Vector regression
# define search space
params = dict()
params['C'] = (1e-6, 100.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['degree'] = (1,3)
params['kernel'] = ['linear', 'rbf', 'sigmoid']

# define evaluation
cv = KFold(n_splits=10, random_state=1)

# define the search
search = BayesSearchCV(estimator=SVR(), search_spaces=params, n_jobs=-1, cv=cv)

# perform the search
search.fit(X_train, y_train)

# report the best result
print(search.best_score_)
print(search.best_params_)


# %%


########## NAIVE BASELINE ##########


# %%
# Naive model (baseline)

def naive_predict(y):
    y_hat = [np.mean(y[0:i].values) for i in range(0,len(y))] 
    y_hat[0] = y[0:1].values[0] #otherwise first value is NaN, because of slice y[0:0]
    return(y_hat)

# Predictions
y_hat = naive_predict(y_test)

rmse = np.sqrt(mean_squared_error(y_test, y_hat))
r2 = r2_score(y_test, y_hat)

print('RMSE: ', rmse) 
print('R2: ', r2)
