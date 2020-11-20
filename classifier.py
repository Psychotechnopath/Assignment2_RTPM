#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import imblearn as imb
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
#%%

df = pd.read_csv("pre_processed.csv", index_col= 0, parse_dates=[0])
df_y = df['payment_class']
df_x = df.drop(columns=['payment_class'])

#%%
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.4, random_state=47, shuffle=False)

#%%
# Initialize a random over sampler with ratio that will be tested
over = imb.over_sampling.RandomOverSampler()
# Initialize a pipeline (One can add extra steps here if required)
steps = [('o', over)]
pipeline = imb.pipeline.Pipeline(steps)
X_train, y_train = pipeline.fit_resample(X_train, y_train)

#%%
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
score2 = rf.score(X_test, y_test)

#%%
xgb = XGBClassifier(objective='binary:logistic', max_depth=10, n_jobs=-1, random_state=47)
xgb.fit(X_train, y_train)
score3 = xgb.score(X_test, y_test)

#%%
from scipy.stats import uniform
import random
C = uniform(loc=0, scale=4)
print(C)

parameters = dict(n_estimators = [i for i in range(50, 151)],
                  criterion = ['gini', 'entropy'],
                  min_samples_split = [i for i in range(1, 6)])
rcv = RandomizedSearchCV(rf, parameters, random_state=47)
rcv.fit(X_train, y_train)
print(rcv.best_params_)
#%%
