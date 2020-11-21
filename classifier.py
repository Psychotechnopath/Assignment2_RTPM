#%%
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import imblearn as imb
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import confusion_matrix

df = pd.read_csv("pre_processed.csv", index_col= 0, parse_dates=[0])
df_y = df['payment_class']
df_x = df.drop(columns=['payment_class'])


#%%
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.4, random_state=47, shuffle=False)

# Initialize a random over sampler with ratio that will be tested
over = imb.over_sampling.RandomOverSampler()
# Initialize a pipeline (One can add extra steps here if required)
steps = [('o', over)]
pipeline = imb.pipeline.Pipeline(steps)
X_train, y_train = pipeline.fit_resample(X_train, y_train)

rf = RandomForestClassifier()
parameters = dict(n_estimators = [i for i in range(50, 250)],
                  criterion = ['gini', 'entropy'])
rcv = RandomizedSearchCV(rf, parameters, random_state=47, verbose=2, n_jobs=-1, refit=True)
rcv.fit(X_train, y_train)
rf_randomsearch = rcv.score(X_test, y_test)

y_pred = rcv.predict(X_test)
confusion_matrix(y_test, y_pred)
rf_score = rcv.score(X_test, y_test)

with open('rf_randomsearch_fitted.pkl', 'wb') as f:
    pickle.dump(rcv, f)

#%%
xgb = XGBClassifier(objective='binary:logistic', max_depth=10, n_jobs=-1, random_state=47)
parameters2 = dict(max_depth = [i for i in range(1,25)])
rxgb = RandomizedSearchCV(xgb, parameters2, random_state=47, verbose=2, n_jobs=-1, refit=True)
rxgb.fit(X_train, y_train)
xgb_randomsearch = rxgb.score(X_test, y_test)

y_pred_2 = rxgb.predict(X_test)
confusion_matrix(y_test, y_pred_2)
rxgb.score(X_test, y_test)

with open('xgb_randomsearch_fitted.pkl', 'wb') as f2:
    pickle.dump(rxgb, f2)






#%%
print()


