#importing the dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import mean_squared_error

#reading the data
data = pd.read_csv('https://github.com/OmololaOkebiorun/Boston_House_Price_prediction/blob/main/boston.csv')

data.head()

df = data.copy()

data.info()

data.describe()

__Data Understanding__

__CRIM: per capita crime rate by town__

__ZN: proportion of residential land zoned for lots over 25,000 sq.ft.__

__INDUS: proportion of non-retail business acres per town__

__CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)__

__NOX: nitric oxides concentration (parts per 10 million)__

__RM: average number of rooms per dwelling__

__AGE: proportion of owner-occupied units built prior to 1940__

__DIS: weighted distances to ﬁve Boston employment centers__

__RAD: index of accessibility to radial highways__

__TAX: full-value property-tax rate per 10,000__

__PTRATIO: pupil-teacher ratio by town__ 

__B: 1000(Bk−0.63)**2 where Bk is the proportion of blacks by town __

__LSTAT: % lower status of the population__

__MEDV: Median value of owner-occupied homes in 1000s__

__Feature Selection__

#distribution of values in each column
data.hist(figsize = (10,8));

#checking the Pearson Correlation coefficient
plt.figure(figsize=(10,6));

sns.heatmap(data.corr(), annot = True, cmap = 'Blues')
plt.title('Corellation between scores', fontsize = 8)

#scatterplot of features by target('MEDV')
fig, axs = plt.subplots(nrows = 2, ncols = 6, figsize = (16,8))
cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX','PTRATIO', 'B', 'LSTAT']
for col, ax in zip(cols, axs.flat):
    sns.regplot(x = data[col], y = data['MEDV'], color = 'blue', ax = ax)

It observed that 'TAX' and 'RAD' are highly correlated (0.91), hence 'RAD' will be dropped. Also, much noise in noticed in 'CHAS'; 'B',  'ZN', hence, they will be dropped.

__Modelling__ 

X = df.drop(['MEDV', 'RAD', 'CHAS','ZN', 'B'], axis = 1)
y = df.MEDV

#feature scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#splitting into test and train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#creating an instance of RandomForestRegressor
model = RandomForestRegressor()

#setting parameters for and grid searching
parm = {"max_depth":[2, 3, 4, 5], "min_samples_split":[1, 3,5], "min_samples_leaf": [1, 2, 3] }
gs_model = GridSearchCV(model, parm, cv=5)
gs_model.fit(X_train, y_train)

#getting the best parameters
gs_model.best_params_

gs_model.best_score_

rfr = RandomForestRegressor(max_depth=5, min_samples_split=5,min_samples_leaf=2,  n_estimators=30, random_state=42)

#fitting the model 
rfr.fit(X_train, y_train)

#getting the score on test data
rfr.score(X_test, y_test)

#getting the score on training data
rfr.score(X_train, y_train)

#predicting X_test
predY = rfr.predict(X_test)

#getting the mean_squared_error
mse = mean_squared_error(y_test, predY)
np.sqrt(mse)

#saving the model
#model_name = 'price_predictor.sav'
#pickle.dump(rfr, open(model_name, 'wb'))

