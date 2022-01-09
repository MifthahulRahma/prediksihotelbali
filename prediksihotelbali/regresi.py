#library
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

#dataset 
dataset=pd.read_csv("hotel.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state=0)

#call model regression
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

#save model
filename = 'model.sav'
joblib.dump(model, filename)

#load model
loaded_model = joblib.load(filename)

#prediction model
loaded_model.predict(20)