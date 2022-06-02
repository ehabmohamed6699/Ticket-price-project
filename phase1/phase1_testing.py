import pickle
import numpy as np
import pandas as pd
import preprocessing
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('airline-test-samples.csv')
data = preprocessing.preprocess(data)

x_test = data[['airline', 'num_code', 'time_taken', 'type']]
y_test = data['price']

#Leaner
pickled_model = pickle.load(open('len_model.sav', 'rb'))
y_pred = pickled_model.predict(x_test)

true_value=np.asarray(y_test)[0]
predicted_value=y_pred[0]
print('Mean Square Error to multiple linear regression', metrics.mean_squared_error(y_pred, y_test))
print('True value in the test set in millions is poly_model: ' + str(true_value))
print('Predicted value in the test set in millions is : ' + str(predicted_value))
#Polynomial
poly = PolynomialFeatures(degree=7)
x_test = poly.fit_transform(x_test)
pickled_model = pickle.load(open('poly_model.sav', 'rb'))
y_pred = pickled_model.predict(x_test)

true_value=np.asarray(y_test)[0]
predicted_value=y_pred[0]
print('Mean Square Error to multiple linear regression', metrics.mean_squared_error(y_pred, y_test))
print('True value in the test set in millions is poly_model: ' + str(true_value))
print('Predicted value in the test set in millions is : ' + str(predicted_value))