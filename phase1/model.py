import pandas as pd
import pickle
import preprocessing
import learn

data = pd.read_csv('airline-price-prediction.csv')

data = preprocessing.preprocess(data)

top_feature = learn.correlateData(data)

Y=data['price'] #Goal
X=data[top_feature]
X = X.drop(['price'], axis = 1)

X = learn.normalizeData(X)

leaner = learn.train_poly_model(X, Y, 7)

sln = learn.train_linear_model(X, Y)

filename1 = 'poly_model.sav'
pickle.dump(leaner, open(filename1, 'wb'))

filename2 = 'len_model.sav'
pickle.dump(sln, open(filename2, 'wb'))



