from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# initializing lists to hold varying daily price values for bitcoin
dataframeX = pd.read_csv('data/smallset.csv',usecols=[5,6,7,8])
dataframeY = pd.read_csv('data/smallsetY.csv',usecols=[8])
dataframeXEval = pd.read_csv('data/trainsetX.csv',usecols=[5,6,7,8])
dataframeYEval = pd.read_csv('data/trainsetY.csv',usecols=[8])
X = dataframeX.as_matrix()
Y = dataframeY.as_matrix()
X_test = dataframeXEval.as_matrix()
Y_test = dataframeYEval.as_matrix()
# Charting data
#sns.lmplot('index','close', data=dataframeX.reset_index(),fit_reg=False)
#plt.show()

# creating the training model
model = Sequential()
model.add(Dense(1,input_shape=(4,)))
model.add(Activation('relu'))

sgd = SGD(0.01)

model.compile(loss='mean_squared_logarithmic_error',optimizer='rmsprop',
             metrics=['sparse_categorical_accuracy','mean_squared_logarithmic_error'])

model.fit(X,Y,nb_epoch=30)

print(model.predict(X_test))
#score = model.evaluate(x_test, y_test, batch_size=32)