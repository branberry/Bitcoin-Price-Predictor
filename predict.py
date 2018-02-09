from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# initializing lists to hold varying daily price values for bitcoin
dataframeX = pd.read_csv('smallset.csv',usecols=[5,6,7,8])
dataframeY = pd.read_csv('smallsetY.csv',usecols=[8])
dataframeXEval = pd.read_csv('data/trainsetY.csv',usecols=[8])

X = dataframeX.as_matrix()
Y = dataframeY.as_matrix()

# Charting data
#sns.lmplot('index','close', data=dataframeX.reset_index(),fit_reg=False)
#plt.show()

# creating the training model
model = Sequential()
model.add(Dense(1,input_shape=(4,)))
model.add(Activation('linear'))

sgd = SGD(0.01)

model.compile(loss='msle',optimizer=sgd,
             metrics=['msle'])

model.fit(X,Y,nb_epoch=10)

#score = model.evaluate(x_test, y_test, batch_size=32)