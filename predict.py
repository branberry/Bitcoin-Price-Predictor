from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd
import seaborn as sns

# initializing lists to hold varying daily price values for bitcoin
dataframeX = pd.read_csv('smallset.csv',usecols=[5,6,7,8])
dataframeY = pd.read_csv('smallsetY.csv',usecols=[8])
print(dataframeX.head())
print(dataframeY.head())

# creating the training model
model = Sequential()