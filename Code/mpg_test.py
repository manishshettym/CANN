import numpy as np
import pandas 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

df = pandas.read_csv("/home/pallavi/SEM_5/AA/Project/CANN/datasets/auto-mpg.csv")

features = list(["cylinders", "displacement", "horsepower", "weight", "acceleration"])
y = df["mpg"]
X = df[features]
X['horsepower']=pandas.to_numeric(X['horsepower'], errors='coerce').fillna(0, downcast='infer')


def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(5, input_dim=5, kernel_initializer='normal', activation='linear'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


estimator = KerasRegressor(build_fn=baseline_model, epochs=1000, verbose=0)


t0 = time.clock()
estimator.fit(X,y)
t1 = time.clock()

prediction = estimator.predict(X)

train_error =  np.abs(y - prediction)
mean_error = np.mean(train_error)
min_error = np.min(train_error)
max_error = np.max(train_error)
std_error = np.std(train_error)



#print('prediction :',prediction)
#print('train error :')
#print(train_error)
print('rmse : ', rmse(np.array(prediction),np.array(y)))
#print('mean error :',mean_error)
print('time taken ',t1-t0)