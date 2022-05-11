import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split #separa os dados em treino e teste baseado na porcentagem fornecida
from sklearn import neighbors #k-nearest-neighbors
from sklearn.metrics import mean_squared_error as mse
from math import sqrt 

#-------------------------------------------------------------------------------------------------------------------------------

dados = np.loadtxt('train_orange_alterado.txt')

i = 1

features = dados[:,0:3]
target = dados[:,3]

train , test = train_test_split(dados, test_size = 0.3)

x_train = train[:,0:3]
y_train = train[:,3]

x_test = test[:,0:3]
y_test = test[:,3]

rmse_val = []
K = 3

model = neighbors.KNeighborsRegressor(n_neighbors = K)

model.fit(x_train,y_train)
pred = model.predict(x_test) #make prediction on test set
error = sqrt(mse(y_test,pred)) #calculate rmse

rmse_val.append(error) #store rmse values
print('RMSE value for k= ' , K , 'is:', error)

#plotting the rmse values against k values
curve = plt(rmse_val) #elbow curve 




