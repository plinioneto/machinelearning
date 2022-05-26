from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split #separa os dados em treino e teste baseado na porcentagem fornecida
from sklearn import neighbors #k-nearest-neighbors
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler
from math import sqrt 

# ------------------------------------------------------------------------------------------------------------------------------------

dados = pd.read_csv('train.csv')
dados.head()

i = 1

#NO BANCO DE DADOS EXISTEM DADOS FALTANDO, PORTANTO ELES DEVEM SER ADICIONADOS
dados.isnull().sum()

mean = dados['Item_Weight'].mean() #Dados de 'Item_Weight' com a média
dados['Item_Weight'].fillna(mean,inplace=True)

mode = dados['Outlet_Size'].mode() #Dados de 'Outlet_Size' com a moda
dados['Outlet_Size'].fillna(mode[0],inplace=True)

# LIDAR COM VARIÁVEIS CATEGÓRICAS E DESCARTAR COLUNAS DE ID
dados.drop(['Item_Identifier','Outlet_Identifier'], axis = 1, inplace = True)
dados = pd.get_dummies(dados)


#features = dados[:,0:3]
#target = dados[:,3]

#CONJUNTOS DE TRAIN E TESTE
train , test = train_test_split(dados, test_size = 0.3)

#x_train = train[:,0:3]
#y_train = train[:,3]
x_train = train.drop('Item_Outlet_Sales', axis = 1)
y_train = train['Item_Outlet_Sales']

#x_test = test[:,0:3]
#y_test = test[:,3]
x_test = test.drop('Item_Outlet_Sales', axis = 1)
y_test = test['Item_Outlet_Sales']

#Pré-processamento -> Redimensionamento dos recursos
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)

rmse_val = [] #Vetor que vai guardar os valores da taxa de erro para diferentes K

for K in range(0,20):
    K = K + 1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(x_train,y_train)
    pred = model.predict(x_test) #make prediction on test set
    error = sqrt(mse(y_test,pred)) #calculate rmse

    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)

#plotting the rmse values against k values
curve = plt.plot(np.array(rmse_val)) #elbow curve 
plt.show()

minval = rmse_val.index(min(rmse_val))

#Avaliar o menor K
print("Avaliando para ", minval, "vizinhos")
model = neighbors.KNeighborsRegressor(n_neighbors=minval)
model.fit(x_train,y_train)
pred = model.predict(x_test)
plt.scatter(y_test,pred)
plt.show()

#linhha