import sys
import time
import sklearn.datasets as skdata
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.optimize as opt
import scipy.sparse as sp
import qpsolvers as qp


# # carga el dataset de iris de sklearn
# dataset = skdata.load_iris()
# carga el dataset de dígitos de sklearn
#dataset = skdata.load_digits()
# # carga el dataset de cancer de sklearn
# dataset = skdata.load_breast_cancer()
# # carga el dataset de vinos de sklearn
dataset = skdata.load_wine()

#guarda el tiempo de inicio
start_time = time.time()

# selecciona el valor alpha para krigging
alph = 0.0
# selecciona el valor beta para krigging
beta = 1e10
# separa los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.5, random_state=42)

#KRIGGING

class KriggingClassifier:
    '''Objeto para clasificación mediante Krigging. La matriz Xtrain contiene la base de datos de entrenamiento, con una fila por cada caractrística y una columna por cada muestra.'''
    # Constructor
    def __init__(self, Xtrain, alpha, beta):
        self.Xtrain = Xtrain
        self.alpha = alpha
        self.beta = beta
        
        self.update_matrices(Xtrain, alpha, beta)

    def update_matrices(self, Xtrain, alpha, beta):
        '''Actualiza las matrices P, q, G, h y A para los datos de entrenamiento Xtrain, el parámetro alpha y el parámetro beta.'''
        # Calcula la matriz P definida como una matriz cuadrada de tamaño 2*N+1 con una matriz diagonal de tamaño N y valor 2 en la esquina superior izquierda
        self.P = np.zeros([2*Xtrain.shape[1]+1, 2*Xtrain.shape[1]+1])
        self.P[:Xtrain.shape[1], :Xtrain.shape[1]] = np.eye(Xtrain.shape[1])*2
        # Convierte P en una matriz dispersa
        self.P = sp.csc_matrix(self.P)

        # Calcula el vector q de tamaño 2*N+1 con valores beta en los primeros N elementos, alpha en los N siguientes y beta en el último
        self.q = np.zeros([2*Xtrain.shape[1]+1])
        self.q[:Xtrain.shape[1]] = beta
        self.q[Xtrain.shape[1]:2*Xtrain.shape[1]] = alpha
        self.q[2*Xtrain.shape[1]] = beta

        # Calcula la matriz G de tamaño 2*N+1 x 2*N con cuatro matrices identidad de tamaño N y una columna de ceros. Todas las matrices identidad tienen signo negativo excepto la esquina superior derecha
        self.G = np.zeros([2*Xtrain.shape[1], 2*Xtrain.shape[1]+1])
        self.G[:Xtrain.shape[1], :Xtrain.shape[1]] = -np.eye(Xtrain.shape[1])
        self.G[Xtrain.shape[1]:2*Xtrain.shape[1], Xtrain.shape[1]:2*Xtrain.shape[1]] = -np.eye(Xtrain.shape[1])
        self.G[Xtrain.shape[1]:2*Xtrain.shape[1], :Xtrain.shape[1]] = -np.eye(Xtrain.shape[1])
        self.G[:Xtrain.shape[1], Xtrain.shape[1]:2*Xtrain.shape[1]] = np.eye(Xtrain.shape[1])
        # Convierte G en una matriz dispersa
        self.G = sp.csc_matrix(self.G)

        # Calcula el vector h de tamaño 2*N con valores cero
        self.h = np.zeros([2*Xtrain.shape[1]])

        # Calcula la matriz A que contiene la matriz de datos de entrenamiento y N+1 columnas de 0 en la parte superior y una fila de 0 y un 1 en la parte inferior
        self.A = np.zeros([Xtrain.shape[0]+1, 2*Xtrain.shape[1]+1])
        self.A[:Xtrain.shape[0], :Xtrain.shape[1]] = Xtrain
        self.A[Xtrain.shape[0], 2*Xtrain.shape[1]] = 1
        # Convierte A en una matriz dispersa
        self.A = sp.csc_matrix(self.A)
        

    def apply(self,x):
        '''Aplica el clasificador a un vector de características x. Devuelve el valor de la función objetivo y el vector de lambdas de los datos de entrenamiento.'''
        # Crea un vector de características ampliado con un 1
        b = np.hstack([x, 1])

        # Calcula el T que minimiza la función Qp usando OSQP
        T = qp.solve_qp(self.P, self.q.T, self.G, self.h, self.A, b, solver='osqp')

        P= self.P.toarray()

        # Obtiene el valor de la función objetivo
        f = 0.5*np.dot(T[:, np.newaxis].T, np.dot(P, T)) + np.dot(self.q, T)

        # Obtiene el vector de lambdas de los datos de entrenamiento
        lambda_i = T[:self.Xtrain.shape[1]]


        return (f, lambda_i)

# PREDICCION KRIGGING
# testeo sobre el conjunto de entrenamiento
# método del vector lambda
# crea un clasificador de krigging
kr_lambda_tr = KriggingClassifier(X_train.T, alph, beta)
# crea un vector de predicciones
y_pred_lambda_tr = np.zeros(X_train.shape[0])

# Crea una lista con los valores de los índices de cada clase con el tamaño del número de clases en el conjunto de entrenamiento
indices = [None]*len(np.unique(y_train))
for i in range(len(np.unique(y_train))):
    indices[i] = np.where(y_train == i)

# para cada muestra de test
for i in range(X_train.shape[0]):
    # aplica el clasificador
    y_pred_lambda = kr_lambda_tr.apply(X_train[i])[1]
    # separa los valores de lambda por clases
    y_pred_lambda = [y_pred_lambda[indices[j]] for j in range(len(np.unique(y_train)))]
    # calcula la suma de los valores de lambda por clases
    y_pred_lambda = [np.sum(y_pred_lambda[j]) for j in range(len(np.unique(y_train)))]
    # selecciona la clase con mayor valor de lambda
    y_pred_lambda = np.argmax(y_pred_lambda)
    # añade la predicción a la lista de predicciones
    y_pred_lambda_tr[i] = y_pred_lambda

# obtiene el numero de clases
num_classes = len(np.unique(y_train))
# calcula la matriz de confusión
conf = np.zeros([num_classes, num_classes])

for i in range(y_train.shape[0]):
    conf[int(y_train[i]), int(y_pred_lambda_tr[i])] += 1
print('Matriz de confusión: \n', conf)

#normaliza la matriz de confusión
confn = conf/conf.sum(axis=1)[:, np.newaxis]
print('Matriz de confusión normalizada: \n', confn)

# calcula la precisión
acc = np.sum(np.diag(conf))/np.sum(conf)
print('Precisión: ', acc)

# testeo sobre el conjunto de test
# método del vector lambda
# crea un clasificador de krigging
kr_lambda = KriggingClassifier(X_train.T, alph, beta)
# crea un vector de predicciones
y_pred_lambda_ts = np.zeros(X_test.shape[0])

# para cada muestra de test
for i in range(X_test.shape[0]):
    # aplica el clasificador
    y_pred_lambda = kr_lambda.apply(X_test[i])[1]
    # separa los valores de lambda por clases
    y_pred_lambda = [y_pred_lambda[indices[j]] for j in range(len(np.unique(y_train)))]
    # calcula la suma de los valores de lambda por clases
    y_pred_lambda = [np.sum(y_pred_lambda[j]) for j in range(len(np.unique(y_train)))]
    # selecciona la clase con mayor valor de lambda
    y_pred_lambda = np.argmax(y_pred_lambda)
    # añade la predicción a la lista de predicciones
    y_pred_lambda_ts[i] = y_pred_lambda

# obtiene el numero de clases
num_classes = len(np.unique(y_train))
# calcula la matriz de confusión
conf = np.zeros([num_classes, num_classes])

for i in range(y_test.shape[0]):
    conf[int(y_test[i]), int(y_pred_lambda_ts[i])] += 1
print('Matriz de confusión: \n', conf)

#normaliza la matriz de confusión
confn = conf/conf.sum(axis=1)[:, np.newaxis]
print('Matriz de confusión normalizada: \n', confn)

# calcula la precisión
acc = np.sum(np.diag(conf))/np.sum(conf)
print('Precisión: ', acc)

