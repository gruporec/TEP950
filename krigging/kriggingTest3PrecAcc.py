import sys
import time
import sklearn.datasets as skdata
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.optimize as opt
import scipy.sparse as sp
import qpsolvers as qp
import isadoralib as isl


# # carga el dataset de iris de sklearn
#dataset = skdata.load_iris()
# carga el dataset de dígitos de sklearn
#dataset = skdata.load_digits()
# # carga el dataset de cancer de sklearn
dataset = skdata.load_breast_cancer()
# # carga el dataset de vinos de sklearn
#dataset = skdata.load_wine()

# selecciona el valor alpha para krigging
alph = 100

# numero de repeticiones
nrep = 10

test_size = 0.7

balanced=False


# #INFO
# print("Dataset: ", dataset['DESCR'])
# print("Número de muestras: ", dataset['data'].shape[0])
# print("Número de características: ", dataset['data'].shape[1])
# print("Número de clases: ", np.unique(dataset['target']).shape[0])
# print("Nombre de las clases: ", dataset['target_names'])
# print("Proporción de clases: ", np.unique(dataset['target'], return_counts=True)[1]/dataset['target'].shape[0])

#KRIGGING

#crea una lista con la precisión de cada repetición
precisiones = np.zeros(nrep)

#crea otra lista para probar el método del valor de la función objetivo
precisiones_fun = np.zeros(nrep)


X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.001)#, random_state=42
# obtiene el numero de clases
num_classes = len(np.unique(y_train))

# inicializa la matriz de confusión
conf = np.zeros([num_classes, num_classes+1])
# crea otra matriz de confusión para el método del valor de la función objetivo
conf_fun = np.zeros([num_classes, num_classes+1])

# crea una lista para guardar los tiempos de ejecución del clasificador lambda
tiempos_lambda = []
# crea una lista para guardar los tiempos de ejecución del clasificador función objetivo
tiempos_fun = []

for rep in range(nrep):

    # separa los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=test_size)#, random_state=42

    # si balanced es True, balancea los datos de entrenamiento
    if balanced:
        #obtiene el número de muestras de cada clase en los datos de entrenamiento
        ntrain = np.zeros(num_classes)
        for i in range(num_classes):
            ntrain[i] = np.where(y_train == i)[0].shape[0]
        # calcula el número de muestras de la clase minoritaria
        nmin = int(np.min(ntrain))
        # por cada clase
        for i in range(num_classes):
            # si el número de muestras de la clase i es mayor que nmin
            if ntrain[i] > nmin:
                #selecciona ntrain[i] - nmin muestras de la clase i para eliminar
                idx = np.random.choice(np.where(y_train == i)[0], int(ntrain[i] - nmin), replace=False)
                # elimina las muestras seleccionadas
                X_train = np.delete(X_train, idx, axis=0)
                y_train = np.delete(y_train, idx, axis=0)


    # método del vector lambda
    # marca el tiempo de inicio
    start_time = time.time()
    # crea un clasificador de krigging
    kr_lambda = isl.KriggingClassifier(X_train.T, alph, y_train)
    # kr_lambda.minTraining()
    # crea un vector de predicciones
    y_pred_lambda_ts = np.zeros(X_test.shape[0])

    # para cada muestra de test
    for i in range(X_test.shape[0]):
        # aplica el clasificador
        y_pred_lambda = kr_lambda.lambda_classifier(X_test[i])
        y_pred_lambda_ts[i] = y_pred_lambda

    # calcula la matriz de confusión

    for i in range(y_test.shape[0]):
        conf[int(y_test[i]), int(y_pred_lambda_ts[i])] += 1

    # calcula la precisión
    acc = np.sum(np.diag(conf))/np.sum(conf)

    # añade la precisión a la lista de precisiones
    precisiones[rep] = acc

    # marca el tiempo de finalización
    end_time = time.time()
    # calcula el tiempo de ejecución
    tiempo = end_time - start_time
    # añade el tiempo a la lista de tiempos
    tiempos_lambda.append(tiempo)



    # método del valor de la función objetivo
    # marca el tiempo de inicio
    start_time = time.time()
    # crea un clasificador de krigging
    kr_fun = isl.KriggingFunctionClassifier(X_train.T, alph, y_train)
    # crea un vector de predicciones
    y_pred_fun_ts = np.zeros(X_test.shape[0])

    # para cada muestra de test
    for i in range(X_test.shape[0]):
        # aplica el clasificador
        y_pred_fun = kr_fun.fun_classifier(X_test[i])
        y_pred_fun_ts[i] = y_pred_fun

    # calcula la matriz de confusión
    for i in range(y_test.shape[0]):
        conf_fun[int(y_test[i]), int(y_pred_fun_ts[i])] += 1
    # print('Matriz de confusión: \n', conf_fun)

    # calcula la precisión
    acc_fun = np.sum(np.diag(conf_fun))/np.sum(conf_fun)
    # print('Precisión: ', acc_fun)

    # añade la precisión a la lista de precisiones
    precisiones_fun[rep] = acc_fun

    # marca el tiempo de finalización
    end_time = time.time()
    # calcula el tiempo de ejecución
    tiempo = end_time - start_time
    # añade el tiempo a la lista de tiempos
    tiempos_fun.append(tiempo)

print('Método del vector lambda:')
# calcula la media de las precisiones
print('Precisión media: ', np.mean(precisiones))
print('Desviación típica: ', np.std(precisiones))
print('conf: \n', conf)

# normaliza la matriz de confusión
confn = conf/conf.sum(axis=1)[:, np.newaxis]

# muestra la matriz de confusión normalizada con 3 decimales
print('Matriz de confusión normalizada: \n', np.round(confn, 3))
print('Tiempo medio de ejecución: ', np.mean(tiempos_lambda))

print('\n\nMétodo del valor de la función objetivo:')
# calcula la media de las precisiones
print('Precisión media: ', np.mean(precisiones_fun))
print('Desviación típica: ', np.std(precisiones_fun))
print('conf_fun: \n', conf_fun)

# normaliza la matriz de confusión
confn_fun = conf_fun/conf_fun.sum(axis=1)[:, np.newaxis]

# muestra la matriz de confusión normalizada con 3 decimales
print('Matriz de confusión normalizada: \n', np.round(confn_fun, 3))
print('Tiempo medio de ejecución: ', np.mean(tiempos_fun))