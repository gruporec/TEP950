import time
import sklearn.datasets as skdata
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.optimize as opt
import multiprocessing as mp


#KRIGGING
# Define la función a minimizar
def fun(A,alph):
    return np.dot(A,A.T)+alph*np.sum(np.abs(A))

# Define las constraints
def const(a,X_train,x):
    # amplia la matriz X_train con una columna de unos y la transpone
    X_train_ext = np.hstack([X_train, np.ones([X_train.shape[0], 1])]).T
    # amplia el vector x con un 1
    x_ext = np.hstack([x, 1])
    # calcula el vector XA-x
    XA_x = np.dot(X_train_ext, a) - x_ext
    # devuelve XA-x
    return XA_x

# Crea una función para realizar el bucle en multiprocesamiento
def Krigging(inputs):
    (X_train, X_test, y_train, alph, i, X_train_pinv, classes, num_classes, scores_test) = inputs
    # Extrae el vector de características del dato de prueba i
    x_i = X_test[i]

    # Crea un vector de características ampliado con un 1
    x_i_ext = np.hstack([X_test[i], 1])
    # Multiplica el vector de características ampliado por la pseudo-inversa para obtener el vector lambda de partida
    lambda_i = np.dot(X_train_pinv, x_i_ext)

    #calcula el vector lambda que minimiza la función fun sujeto a las constraints const
    lambda_i = opt.minimize(fun, lambda_i, args=(alph), constraints={'fun': const, 'type': 'eq', 'args': (X_train, x_i)}, tol=1e-6).x

    # Para cada clase
    for j in range(num_classes):
        # Obtiene los índices correspondientes a la clase j en el conjunto de entrenamiento
        idx = np.where(y_train==classes[j])

        # Obtiene el score de la clase j para el dato de prueba i sumando los lambdas de los datos de la clase j
        scores_test[j, i] = np.sum(lambda_i[idx])


    # devuelve la clase con mayor score para el dato de prueba i
    return classes[np.argmax(scores_test[:, i])]

# clase main
if __name__ == '__main__':
    threads = 12

    # # carga el dataset de iris de sklearn
    # dataset = skdata.load_iris()
    # carga el dataset de dígitos de sklearn
    dataset = skdata.load_digits()
    # # carga el dataset de cancer de sklearn
    # dataset = skdata.load_breast_cancer()
    # # carga el dataset de vinos de sklearn
    # dataset = skdata.load_wine()

    # guarda el tiempo de inicio
    start_time = time.time()
    # selecciona el valor alpha para krigging
    alph = 0.5
    # separa los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.66, random_state=42)

    # PREDICCION KRIGGING
    # Sobre entrenamiento

    # Crea un array de predicciones vacío
    #y_pred = np.zeros(y_train.shape)

    # Obtiene las clases únicas
    classes = np.unique(y_train)
    # Obtiene el número de clases
    num_classes = classes.shape[0]

    # Crea un array de scores vacío
    scores_train = np.zeros([num_classes, X_train.shape[0]])

    # crea una matriz de numpy de datos ampliada con una columna de unos y la transpone
    X_train_ext = np.hstack([X_train, np.ones([X_train.shape[0], 1])]).T
    # calcula la pseudo-inversa
    X_train_pinv = np.linalg.pinv(X_train_ext)

    
    print('Calculando predicciones...')

    # Crea una lista de inputs para los procesos
    inputs = [(X_train, X_train, y_train, alph, i, X_train_pinv, classes, num_classes, scores_train) for i in range(X_train.shape[0])]

    # Crea una pool de procesos
    pool = mp.Pool(processes=threads)

    print('Esperando a que terminen los procesos...')
    # Lanza los procesos
    y_pred = pool.map(Krigging,inputs)


    # Cierra la pool
    pool.close()


    # calcula la precisión del modelo sobre el conjunto de entrenamiento
    acc = np.sum(y_pred==y_train)/y_train.shape[0]

    # calcula la matriz de confusión sobre el conjunto de entrenamiento
    conf = np.zeros([num_classes, num_classes])
    for i in range(y_train.shape[0]):
        conf[int(y_train[i]), int(y_pred[i])] += 1

    # calcula la matriz de confusión normalizada sobre el conjunto de entrenamiento
    conf_n = conf/np.sum(conf, axis=1)[:, np.newaxis]
    # muestra la precisión y la matriz de confusión
    print('Accuracy: %.2f' % acc)
    print('Confusion matrix:')
    print(conf)
    # muestra la matriz de confusión normalizada con 2 decimales
    print('Normalized confusion matrix:')
    np.set_printoptions(precision=2)
    print(conf_n)

    # Sobre prueba

    # Crea un array de predicciones vacío
    #y_pred = np.zeros(y_test.shape)

    # Obtiene las clases únicas
    classes = np.unique(y_train)
    # Obtiene el número de clases
    num_classes = classes.shape[0]

    # Crea un array de scores vacío
    scores_test = np.zeros([num_classes, X_test.shape[0]])

    print('Calculando predicciones...')

    # Crea una lista de inputs para los procesos
    inputs = [(X_train, X_test, y_train, alph, i, X_train_pinv, classes, num_classes, scores_test) for i in range(X_test.shape[0])]

    # Crea una pool de procesos
    pool = mp.Pool(processes=threads)

    print('Esperando a que terminen los procesos...')
    # Lanza los procesos
    y_pred = pool.map(Krigging,inputs)


    # Cierra la pool
    pool.close()

    # calcula la precisión del modelo sobre el conjunto de prueba
    acc = np.sum(y_pred==y_test)/y_test.shape[0]
    # calcula la matriz de confusión sobre el conjunto de prueba
    conf = np.zeros([num_classes, num_classes])
    for i in range(y_test.shape[0]):
        conf[int(y_test[i]), int(y_pred[i])] += 1

    # calcula la matriz de confusión normalizada sobre el conjunto de prueba
    conf_n = conf/np.sum(conf, axis=1)[:, np.newaxis]

    # muestra la precisión y la matriz de confusión
    print('Accuracy: %.2f' % acc)
    print('Confusion matrix:')
    print(conf)
    # muestra la matriz de confusión normalizada con 2 decimales
    print('Normalized confusion matrix:')
    np.set_printoptions(precision=2)
    print(conf_n)

    # muestra el tiempo de ejecución
    print('Tiempo de ejecución: %.2f segundos' % (time.time() - start_time))