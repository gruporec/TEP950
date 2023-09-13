import sklearn.datasets as skdata
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# # carga el dataset de iris de sklearn
# dataset = skdata.load_iris()
# carga el dataset de dígitos de sklearn
dataset = skdata.load_digits()
# # carga el dataset de cancer de sklearn
# dataset = skdata.load_breast_cancer()
# # carga el dataset de vinos de sklearn
# dataset = skdata.load_wine()

# separa los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target)#, test_size=0.66

# ENTRENAMIENTO KRIGGING
# crea una matriz de numpy de datos ampliada con una columna de unos
X_train_ext = np.hstack([X_train, np.ones([X_train.shape[0], 1])])
# calcula la pseudo-inversa
X_train_pinv = np.linalg.pinv(X_train_ext)

# PREDICCION KRIGGING
# Sobre entrenamiento

# Crea un array de predicciones vacío
y_train_pred = np.zeros(y_train.shape)

# Obtiene las clases únicas
classes = np.unique(y_train)
# Obtiene el número de clases
num_classes = classes.shape[0]

# Crea un array de scores vacío
scores_train = np.zeros([num_classes, X_train.shape[0]])

# Para cada dato de entrenamiento original
for i in range(X_train.shape[0]):
    # Crea un vector de características ampliado con un 1
    x_i = np.hstack([X_train[i], 1])
    # Multiplica el vector de características ampliado por la pseudo-inversa para obtener el vector lambda
    lambda_i = np.dot(x_i,X_train_pinv)

    # Para cada clase
    for j in range(num_classes):
        # Obtiene los índices correspondientes a la clase j en el conjunto de entrenamiento
        idx = np.where(y_train==classes[j])

        # Obtiene el score de la clase j para el dato de entrenamiento i sumando los lambdas de los datos de la clase j
        scores_train[j, i] = np.sum(lambda_i[idx])

    # guarda la clase con mayor score para el dato de entrenamiento i en el array de predicciones
    y_train_pred[i] = np.argmax(scores_train[:, i])

# calcula la precisión del modelo sobre el conjunto de entrenamiento
acc = np.sum(y_train_pred==y_train)/y_train.shape[0]

# calcula la matriz de confusión sobre el conjunto de entrenamiento
conf = np.zeros([num_classes, num_classes])
for i in range(y_train.shape[0]):
    conf[int(y_train[i]), int(y_train_pred[i])] += 1

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
y_test_pred = np.zeros(y_test.shape)

# Obtiene las clases únicas
classes = np.unique(y_train)
# Obtiene el número de clases
num_classes = classes.shape[0]

# Crea un array de scores vacío
scores_test = np.zeros([num_classes, X_test.shape[0]])

# Para cada dato de prueba
for i in range(X_test.shape[0]):
    # Crea un vector de características ampliado con un 1
    x_i = np.hstack([X_test[i], 1])
    # Multiplica el vector de características ampliado por la pseudo-inversa para obtener el vector lambda
    lambda_i = np.dot(x_i,X_train_pinv)

    # Para cada clase
    for j in range(num_classes):
        # Obtiene los índices correspondientes a la clase j en el conjunto de entrenamiento
        idx = np.where(y_train==classes[j])

        # Obtiene el score de la clase j para el dato de prueba i sumando los lambdas de los datos de la clase j
        scores_test[j, i] = np.sum(lambda_i[idx])

    # guarda la clase con mayor score para el dato de prueba i en el array de predicciones
    y_test_pred[i] = np.argmax(scores_test[:, i])

# calcula la precisión del modelo sobre el conjunto de prueba
acc = np.sum(y_test_pred==y_test)/y_test.shape[0]
# calcula la matriz de confusión sobre el conjunto de prueba
conf = np.zeros([num_classes, num_classes])
for i in range(y_test.shape[0]):
    conf[int(y_test[i]), int(y_test_pred[i])] += 1

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

# obtiene los índices de los datos de entrenamiento que no se han predicho correctamente
idx_train = np.where(y_train_pred!=y_train)[0]
# obtiene las características de los datos de entrenamiento que no se han predicho correctamente
train_fail = X_train[idx_train]
# obtiene la clase real de los datos de entrenamiento que no se han predicho correctamente
train_fail_class = y_train[idx_train]
# obtiene la clase predicha de los datos de entrenamiento que no se han predicho correctamente
train_fail_pred = y_train_pred[idx_train]

# obtiene los índices de los datos de prueba que no se han predicho correctamente
idx_test = np.where(y_test_pred!=y_test)[0]
# obtiene las características de los datos de prueba que no se han predicho correctamente
test_fail = X_test[idx_test]
# obtiene la clase real de los datos de prueba que no se han predicho correctamente
test_fail_class = y_test[idx_test]
# obtiene la clase predicha de los datos de prueba que no se han predicho correctamente
test_fail_pred = y_test_pred[idx_test]

#obtiene el número de datos que no se han predicho correctamente en el conjunto de entrenamiento
num_fail_train = train_fail.shape[0]
#obtiene el número de datos que no se han predicho correctamente en el conjunto de prueba
num_fail_test = test_fail.shape[0]

#Para los datos de entrenamiento
#obtiene el numero total de datos de entrenamiento
num_train = X_train.shape[0]
# calcula el cuadrado más cercano al numero total de datos
n = np.ceil(np.sqrt(num_train))
m=n

# calcula si se van a quedar filas vacías
while n*m > num_train+n-1:
    m = m-1
# crea una figura con un tamaño de m x n
fig, ax = plt.subplots(int(m), int(n), figsize=(10, 10))
#elimina los ejes en todos los subplots
for i in range(int(m)):
    for j in range(int(n)):
        ax[i, j].axis('off')
#para los datos de entrenamiento
for i in range(num_train):
    #extrae las características del dato
    x_0 = X_train[i]
    # convierte la imagen a 8x8
    x_0 = np.reshape(x_0, [8, 8])
    # si el dato no se ha predicho correctamente
    if i in idx_train:
        # pinta la imagen en la posición correspondiente en rojo
        ax[int(i/n), int(i%n)].imshow(x_0, cmap='Reds')
    # si el dato se ha predicho correctamente
    else:
        # pinta la imagen en la posición correspondiente en azul
        ax[int(i/n), int(i%n)].imshow(x_0, cmap='Blues')

#Para los datos de prueba
#obtiene el numero total de datos de prueba
num_test = X_test.shape[0]
# calcula el cuadrado más cercano al numero total de datos
n = np.ceil(np.sqrt(num_test))
m=n

# calcula si se van a quedar filas vacías
while n*m > num_test+n-1:
    m = m-1
# crea una figura con un tamaño de m x n
fig, ax = plt.subplots(int(m), int(n), figsize=(10, 10))
#elimina los ejes en todos los subplots
for i in range(int(m)):
    for j in range(int(n)):
        ax[i, j].axis('off')
#para los datos de prueba
for i in range(num_test):
    #extrae las características del dato
    x_0 = X_test[i]
    # convierte la imagen a 8x8
    x_0 = np.reshape(x_0, [8, 8])
    # si el dato no se ha predicho correctamente
    if i in idx_test:
        # pinta la imagen en la posición correspondiente en rojo
        ax[int(i/n), int(i%n)].imshow(x_0, cmap='Reds')
    # si el dato se ha predicho correctamente
    else:
        # pinta la imagen en la posición correspondiente en azul
        ax[int(i/n), int(i%n)].imshow(x_0, cmap='Blues')
        

# muestra la figura
plt.show()