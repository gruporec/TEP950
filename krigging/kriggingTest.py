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
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.66)

# ENTRENAMIENTO KRIGGING
class Krigging_classifier:
    """Krigging-based classifier class."""
    def __init__(self,Xtrain,Ytrain):
        """Class constructor.

        Parameters
        ----------
        Xtrain : 2D numpy array containing the training database. Each row corresponds to a sample, and each column to a feature.
        Ytrain : 1D numpy array containing the labels of the training database"""
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.K = np.linalg.pinv(np.hstack([Xtrain, np.ones([Xtrain.shape[0], 1])]).T)
        self.classes = np.unique(Ytrain)
        self.num_classes = self.classes.shape[0]
    
    def predictScores(self,Xpred):
        """Predicts the scores for the samples in Xpred, for each trained class.
        
        Parameters
        ----------
        Xpred : 2D numpy array containing the samples to be classified. Each row corresponds to a sample, and each column to a feature.
        
        Returns
        -------
        scores : 2D numpy array containing the scores for each class and sample. Each row corresponds to a sample, and each column to a class."""
        scores = np.zeros([self.num_classes, Xpred.shape[0]])
        for i in range(Xpred.shape[0]):
            x_i = np.hstack([Xpred[i], 1])
            lambda_i = np.dot(self.K,x_i)
            for j in range(self.num_classes):
                idx = np.where(self.Ytrain==self.classes[j])
                scores[i,j] = np.sum(lambda_i[idx])
        return scores
    
    def predict(self,Xpred):
        """Predicts the class of the samples in Xpred.
        
        Parameters
        ----------
        Xpred : 2D numpy array containing the samples to be classified. Each row corresponds to a sample, and each column to a feature.
        
        Returns
        -------
        Ypred : 1D numpy array containing the predicted class for each sample."""
        scores = self.predictScores(Xpred)
        Ypred = np.argmax(scores,axis=1)
        return Ypred
    
krg=Krigging_classifier(X_train,y_train)

# crea una matriz de numpy de datos ampliada con una columna de unos y la transpone
X_train_ext = np.hstack([X_train, np.ones([X_train.shape[0], 1])]).T
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
    lambda_i = np.dot(X_train_pinv, x_i)

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
    lambda_i = np.dot(X_train_pinv, x_i)

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