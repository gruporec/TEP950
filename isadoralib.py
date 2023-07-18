import pandas as pd
import numpy as np
import scipy.sparse as sp
import qpsolvers as qp

def cargaDatos(year,sufix):
    '''Carga los datos de un año almacenados en los archivos [year][sufix].csv y validacion[year].csv y devuelve una tupla (tdv,ltp,meteo,estado hídrico).'''
    # Carga de datos
    df = pd.read_csv("rawMinutales"+year+sufix+".csv",na_values='.')
    df.loc[:,"Fecha"]=pd.to_datetime(df.loc[:,"Fecha"])# Fecha como datetime
    df=df.drop_duplicates(subset="Fecha")
    df.dropna(subset = ["Fecha"], inplace=True)
    df=df.set_index("Fecha")
    df=df.apply(pd.to_numeric, errors='coerce')

    # separa dfT en tdv y ltp en función del principio del nombre de cada columna y guarda el resto en meteo
    tdv = df.loc[:,df.columns.str.startswith('TDV')]
    ltp = df.loc[:,df.columns.str.startswith('LTP')]
    meteo = df.drop(df.columns[df.columns.str.startswith('TDV')], axis=1)
    meteo = meteo.drop(meteo.columns[meteo.columns.str.startswith('LTP')], axis=1)

    # Carga datos de validacion
    valdatapd=pd.read_csv("validacion"+year+".csv")
    valdatapd.dropna(inplace=True)
    valdatapd['Fecha'] = pd.to_datetime(valdatapd['Fecha'])
    valdatapd.set_index('Fecha',inplace=True)

    return (tdv,ltp,meteo,valdatapd)

def cargaDatosTDV(year,sufix):
    '''Carga los datos de un año almacenados en los archivos [year][sufix].csv y validacion[year].csv y devuelve una tupla (tdv,ltp,meteo,estado hídrico).'''
    # Carga de datos
    df = pd.read_csv("rawMinutales"+year+sufix+".csv",na_values='.')
    df.loc[:,"Fecha"]=pd.to_datetime(df.loc[:,"Fecha"])# Fecha como datetime
    df=df.drop_duplicates(subset="Fecha")
    df.dropna(subset = ["Fecha"], inplace=True)
    df=df.set_index("Fecha")
    df=df.apply(pd.to_numeric, errors='coerce')

    # separa dfT en tdv y ltp en función del principio del nombre de cada columna y guarda el resto en meteo
    tdv = df.loc[:,df.columns.str.startswith('TDV')]
    ltp = df.loc[:,df.columns.str.startswith('LTP')]
    meteo = df.drop(df.columns[df.columns.str.startswith('TDV')], axis=1)
    meteo = meteo.drop(meteo.columns[meteo.columns.str.startswith('LTP')], axis=1)

    # Carga datos de validacion
    valdatapd=pd.read_csv("validacion"+year+"TDV.csv")
    #valdatapd.dropna(inplace=True)
    valdatapd['Fecha'] = pd.to_datetime(valdatapd['Fecha'])
    valdatapd.set_index('Fecha',inplace=True)

    return (tdv,ltp,meteo,valdatapd)

def datosADataframe(ltp:pd.DataFrame,meteo:pd.DataFrame,valdatapd:pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    '''Almacena los datos de ltp y meteo en un dataframe x y los de valdata en una serie y con la forma adecuada para convertirlos a arrays de numpy para scikit o bien para continuar su procesado. X e Y no se reducen a columnas comunes.'''
    ltp['Dia'] = pd.to_datetime(ltp.index).date
    ltp['Delta'] = pd.to_datetime(ltp.index) - pd.to_datetime(ltp.index).normalize()


    meteo['Dia'] = pd.to_datetime(meteo.index).date
    meteo['Delta'] = pd.to_datetime(meteo.index) - pd.to_datetime(meteo.index).normalize()

    # ltpPdia = ltpP.loc[meteoP['R_Neta_Avg']>0]

    ltp=ltp.set_index(['Dia','Delta']).unstack(0)
    meteo=meteo.set_index(['Dia','Delta']).unstack(0).stack(0)
    valdatapd=valdatapd.unstack()

    #common_col = ltp.columns.intersection(valdatapd.index)
    #ltp=ltp[common_col]
    y=valdatapd#[common_col]

    meteoPext=pd.DataFrame(columns=ltp.columns)
    for col in meteoPext:
        meteoPext[col]=meteo[col[1]]
    x=meteoPext.unstack(0)
    x.loc['LTP']=ltp.unstack(0)
    print(x)
    x=x.stack(2)
    return (x, y)


#KRIGGING

class KriggingClassifier:
    '''Objeto para clasificación mediante Krigging. La matriz Xtrain contiene la base de datos de entrenamiento, con una fila por cada caractrística y una columna por cada muestra. El vector ytrain contiene las clases de cada muestra.'''
    # Constructor
    def __init__(self, Xtrain, alpha, ytrain):
        self.Xtrain = Xtrain
        self.alpha = alpha
        if ytrain is not None:
            self.ytrain = ytrain
        else:
            self.ytrain = np.zeros(Xtrain.shape[0])
        self.num_classes = np.unique(ytrain).shape[0]
        # obtiene los indices de las clases en el conjunto de entrenamiento
        self.indices = [np.where(self.ytrain == i)[0] for i in range(len(np.unique(self.ytrain)))]
        
        self.update_matrices(Xtrain, alpha)

    def update_matrices(self, Xtrain, alpha):
        '''Actualiza las matrices P, q, G, h y A para los datos de entrenamiento Xtrain y el parámetro alpha.'''
        # Calcula la matriz P definida como una matriz cuadrada de tamaño 2*N+1 con una matriz diagonal de tamaño N y valor 2 en la esquina superior izquierda
        self.P = np.zeros([2*Xtrain.shape[1], 2*Xtrain.shape[1]])
        self.P[:Xtrain.shape[1], :Xtrain.shape[1]] = np.eye(Xtrain.shape[1])*2
        # Convierte P en una matriz dispersa
        self.P = sp.csc_matrix(self.P)

        # Calcula el vector q de tamaño 2*N+1 con valores alpha en los N últimos elementos
        self.q = np.zeros([2*Xtrain.shape[1]])
        self.q[Xtrain.shape[1]:2*Xtrain.shape[1]] = alpha

        # Calcula la matriz G de tamaño 2*N+1 x 2*N con cuatro matrices identidad de tamaño N y una columna de ceros. Todas las matrices identidad tienen signo negativo excepto la esquina superior derecha
        self.G = np.zeros([2*Xtrain.shape[1], 2*Xtrain.shape[1]])
        self.G[:Xtrain.shape[1], :Xtrain.shape[1]] = -np.eye(Xtrain.shape[1])
        self.G[Xtrain.shape[1]:2*Xtrain.shape[1], Xtrain.shape[1]:2*Xtrain.shape[1]] = -np.eye(Xtrain.shape[1])
        self.G[Xtrain.shape[1]:2*Xtrain.shape[1], :Xtrain.shape[1]] = -np.eye(Xtrain.shape[1])
        self.G[:Xtrain.shape[1], Xtrain.shape[1]:2*Xtrain.shape[1]] = np.eye(Xtrain.shape[1])
        # Convierte G en una matriz dispersa
        self.G = sp.csc_matrix(self.G)

        # Calcula el vector h de tamaño 2*N con valores cero
        self.h = np.zeros([2*Xtrain.shape[1]])

        # Calcula la matriz A que contiene la matriz de datos de entrenamiento y N+1 columnas de 0 en la parte superior y una fila de 0 y un 1 en la parte inferior

        self.A = np.zeros([Xtrain.shape[0]+1, 2*Xtrain.shape[1]])
        self.A[:Xtrain.shape[0], :Xtrain.shape[1]] = Xtrain
        self.A[Xtrain.shape[0], :Xtrain.shape[1]] = 1
        # Convierte A en una matriz dispersa
        self.A = sp.csc_matrix(self.A)

    def update_ytrain(self, ytrain):
        '''Actualiza el vector ytrain de los datos de entrenamiento.'''
        self.ytrain = ytrain
        self.num_classes = np.unique(ytrain).shape[0]
        self.indices = [np.where(self.ytrain == i)[0] for i in range(len(np.unique(self.ytrain)))]

        

    def apply(self,x):
        '''Aplica el clasificador a un vector de características x. Devuelve el valor de la función objetivo y el vector de lambdas.'''
        # Crea un vector de características ampliado con un 1
        b = np.hstack([x, 1])

        # Calcula el T que minimiza la función Qp usando OSQP
        T = qp.solve_qp(self.P, self.q.T, self.G, self.h, self.A, b, solver='osqp')

        P= self.P.toarray()

        #comprueba si T es none
        if T is None:
            #fija el valor de la función objetivo a infinito
            f = np.inf
            #fija el vector de lambdas de los datos de entrenamiento a None
            lambda_i = None
        else:
            # Obtiene el valor de la función objetivo
            f = 0.5*np.dot(T.T, np.dot(P, T)) + np.dot(self.q, T)

            # Obtiene el vector de lambdas de los datos de entrenamiento
            lambda_i = T[:self.Xtrain.shape[1]]
        return (f, lambda_i)
    
    def lambda_classifier(self, x):
        '''Aplica el clasificador a un vector de características x. Devuelve la clase predicha en función del vector de lambdas.'''
        # Aplica el clasificador a x
        y_pred_lambda = self.apply(x)[1]
        #si el valor de lambda es None
        if y_pred_lambda is None:
            # asigna una clase mayor que el número de clases
            y_pred_lambda = self.num_classes
        else:
            # separa los valores de lambda por clases
            y_pred_lambda = [y_pred_lambda[self.indices[j]] for j in range(len(np.unique(self.ytrain)))]
            # calcula la suma de los valores de lambda por clases
            y_pred_lambda = [np.sum(y_pred_lambda[j]) for j in range(len(np.unique(self.ytrain)))]
            # selecciona la clase con mayor valor de lambda
            y_pred_lambda = np.argmax(y_pred_lambda)
        return y_pred_lambda

    def minTraining(self):
        '''Calcula el menor número de muestras de entrenamiento que se pueden usar para entrenar el clasificador'''
        # Calcula el número de características de los datos de entrenamiento
        n = self.Xtrain.shape[1]

        # Calcula el número de clases de los datos de entrenamiento
        nclases = np.unique(self.ytrain).shape[0]

        # Por cada clase, calcula el número de muestras de entrenamiento
        ntrain = np.zeros(nclases)
        for i in range(nclases):
            ntrain[i] = np.where(self.ytrain == i)[0].shape[0]
        
        # calcula el número de muestras de entrenamiento
        m = self.Xtrain.shape[0]


        print("n+1/m: ",n+1,"/",m)
        print("ntrain: ",ntrain)
        print("P: ",self.P.shape)
        print("q: ",self.q.shape)
        print("G: ",self.G.shape)
        print("h: ",self.h.shape)
        print("A: ",self.A.shape)

class KriggingFunctionClassifier:
    '''Clasificador Krigging basado en el valor de la función objetivo.'''
    def __init__(self, Xtrain, alpha, ytrain=None):
        '''Constructor de la clase. Recibe los datos de entrenamiento y el valor de alpha.'''
        # Guarda los datos de entrenamiento
        self.Xtrain = Xtrain
        if ytrain is None:
            self.ytrain = np.zeros(Xtrain.shape[0])
        # Guarda el valor de alpha
        self.alpha = alpha
        # Actualiza el vector ytrain de los datos de entrenamiento
        self.update_ytrain(ytrain)

    def update_ytrain(self, ytrain):
        '''Actualiza el vector ytrain de los datos de entrenamiento.'''
        self.ytrain = ytrain
        self.num_classes = np.unique(ytrain).shape[0]

        self.kriggings = []
        # crea un clasificador de krigging para cada clase
        for i in range(self.num_classes):
            # obtiene los datos de entrenamiento de la clase i
            Xtrain_i = self.Xtrain[:,np.where(self.ytrain == i)[0]]
            ytrain_i = self.ytrain[np.where(self.ytrain == i)[0]]
            # crea un clasificador de krigging en la lista de clasificadores
            self.kriggings.append(KriggingClassifier(Xtrain_i, self.alpha, ytrain_i))
        
    def fun_classifier(self,x):
        '''Aplica el clasificador a un vector de características x. Devuelve la clase predicha en función del valor de la función objetivo.'''
        # Aplica el clasificador a x
        y_pred_fun = [self.kriggings[i].apply(x)[0] for i in range(self.num_classes)]
        # selecciona la clase con menor valor de la función objetivo
        y_pred_fun = np.argmin(y_pred_fun)
        #si y_pred_fun es infinito, asigna una clase mayor que el número de clases
        if y_pred_fun == np.inf:
            y_pred_fun = self.num_classes
        return y_pred_fun