import pandas as pd
import numpy as np
import scipy.sparse as sp
import qpsolvers as qp

def cargaDatos(year,sufix):
    '''Load data corresponding to a year stored in the files [year][sufix].csv and validacion[year].csv and returns a tuple (tdv,ltp,meteo,hidric stress level).'''
    # Load data
    df = pd.read_csv("..\\rawMinutales"+year+sufix+".csv",na_values='.')
    df.loc[:,"Fecha"]=pd.to_datetime(df.loc[:,"Fecha"])# Date as datetime
    df=df.drop_duplicates(subset="Fecha")
    df.dropna(subset = ["Fecha"], inplace=True)
    df=df.set_index("Fecha")
    df=df.apply(pd.to_numeric, errors='coerce')

    # split dfT into tdv and ltp depending on the beginning of the name of each column and save the rest in meteo
    tdv = df.loc[:,df.columns.str.startswith('TDV')]
    ltp = df.loc[:,df.columns.str.startswith('LTP')]
    meteo = df.drop(df.columns[df.columns.str.startswith('TDV')], axis=1)
    meteo = meteo.drop(meteo.columns[meteo.columns.str.startswith('LTP')], axis=1)

    # Load validation data
    valdatapd=pd.read_csv("validacion"+year+".csv")
    valdatapd.dropna(inplace=True)
    valdatapd['Fecha'] = pd.to_datetime(valdatapd['Fecha'])
    valdatapd.set_index('Fecha',inplace=True)

    return (tdv,ltp,meteo,valdatapd)

def cargaDatosTDV(year,sufix):
    '''Load data corresponding to a year stored in the files [year][sufix].csv and validacion[year].csv and returns a tuple (tdv,ltp,meteo,hidric stress level).'''
    # Load data
    df = pd.read_csv("rawMinutales"+year+sufix+".csv",na_values='.')
    df.loc[:,"Fecha"]=pd.to_datetime(df.loc[:,"Fecha"])# Date as datetime
    df=df.drop_duplicates(subset="Fecha")
    df.dropna(subset = ["Fecha"], inplace=True)
    df=df.set_index("Fecha")
    df=df.apply(pd.to_numeric, errors='coerce')

    # split dfT into tdv and ltp depending on the beginning of the name of each column and save the rest in meteo
    tdv = df.loc[:,df.columns.str.startswith('TDV')]
    ltp = df.loc[:,df.columns.str.startswith('LTP')]
    meteo = df.drop(df.columns[df.columns.str.startswith('TDV')], axis=1)
    meteo = meteo.drop(meteo.columns[meteo.columns.str.startswith('LTP')], axis=1)

    # Load validation data
    valdatapd=pd.read_csv("validacion"+year+"TDV.csv")
    #valdatapd.dropna(inplace=True)
    valdatapd['Fecha'] = pd.to_datetime(valdatapd['Fecha'])
    valdatapd.set_index('Fecha',inplace=True)

    return (tdv,ltp,meteo,valdatapd)

def datosADataframe(ltp:pd.DataFrame,meteo:pd.DataFrame,valdatapd:pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    '''Save ltp and meteo data in a dataframe x and valdata in a series y with the proper shape to convert them to numpy arrays for scikit or to continue processing them. X and Y are not reduced to common columns.'''
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
    '''Krigging classifier object. The matrix Xtrain contains the training database, with a row for each feature and a column for each sample. The vector ytrain contains the classes of each sample.'''
    # Constructor
    def __init__(self, Xtrain, alpha, ytrain):
        self.Xtrain = Xtrain
        self.alpha = alpha
        if ytrain is not None:
            self.ytrain = ytrain
        else:
            self.ytrain = np.zeros(Xtrain.shape[0])
        self.num_classes = np.unique(ytrain).shape[0]
        # get the indices of the classes in the training set
        self.indices = [np.where(self.ytrain == i)[0] for i in range(len(np.unique(self.ytrain)))]
        
        self.update_matrices(Xtrain, alpha)

    def update_matrices(self, Xtrain, alpha):
        '''Updates the matrices P, q, G, h and A for the training data Xtrain and the parameter alpha.'''
        # Get P matrix as a square matrix of size 2*N+1 with a diagonal matrix of size N and value 2 in the upper left corner
        self.P = np.zeros([2*Xtrain.shape[1], 2*Xtrain.shape[1]])
        self.P[:Xtrain.shape[1], :Xtrain.shape[1]] = np.eye(Xtrain.shape[1])*2
        # Matrix P as a sparse matrix
        self.P = sp.csc_matrix(self.P)

        # Get q vector of size 2*N+1 with alpha values in the last N elements
        self.q = np.zeros([2*Xtrain.shape[1]])
        self.q[Xtrain.shape[1]:2*Xtrain.shape[1]] = alpha

        # Get G matrix of size 2*N+1 x 2*N with four identity matrices of size N and a column of zeros. All identity matrices have negative sign except the upper right corner
        self.G = np.zeros([2*Xtrain.shape[1], 2*Xtrain.shape[1]])
        self.G[:Xtrain.shape[1], :Xtrain.shape[1]] = np.eye(Xtrain.shape[1])
        self.G[Xtrain.shape[1]:2*Xtrain.shape[1], Xtrain.shape[1]:2*Xtrain.shape[1]] = -np.eye(Xtrain.shape[1])
        self.G[Xtrain.shape[1]:2*Xtrain.shape[1], :Xtrain.shape[1]] = -np.eye(Xtrain.shape[1])
        self.G[:Xtrain.shape[1], Xtrain.shape[1]:2*Xtrain.shape[1]] = -np.eye(Xtrain.shape[1])
        # G as a sparse matrix
        self.G = sp.csc_matrix(self.G)

        # Get h vector of size 2*N with zero values
        self.h = np.zeros([2*Xtrain.shape[1]])

        # Get A matrix of size M+1 x 2*N with the training data matrix and a row of zeros and a 1 in the last column
        self.A = np.zeros([Xtrain.shape[0]+1, 2*Xtrain.shape[1]])
        self.A[:Xtrain.shape[0], :Xtrain.shape[1]] = Xtrain
        self.A[Xtrain.shape[0], :Xtrain.shape[1]] = 1
        # A as a sparse matrix
        self.A = sp.csc_matrix(self.A)

    def update_ytrain(self, ytrain):
        '''Updates the vector ytrain of the training data.'''
        self.ytrain = ytrain
        self.num_classes = np.unique(ytrain).shape[0]
        self.indices = [np.where(self.ytrain == i)[0] for i in range(len(np.unique(self.ytrain)))]

        

    def apply(self,x):
        '''Applies the classifier to a feature vector x. Returns the value of the objective function and the vector of lambdas.'''
        # Create an extended feature vector with a 1
        b = np.hstack([x, 1])

        # Get T that minimizes the QP function using OSQP
        T = qp.solve_qp(self.P, self.q.T, self.G, self.h, self.A, b, solver='osqp')

        P= self.P.toarray()

        # check if T is None
        if T is None:
            # set the value of the objective function to infinity
            f = np.inf
            # set the vector of lambdas of the training data to None
            lambda_i = None
        else:
            # get the value of the objective function
            f = 0.5*np.dot(T.T, np.dot(P, T)) + np.dot(self.q, T)

            # Get the vector of lambdas of the training data
            lambda_i = T[:self.Xtrain.shape[1]]
            t_i = T[self.Xtrain.shape[1]:]
        return (f, lambda_i)
    
    def lambda_classifier(self, x):
        '''Apply the classifier to a feature vector x. Returns the predicted class based on the lambda vector. If no class can be predicted, returns a class greater than the number of training classes.'''
        # Apply the classifier to x
        y_pred_lambda = self.apply(x)[1]
        # if lambda is None:
        if y_pred_lambda is None:
            # assign a class greater than the number of classes
            y_pred_lambda = self.num_classes
        else:
            # split the lambda values by classes
            y_pred_lambda = [y_pred_lambda[self.indices[j]] for j in range(len(np.unique(self.ytrain)))]
            # get the sum of the lambda values by classes
            y_pred_lambda = [np.sum(y_pred_lambda[j]) for j in range(len(np.unique(self.ytrain)))]
            # select the class with the greatest lambda value
            y_pred_lambda = np.argmax(y_pred_lambda)
        return y_pred_lambda

    def minTraining(self):
        '''Calculates the minimum number of training samples that can be used to train the classifier. Testing showed it is not useful and will be deprecated.'''
        # get the number of features from the training data
        n = self.Xtrain.shape[1]

        # get the number of classes from the training data
        nclases = np.unique(self.ytrain).shape[0]

        # For each class, calculate the number of training samples
        ntrain = np.zeros(nclases)
        for i in range(nclases):
            ntrain[i] = np.where(self.ytrain == i)[0].shape[0]
        
        # get the number of training samples
        m = self.Xtrain.shape[0]


        print("minTraining is not useful and will be deprecated. Use of this function is not necessary as the classifier should work by itself.")

class KriggingFunctionClassifier:
    '''Krigging classifier object based on the value of the objective function. Krigging classifier object. The matrix Xtrain contains the training database, with a row for each feature and a column for each sample. The vector ytrain contains the classes of each sample.'''
    
    def __init__(self, Xtrain, alpha, ytrain=None):
        '''Class constructor. Receives the training data and the value of alpha.'''
        # Store the training data
        self.Xtrain = Xtrain
        if ytrain is None:
            self.ytrain = np.zeros(Xtrain.shape[0])
        # Store alpha value
        self.alpha = alpha
        # Update ytrain vector
        self.update_ytrain(ytrain)

    def update_ytrain(self, ytrain):
        '''Updates ytrain vector with the training data classes.'''
        self.ytrain = ytrain
        self.num_classes = np.unique(ytrain).shape[0]

        self.kriggings = []
        # create a krigging classifier for each class
        for i in range(self.num_classes):
            # get the training data for class i
            Xtrain_i = self.Xtrain[:,np.where(self.ytrain == i)[0]]
            ytrain_i = self.ytrain[np.where(self.ytrain == i)[0]]
            # create a krigging classifier in the list of classifiers
            self.kriggings.append(KriggingClassifier(Xtrain_i, self.alpha, ytrain_i))
        
    def fun_classifier(self,x):
        '''Applies the classifier to a feature vector x. Returns the predicted class based on the value of the objective function.'''
        # apply the classifier to x
        y_pred_fun = [self.kriggings[i].apply(x)[0] for i in range(self.num_classes)]
        # select the class with the lowest value of the objective function
        y_pred_fun = np.argmin(y_pred_fun)
        # if y_pred_fun is infinite, assign a class greater than the number of classes
        if y_pred_fun == np.inf:
            y_pred_fun = self.num_classes
        return y_pred_fun