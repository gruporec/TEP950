import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp
import isadoralib as isl
import seaborn as sns
import multiprocessing as mp
import tqdm

sns.set(rc={'figure.figsize':(11.7,8.27)})

dataset="db14151619.csv"

# modos de selección de datos de entrenamiento: por año (mixed=False) o mezclados (mixed=True)
mixed=True
#manteder proporciones de clases originales. Solo se usa si mixed=True
keepClassProportions = False
#balancear clases de entrenamiento. Solo se usa si keepClassProportions=False. Si ambos son False, se seleccionan los datos de entrenamiento aleatoriamente
balanceTrainingClasses = True

# número de repeticiones para mixed=True
nrep=1000

# fracción de datos de entrenamiento si mixed=True
mixedtrains=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

# años de datos de entrenamiento si mixed=False
year_train=["2014"]

# aplicar PCA
dopca=True
# componentes para PCA
ncomp=13

# tipo de clasificador: "lda", "qda", "kriggingfun","krigginglam"
clasif="qda"

# alpha para krigging
alphas=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

# carga los datos de la base de datos teniendo en cuenta que hay 2 columnas de índice
db=pd.read_csv(dataset,index_col=[0,1])

# crea una lista vacía para la precisión media de test y otra para la precisión media de entrenamiento
testacc=[]
trainacc=[]

# crea una lista vacía para el valor máximo de la precisión de test y otra para el valor máximo de la precisión de entrenamiento
testaccmax=[]
trainaccmax=[]

# crea una lista vacía para el valor mínimo de la precisión de test y otra para el valor mínimo de la precisión de entrenamiento
testaccmin=[]
trainaccmin=[]
if not mixed:
    nrep=1

#si el clasificador no es krigging, alpha se pone a 0
if clasif!="kriggingfun" and clasif!="krigginglam":
    alphas=[0]

def process(alphamixedtrain):
    #separa alpha y mixedtrain
    alpha,mixedtrain=alphamixedtrain
    if mixed:
        if keepClassProportions:
            #crea una lista de dataframes separados por clase
            dblist=[db.loc[db["Y"]==clase] for clase in db["Y"].unique()]
            #por cada dataframe de la lista
            for i in range(len(dblist)):
                #calcula el número de datos de entrenamiento
                n_train=int(dblist[i].shape[0]*mixedtrain)
                #selecciona los datos de entrenamiento aleatoriamente
                dblist[i]=dblist[i].sample(n=n_train)
            #concatena los dataframes de la lista en un único dataframe
            dbtrain=pd.concat(dblist)
        elif balanceTrainingClasses:
            #obtiene el número de clases
            nclasses = len(db["Y"].unique())
            
            #crea una lista de dataframes separados por clase
            dblist=[db.loc[db["Y"]==clase] for clase in db["Y"].unique()]

            #obtiene el número mínimo de datos de cada clase
            n_min=min([dblist[i].shape[0] for i in range(len(dblist))])

            #obtiene el número de datos de entrenamiento por clase
            n_train=int(n_min*mixedtrain/nclasses)

            #por cada dataframe de la lista
            for i in range(len(dblist)):
                #selecciona los datos de entrenamiento aleatoriamente
                dblist[i]=dblist[i].sample(n=n_train)
            #concatena los dataframes de la lista en un único dataframe
            dbtrain=pd.concat(dblist)

        else:
            #calcula el número de datos de entrenamiento
            n_train=int(db.shape[0]*mixedtrain)
            #selecciona los datos de entrenamiento aleatoriamente
            dbtrain=db.sample(n=n_train)
    else:
        #obtiene el año de cada dato a partir del segundo índice (yyyy-mm-dd) haciendo un split por "-"
        db["year"]=db.index.get_level_values(1).str.split("-").str[0]

        #selecciona los datos de entrenamiento como los que están en la lista year_train
        dbtrain=db.loc[db["year"].isin(year_train)]

        #elimina la columna year
        dbtrain.drop(columns=["year"],inplace=True)
        db.drop(columns=["year"],inplace=True)
        
    #ordena dbtrain por el primer índice y luego por el segundo
    dbtrain.sort_index(level=[0,1],inplace=True)

    #selecciona los datos de test como los que no están en train
    dbtest=db.drop(dbtrain.index)
    #ordena dbtest por el primer índice y luego por el segundo
    dbtest.sort_index(level=[0,1],inplace=True)

    #separa los datos de entrenamiento en X y Y
    Xtrain=dbtrain.iloc[:,:-1]
    Ytrain=dbtrain.iloc[:,-1]

    #separa los datos de test en X y Y
    Xtest=dbtest.iloc[:,:-1]
    Ytest=dbtest.iloc[:,-1]

    #realiza PCA si dopca=True
    if dopca:
        pca = skdecomp.PCA(n_components=ncomp)
        pca.fit(Xtrain)
        Xtrain=pca.transform(Xtrain)
        Xtest=pca.transform(Xtest)

    #haz un match case para seleccionar el clasificador
    match clasif:
        case "lda":
            clf=sklda.LinearDiscriminantAnalysis()

            #entrena el clasificador
            clf.fit(Xtrain,Ytrain)

            #aplica el clasificador a los datos de entrenamiento y de test
            Ytrain_pred=clf.predict(Xtrain)
            Ytest_pred=clf.predict(Xtest)

        case "qda":
            clf=sklda.QuadraticDiscriminantAnalysis()

            #entrena el clasificador
            clf.fit(Xtrain,Ytrain)

            #aplica el clasificador a los datos de entrenamiento y de test
            Ytrain_pred=clf.predict(Xtrain)
            Ytest_pred=clf.predict(Xtest)
        case "kriggingfun":
            kr_lambda = isl.KriggingFunctionClassifier(Xtrain.T, alpha, Ytrain)

            #aplica el clasificador a los datos de entrenamiento y de test
            Ytrain_pred=np.empty(Xtrain.shape[0])
            for i in range(Xtrain.shape[0]):
                # aplica el clasificador
                Ytrain_pred[i] = kr_lambda.fun_classifier(Xtrain[i])
            
            Ytest_pred=np.empty(Xtest.shape[0])
            for i in range(Xtest.shape[0]):
                # aplica el clasificador
                Ytest_pred[i] = kr_lambda.fun_classifier(Xtest[i])
        case "krigginglam":
            kr_lambda = isl.KriggingClassifier(Xtrain.T, alpha, Ytrain)

            #aplica el clasificador a los datos de entrenamiento y de test
            Ytrain_pred=np.empty(Xtrain.shape[0])
            for i in range(Xtrain.shape[0]):
                Ytrain_pred[i]= kr_lambda.lambda_classifier(Xtrain[i])

            Ytest_pred=np.empty(Xtest.shape[0])
            for i in range(Xtest.shape[0]):
                Ytest_pred[i]= kr_lambda.lambda_classifier(Xtest[i])
            
        case _:
            print("clasificador no válido")

    #calcula la precisión balanceada de los datos de entrenamiento y de test
    tracc=skmetrics.balanced_accuracy_score(Ytrain,Ytrain_pred)
    teacc=skmetrics.balanced_accuracy_score(Ytest,Ytest_pred)

    # print("Alpha: ",alpha)
    # print("Precisión balanceada de entrenamiento: ",tracc)
    # print("Precisión balanceada de test: ",teacc)
    #añade la precisión de entrenamiento y de test a las listas
    return (tracc,teacc)

#clase main
if __name__ == "__main__":
    for mixedtrain in mixedtrains:
        for alpha in alphas:
            # crea una lista vacía para la precisión de test y otra para la precisión de entrenamiento
            testaccalpha=[]
            trainaccalpha=[]
            # crea una pool de procesos
            pool = mp.Pool(mp.cpu_count())

            #crea una lista con las tuplas (mixedtrain,alpha) para cada repetición
            alphal=[(alpha,mixedtrain)]*nrep
            out = tqdm.tqdm(pool.imap(process, alphal), total=nrep)
            #separa out en trainaccalpha,testaccalpha teniendo en cuenta que out es una lista de tuplas (trainacc,testacc)
            for trainaccout,testaccout in out:
                trainaccalpha.append(trainaccout)
                testaccalpha.append(testaccout)
            #calcula la precisión media de entrenamiento y de test
            tracc=np.mean(trainaccalpha)
            teacc=np.mean(testaccalpha)
            #añade la precisión media de entrenamiento y de test a las listas
            trainacc.append(tracc)
            testacc.append(teacc)

            #añade el valor máximo de la precisión de test a la lista
            testaccmax.append(max(testaccalpha))
            trainaccmax.append(max(trainaccalpha))

            #añade el valor mínimo de la precisión de test a la lista
            testaccmin.append(min(testaccalpha))
            trainaccmin.append(min(trainaccalpha))

    # resta al valor máximo y al mínimo el valor medio para obtener el error
    testaccmax=np.array(testaccmax)-np.array(testacc)
    trainaccmax=np.array(trainaccmax)-np.array(trainacc)
    testaccmin=np.array(testaccmin)-np.array(testacc)
    trainaccmin=np.array(trainaccmin)-np.array(trainacc)
    # combina los valores máximos y mínimos en un array con la forma (2,N)
    testaccerr=np.array([testaccmin,testaccmax])
    trainaccerr=np.array([trainaccmin,trainaccmax])

    #convierte los errores en valores absolutos
    testaccerr=np.abs(testaccerr)
    trainaccerr=np.abs(trainaccerr)

    #crea una gráfica con los valores de alpha y la precisión balanceada
    # plt.plot(alphas,trainacc,label='train')
    # plt.plot(alphas,testacc,label='test')
    plt.errorbar(mixedtrains,trainacc,yerr=trainaccerr,label='train',capsize=5,fmt='o-')
    plt.errorbar(mixedtrains,testacc,yerr=testaccerr,label='test',capsize=5, fmt='o-')
    plt.xlabel('fraction of data used for train')
    plt.ylabel('balanced accuracy')
    plt.legend()
    plt.show()

