import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp
from sklearn import neighbors
import seaborn as sns
import multiprocessing as mp
import tqdm
import sys
import os
#add the path to the lib folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
# import the isadoralib library
import isadoralib as isl

sns.set(rc={'figure.figsize':(11.7,8.27)})

#use agg when intended for saving the image and not for showing it
plt.switch_backend('agg')

# ---CONFIGURATION---

#dataset file path
datasetpath=os.path.dirname(__file__)+"\\db\\"

# grid step
gridstep=0.01

#margin for the plot
margin=0.1

# clasifier types to use. Available types: "lda", "qda", "kriggingfun","krigginglam"
# "lda" and "qda" are the linear and quadratic discriminant analysis from sklearn
# "kriggingfun" and "krigginglam" are the krigging based classifiers from isadoralib
clasifs=["lda","qda","kriggingfun","krigginglam","nearestneighbours"]

# alpha for krigging 
alpha=0

#databases to plot
files=[0,1,2,3]

# ---END OF CONFIGURATION---
for file in files:
    dataset=datasetpath+str(file)+".csv"
    # load the database taking into account that there is no index
    db=pd.read_csv(dataset,index_col=False)

    # split the database into the data and the labels
    Xtrain=db[["a","b"]].to_numpy()
    Ytrain=db["Y"].to_numpy()

    # get the limits in X
    xmin=np.min(Xtrain[:,0])
    xmax=np.max(Xtrain[:,0])
    ymin=np.min(Xtrain[:,1])
    ymax=np.max(Xtrain[:,1])

    #round the limits to the grid step
    xmin=np.floor(xmin/gridstep)*gridstep
    xmax=np.ceil(xmax/gridstep)*gridstep
    ymin=np.floor(ymin/gridstep)*gridstep
    ymax=np.ceil(ymax/gridstep)*gridstep

    # add a margin to the limits
    xmin=xmin-margin
    xmax=xmax+margin
    ymin=ymin-margin
    ymax=ymax+margin

    # create a grid of points with the limits and the grid step
    x=np.arange(xmin,xmax,gridstep)
    y=np.arange(ymin,ymax,gridstep)
    xx,yy=np.meshgrid(x,y)

    # create a list with the grid points
    Xtest=np.c_[xx.ravel(),yy.ravel()]


    for clasif in clasifs:
        # select the classifier according to clasif
        match clasif:
            case "lda":
                clf=sklda.LinearDiscriminantAnalysis()

                # train the classifier
                clf.fit(Xtrain,Ytrain)

                # apply the classifier to the training and test data
                Ytrain_pred=clf.predict(Xtrain)
                Ytest_pred=clf.predict(Xtest)

            case "qda":
                clf=sklda.QuadraticDiscriminantAnalysis()

                # train the classifier
                clf.fit(Xtrain,Ytrain)

                # apply the classifier to the training and test data
                Ytrain_pred=clf.predict(Xtrain)
                Ytest_pred=clf.predict(Xtest)

            case "kriggingfun":
                kr_lambda = isl.KriggingFunctionClassifier(Xtrain.T, alpha, Ytrain)

                # apply the classifier to the training and test data
                Ytrain_pred=np.empty(Xtrain.shape[0])
                for i in range(Xtrain.shape[0]):
                    Ytrain_pred[i] = kr_lambda.fun_classifier(Xtrain[i])
                
                Ytest_pred=np.empty(Xtest.shape[0])
                for i in range(Xtest.shape[0]):
                    Ytest_pred[i] = kr_lambda.fun_classifier(Xtest[i])

            case "krigginglam":
                kr_lambda = isl.KriggingClassifier(Xtrain.T, alpha, Ytrain)

                # apply the classifier to the training and test data
                Ytrain_pred=np.empty(Xtrain.shape[0])
                for i in range(Xtrain.shape[0]):
                    Ytrain_pred[i]= kr_lambda.lambda_classifier(Xtrain[i])

                Ytest_pred=np.empty(Xtest.shape[0])
                for i in range(Xtest.shape[0]):
                    Ytest_pred[i]= kr_lambda.lambda_classifier(Xtest[i])
            
            case "nearestneighbours":
                clf = neighbors.KNeighborsClassifier(5)

                # train the classifier
                clf.fit(Xtrain,Ytrain)

                # apply the classifier to the training and test data
                Ytrain_pred=clf.predict(Xtrain)
                Ytest_pred=clf.predict(Xtest)
                
            case _:
                # if the clasifier is not valid, print an error message
                print("unvalid classifier")

        # Plot the grid and color it according to the classifier
        plt.contourf(xx,yy,Ytest_pred.reshape(xx.shape),alpha=0.5)
        # Plot the training data
        plt.scatter(Xtrain[:,0],Xtrain[:,1],c=Ytrain)
        
        # save the plot using the clasifier name and the database name as the name of the file
        plt.savefig(clasif+"_"+str(file)+".png")

        #close the plot
        plt.close()