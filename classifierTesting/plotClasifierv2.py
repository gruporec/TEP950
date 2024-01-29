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
datasetpath=os.path.dirname(__file__)+"\\db2\\"

# grid step
gridstep=0.01

#margin for the plot
margin=0.1

#accuracy margin
acc_margin=0.1

# clasifier types to use. Available types: "lda", "qda", "krigingfun","kriginglam"
# "lda" and "qda" are the linear and quadratic discriminant analysis from sklearn
# "krigingfun", "kriginglam" and "krigingqda" are the kriging based classifiers from isadoralib
clasifs=["lda","qda","krigingfun","kriginglam","nearestneighbours"]
clasifs=["qda","kriging"]
clasifs=["krigingOpt"]

# lambda for kriging 
kr_lambda=0

fileprefix=""

# alpha for krig

#databases to plot
files=[0,1,2,3,4,5]

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
            # case "lda":
            #     clf=sklda.LinearDiscriminantAnalysis()

            #     # train the classifier
            #     clf.fit(Xtrain,Ytrain)

            #     # apply the classifier to the training and test data
            #     Ytrain_pred=clf.predict(Xtrain)
            #     Ytest_pred=clf.predict(Xtest)

            case "qda":
                clf=sklda.QuadraticDiscriminantAnalysis()

                # train the classifier
                clf.fit(Xtrain,Ytrain)

                # apply the classifier to the training and test data to obtain the probabilities
                Ytrain_pred=clf.predict_proba(Xtrain)
                Ytest_pred=clf.predict_proba(Xtest)
                #print(Ytrain_pred.shape)


            
            # case "customqda":
            #     clf=isl.qdaClassifier(Xtrain.T, Ytrain)

            #     # apply the classifier to the training and test data
            #     Ytrain_pred=np.empty(Xtrain.shape[0])
            #     for i in range(Xtrain.shape[0]):
            #         Ytrain_pred[i]=clf.qda_classifier(Xtrain[i])

            #     Ytest_pred=np.empty(Xtest.shape[0])
            #     for i in range(Xtest.shape[0]):
            #         Ytest_pred[i]=clf.qda_classifier(Xtest[i])

            case "kriging":
                # calculate the number of classes
                nclasses=len(np.unique(Ytrain))

                # calculate the number of data per class in the training data
                Nk=np.array([np.sum(Ytrain==i) for i in range(nclasses)])

                # calculate alpha
                alphak=[Nk[i]/2 for i in range(nclasses)]
                # alphak=None

                # create the classifier
                clf=isl.KrigBayesian(Xtrain.T,krig_lambda=kr_lambda, alphak=alphak, Fk=None, ytrain=Ytrain)

                # create a matrix of size nclases x Xtrain.shape[0] to store the results
                Ytrain_pred=np.empty([nclasses,Xtrain.shape[0]])

                # apply the classifier to the training and test data
                for i in range(Xtrain.shape[0]):
                    # store the results in the matrix.
                    tempres=clf.class_prob(Xtrain[i])
                    Ytrain_pred[:,i]=tempres

                # create a matrix of size nclasses x Xtest.shape[0] to store the results
                Ytest_pred=np.empty([nclasses,Xtest.shape[0]])
                for i in range(Xtest.shape[0]):
                    # store the results in the matrix.
                    tempres=clf.class_prob(Xtest[i])
                    Ytest_pred[:,i]=tempres
                # Transpose the results to match the shape of the grid
                Ytrain_pred=Ytrain_pred.T
                Ytest_pred=Ytest_pred.T
                #print(Ytrain_pred.shape)

            case "krigingOpt":
                # calculate the number of classes
                nclasses=len(np.unique(Ytrain))

                # create the classifier
                clf=isl.KrigOpt(Xtrain.T,krig_lambda=kr_lambda, alphak=None, Fk=None, ytrain=Ytrain)

                # create a matrix of size nclases x Xtrain.shape[0] to store the results
                Ytrain_pred=np.empty([nclasses,Xtrain.shape[0]])

                # apply the classifier to the training and test data
                for i in range(Xtrain.shape[0]):
                    # store the results in the matrix.
                    tempres=clf.class_prob(Xtrain[i])
                    Ytrain_pred[:,i]=tempres

                # create a matrix of size nclasses x Xtest.shape[0] to store the results
                Ytest_pred=np.empty([nclasses,Xtest.shape[0]])
                for i in range(Xtest.shape[0]):
                    # store the results in the matrix.
                    tempres=clf.class_prob(Xtest[i])
                    Ytest_pred[:,i]=tempres
                # Transpose the results to match the shape of the grid
                Ytrain_pred=Ytrain_pred.T
                Ytest_pred=Ytest_pred.T
                #print(Ytrain_pred.shape)

            # case "krigingfun":
            #     kr_lambda = isl.KriggingFunctionClassifier(Xtrain.T, kr_lambda, Ytrain)

            #     # apply the classifier to the training and test data
            #     Ytrain_pred=np.empty(Xtrain.shape[0])
            #     for i in range(Xtrain.shape[0]):
            #         Ytrain_pred[i] = kr_lambda.fun_classifier(Xtrain[i])
                
            #     Ytest_pred=np.empty(Xtest.shape[0])
            #     for i in range(Xtest.shape[0]):
            #         Ytest_pred[i] = kr_lambda.fun_classifier(Xtest[i])

            # case "kriginglam":
            #     kr_lambda = isl.KriggingClassifier(Xtrain.T, kr_lambda, Ytrain)

            #     # apply the classifier to the training and test data
            #     Ytrain_pred=np.empty(Xtrain.shape[0])
            #     for i in range(Xtrain.shape[0]):
            #         Ytrain_pred[i]= kr_lambda.lambda_classifier(Xtrain[i])

            #     Ytest_pred=np.empty(Xtest.shape[0])
            #     for i in range(Xtest.shape[0]):
            #         Ytest_pred[i]= kr_lambda.lambda_classifier(Xtest[i])

            # case "krigingqda":
            #     kr_lambda = isl.KriggingQDA(Xtrain.T, kr_lambda, Ytrain)

            #     # apply the classifier to the training and test data
            #     Ytrain_pred=np.empty(Xtrain.shape[0])
            #     for i in range(Xtrain.shape[0]):
            #         Ytrain_pred[i]= kr_lambda.qda_classifier(Xtrain[i])
                
            #     Ytest_pred=np.empty(Xtest.shape[0])
            #     for i in range(Xtest.shape[0]):
            #         Ytest_pred[i]= kr_lambda.qda_classifier(Xtest[i])
            
            # case "nearestneighbours":
            #     clf = neighbors.KNeighborsClassifier(5)

            #     # train the classifier
            #     clf.fit(Xtrain,Ytrain)

            #     # apply the classifier to the training and test data
            #     Ytrain_pred=clf.predict(Xtrain)
            #     Ytest_pred=clf.predict(Xtest)
                
            case _:
                # if the clasifier is not valid, print an error message
                print("unvalid classifier")
        # If the number of classes is 2, add a column of zeros to the probabilities
        if Ytrain_pred.shape[1]==2:
            Ytrain_pred=np.c_[Ytrain_pred,np.zeros(Ytrain_pred.shape[0])]
            Ytest_pred=np.c_[Ytest_pred,np.zeros(Ytest_pred.shape[0])]
        #print(Ytest_pred)
        

        colors=["r","g","b"]
        colors_pred = [colors[col] for col in Ytrain]

        # for each element in Ytest_pred
        for i in range(Ytest_pred.shape[0]):
            # get a ordered list of the probabilities
            problist=np.argsort(Ytest_pred[i,:])
            # get the difference between the first and second highest probabilities
            diff=Ytest_pred[i,problist[-1]]-Ytest_pred[i,problist[-2]]
            # if the difference is less than the accuracy margin
            if diff<acc_margin:
                # divide the values by 2
                Ytest_pred[i,:]=Ytest_pred[i,:]/2


        # reshape the results to the grid shape
        Ytest_pred=Ytest_pred.reshape([xx.shape[0],xx.shape[1],3])

        # plot the results using imshow
        plt.imshow(Ytest_pred,extent=[xmin,xmax,ymin,ymax],origin="lower", alpha=0.5)

        # Plot the training data with the colors defined by the colors array
        plt.scatter(Xtrain[:,0],Xtrain[:,1],c=colors_pred,marker="x",s=100)
        
        #add a title
        plt.title(clasif+" Lambda="+str(kr_lambda))
        
        #check if the plots folder exists and create it if it doesn't
        if not os.path.exists("Plots"):
            os.makedirs("Plots")
            
        # save the plot using the clasifier name and the database name as the name of the file
        plt.savefig("Plots\\kriging\\"+fileprefix+clasif+"Lambda"+str(int(kr_lambda*10))+"_"+str(file)+".png")

        #close the plot
        plt.close()