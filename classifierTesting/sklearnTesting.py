import pandas as pd
import numpy as np
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp
import sklearn.datasets as skdatasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import multiprocessing as mp
import tqdm
import sys
import os
import random

#add the path to the lib folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
# import the isadoralib library
import isadoralib as isl

# create a function to apply the classifier to the training and test data in parallel
def classify(arg):
    (x,clf)=arg
    ret=clf.classify(x)
    return ret


if __name__ == '__main__':

    # Select the sklearn toy datasets to use. Options are: "iris", "digits", "wine", "breast_cancer"
    datasets=["iris", "digits", "linnerud", "wine", "breast_cancer"]
    #datasets=["iris"]

    # Select the split ratio for the training and testing data
    test_size=0.5

    # Select the value of the gamma parameter for the dissimilarity function classifier
    gam=0


    # For each selected dataset
    for dataset in datasets:
        # Load the dataset
        match dataset:
            case "iris":
                # Load the iris dataset
                data = skdatasets.load_iris()
            case "diabetes":
                # Load the diabetes dataset
                data = skdatasets.load_diabetes()
            case "digits":
                # Load the digits dataset
                data = skdatasets.load_digits()
            case "linnerud":
                # Load the linnerud dataset
                data = skdatasets.load_linnerud()
            case "wine":
                # Load the wine dataset
                data = skdatasets.load_wine()
            case "breast_cancer":
                # Load the breast_cancer dataset
                data = skdatasets.load_breast_cancer()
            case _:
                raise ValueError("Invalid dataset name")
        # Get the data and target
        xdata = data.data
        ydata = data.target

        # Obtain a random training and testing split
        X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=test_size, random_state=42, stratify=ydata)

        # create the qda classifier
        clfqda = sklda.QuadraticDiscriminantAnalysis()
        # fit the classifier to the training data  
        clfqda.fit(X_train, y_train)
        # apply the classifier to the test data
        y_pred = clfqda.predict(X_test)
        # calculate the accuracy
        accuracyqda = skmetrics.accuracy_score(y_test, y_pred)
        print(f"Accuracy for {dataset} using QDA: {accuracyqda}")

        #get the value of ck for the dissimilarity function classifier
        ck=[np.sum(y_train==i)/2 for i in range(len(np.unique(y_train)))]
        # create the dissimilarity function classifier
        clf=isl.DisFunClass(X_train.T, y_train,ck=ck,Fk=None,gam=gam)
        # apply the classifier to the test data
        with mp.Pool(mp.cpu_count()) as pool:
            y_pred = list(tqdm.tqdm(pool.imap(classify, [(x, clf) for x in X_test]), total=len(X_test)))
        
        # calculate the accuracy
        accuracydf = skmetrics.accuracy_score(y_test, y_pred)

        print(f"Accuracy for {dataset} using Dissimilarity Function: {accuracydf}")


                        # clf=isl.DisFunClass(Xtrain.T, Ytrain,ck=None,Fk=None)
                        
                        # #apply the classifier to the training and test data to obtain the probabilities. these loops can be parallelized
                        # with mp.Pool(mp.cpu_count()) as pool:
                        #     Ytrain_pred = list(tqdm.tqdm(pool.imap(classify_probs, [(x, clf) for x in Xtrain]), total=len(Xtrain)))
                        # with mp.Pool(mp.cpu_count()) as pool:
                        #     Ytest_pred = list(tqdm.tqdm(pool.imap(classify_probs, [(x, clf) for x in Xtest]), total=len(Xtest)))