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

    # Create an empty dataframe to store the results
    results = pd.DataFrame(columns=["Dataset", "QDA accuracy", "Dissim accuracy"])
    # Set dataset as the index
    results.set_index("Dataset", inplace=True)

    # Create another dataframe to store the values of Fk and ck
    Fkckdf = pd.DataFrame(columns=["Dataset", "Fk", "ck"])
    # Set dataset as the index
    Fkckdf.set_index("Dataset", inplace=True)

    # Select the sklearn toy datasets to use. Options are: "iris", "digits", "wine", "breast_cancer", "covtype"
    datasets=["iris", "digits", "wine", "breast_cancer", "covtype"]
    #datasets=["digits"]

    # Select the amount of samples in total to use for the training and testing data
    n_samples=None

    # Select the split ratio or the amount of samples for the training and testing data
    test_size=0.5

    # Split again the training data to obtain a calibration set
    cal_size=0.5

    # Select the value of the gamma parameter for the dissimilarity function classifier
    gam=7

    # Default pca components to analize. 0 or None to not use pca
    pca_comp=None


    # For each selected dataset
    for dataset in datasets:
        # Load the dataset
        match dataset:
            case "iris":
                # Load the iris dataset
                data = skdatasets.load_iris()
                
                # Get the data and target
                xdata = data.data
                ydata = data.target
            case "diabetes":
                # Load the diabetes dataset
                data = skdatasets.load_diabetes()
                
                # Get the data and target
                xdata = data.data
                ydata = data.target
            case "digits":
                # pca components to analize. 0 or None to not use pca
                pca_comp=10
                # Load the digits dataset
                data = skdatasets.load_digits()
                
                # Get the data and target
                xdata = data.data
                ydata = data.target
            case "linnerud":
                # Load the linnerud dataset
                data = skdatasets.load_linnerud()
                
                # Get the data and target
                xdata = data.data
                ydata = data.target
            case "wine":
                # Load the wine dataset
                data = skdatasets.load_wine()
                
                # Get the data and target
                xdata = data.data
                ydata = data.target
            case "breast_cancer":
                # Load the breast_cancer dataset
                data = skdatasets.load_breast_cancer()
                
                # Get the data and target
                xdata = data.data
                ydata = data.target
            case "covtype":
                # pca components to analize. 0 or None to not use pca
                pca_comp=10
                # Load the covtype dataset
                data = skdatasets.fetch_covtype()

                # Get the data and target
                xdata = data.data
                ydata = data.target

                # Classes start at 0
                ydata = ydata - 1
                
                # Select the amount of samples in total to use for the training and testing data
                n_samples=10000
                
            case _:
                raise ValueError("Invalid dataset name")
        
        # If n_samples is not None and n_samples>0, select a random subset of the data
        if n_samples is not None and n_samples>0:
            # obtain a random subset of the data
            xdata,_,ydata,_=train_test_split(xdata, ydata, train_size=n_samples, random_state=42, stratify=ydata)

        # if pca_comp is not None and pca_comp>0:
        if pca_comp is not None and pca_comp>0:
            # create the pca object
            pca = skdecomp.PCA(n_components=pca_comp)
            # fit the pca to the data
            pca.fit(xdata)
            # apply the pca to the data
            xdata = pca.transform(xdata)

        # Obtain a random training and testing split
        X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=test_size, random_state=42, stratify=ydata)

        # Obtain a random calibration split
        X_t, X_cal, y_t, y_cal = train_test_split(X_train, y_train, test_size=cal_size, random_state=42, stratify=y_train)

        # create the qda classifier
        clfqda = sklda.QuadraticDiscriminantAnalysis()
        # fit the classifier to the training data  
        clfqda.fit(X_t, y_t)
        # apply the classifier to the test data
        y_pred = clfqda.predict(X_test)
        # calculate the accuracy
        accuracyqda = skmetrics.accuracy_score(y_test, y_pred)
        print(f"Accuracy for {dataset} using QDA: {accuracyqda}")

        # add the accuracy to the results dataframe
        results.loc[dataset, "QDA accuracy"] = accuracyqda

    print(results)
    # Save the results to a csv file, including the test split ratio in the name
    results.to_csv("classifierTesting\\results\\qdares.csv")
