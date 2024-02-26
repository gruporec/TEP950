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
    gam=19

    # Default pca components to analize. 0 or None to not use pca
    pca_comp=None

    Alreadysaved=False
    knownCK=None
    knownFK=None
    # If there is a file with the results, load the preprocessed FK and CK values and change the Alreadysaved flag to True
    if os.path.exists("classifierTesting\\results\\FkckValuesTest"+"{:.2f}".format(test_size).replace(".","") + "Cal" + "{:.2f}".format(cal_size).replace(".","") + "gamma" + "{:.2f}".format(7).replace(".","") + ".csv"):
        Fkckdf = pd.read_csv("classifierTesting\\results\\FkckValuesTest"+"{:.2f}".format(test_size).replace(".","") + "Cal" + "{:.2f}".format(cal_size).replace(".","") + "gamma" + "{:.2f}".format(7).replace(".","") + ".csv")
        Fkckdf.set_index("Dataset", inplace=True)
        Alreadysaved=True

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

        # If already saved, get the values of Fk and ck
        if Alreadysaved:
            knownCK=eval(Fkckdf.loc[dataset, "ck"])
            knownFK=eval(Fkckdf.loc[dataset, "Fk"])
            print("Ck: ", knownCK)
            print("Fk: ", knownFK)
        
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
        clfqda.fit(X_train, y_train)
        # apply the classifier to the test data
        y_pred = clfqda.predict(X_test)
        # calculate the accuracy
        accuracyqda = skmetrics.balanced_accuracy_score(y_test, y_pred)
        print(f"Accuracy for {dataset} using QDA: {accuracyqda}")

        # Count the number of samples in each class
        Nk=[np.sum(y_test==i) for i in range(len(np.unique(y_test)))]

        # Create a list that's 1 if the sample is misclassified and 0 if it's correctly classified
        err=[(y_test[i]!=y_pred[i]) for i in range(len(y_test))]

        # Divide the number of misclassified samples by the total number of samples in each class
        err=[err[i]/Nk[y_test[i]] for i in range(len(y_test))]

        # sum the values of err grouping by the value of y_test
        err=[np.sum([err[i] for i in range(len(y_test)) if y_test[i]==j]) for j in range(len(np.unique(y_test)))]

        #invert the values of err to get the class accuracy
        err=[1-i for i in err]

        print(f"Class accuracy for {dataset} using QDA: {err}")



        # add the accuracy to the results dataframe
        results.loc[dataset, "QDA accuracy"] = accuracyqda

        #get the value of ck for the dissimilarity function classifier
        ck=[np.sum(y_train==i)/2 for i in range(len(np.unique(y_train)))]

        # get the value of Fk for the dissimilarity function classifier as \frac{e^{\frac{1}{2}}}{(2\pi)^{d/2}|\Sigma_k|^{1/2}}
        Fk = [np.exp(0.5)/(np.power(2*np.pi, X_train.shape[1]/2)*np.sqrt(np.linalg.det(np.cov(X_train[y_train==i].T))) ) for i in range(len(np.unique(y_train)))]
        # create the dissimilarity function classifier
        clf=isl.DisFunClass(X_train.T, y_train, Xcal=X_cal.T, ycal=y_cal,ck=knownCK,Fk=knownFK, gam=gam, ck_init=ck, Fk_init=Fk)
        # apply the classifier to the test data
        with mp.Pool(mp.cpu_count()) as pool:
            y_pred = list(tqdm.tqdm(pool.imap(classify, [(x, clf) for x in X_test]), total=len(X_test)))

        #Store the value of Fk and ck as a string
        Fkstring=str(clf.Fk)
        ckstring=str(clf.ck)
        Fkckdf.loc[dataset, "Fk"] = Fkstring
        Fkckdf.loc[dataset, "ck"] = ckstring
        
        # calculate the accuracy
        accuracydf = skmetrics.balanced_accuracy_score(y_test, y_pred)

        print(f"Accuracy for {dataset} using Dissimilarity Function: {accuracydf}")

        # Count the number of samples in each class
        Nk=[np.sum(y_test==i) for i in range(len(np.unique(y_test)))]

        # Create a list that's 1 if the sample is misclassified and 0 if it's correctly classified
        err=[(y_test[i]!=y_pred[i]) for i in range(len(y_test))]
        # Divide the number of misclassified samples by the total number of samples in each class
        err=[err[i]/Nk[y_test[i]] for i in range(len(y_test))]
        
        # sum the values of err grouping by the value of y_test
        err=[np.sum([err[i] for i in range(len(y_test)) if y_test[i]==j]) for j in range(len(np.unique(y_test)))]

        #invert the values of err to get the class accuracy
        err=[1-i for i in err]

        print(f"Class accuracy for {dataset} using Dissimilarity Function: {err}")
        

        # add the accuracy to the results dataframe
        results.loc[dataset, "Dissim accuracy"] = accuracydf

    print(results)
    # Save the results to a csv file, including the test split ratio in the name
    results.to_csv("classifierTesting\\results\\sklearnTestResults"+"{:.2f}".format(test_size).replace(".","") + "Cal" + "{:.2f}".format(cal_size).replace(".","") + "gamma" + "{:.2f}".format(gam).replace(".","") + ".csv")
    # Save the Fk and ck values to a csv file, including the test split ratio in the name
    Fkckdf.to_csv("classifierTesting\\results\\FkckValuesTest"+"{:.2f}".format(test_size).replace(".","") + "Cal" + "{:.2f}".format(cal_size).replace(".","") + "gamma" + "{:.2f}".format(gam).replace(".","") + ".csv")