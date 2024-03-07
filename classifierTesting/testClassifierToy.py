import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import qpsolvers as qp
import sys
import matplotlib.path as mpath
import dissimClassLib as dcl
import sklearn.discriminant_analysis as sklda
import multiprocessing as mp
import tqdm
import winsound
import sklearn.datasets as skds
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn.decomposition as skd

if __name__ == '__main__':
    #fix the seed
    np.random.seed(42)

    ############################################ PARAMETER SETTING ############################################

    # fraction of data to use for validation
    n = 0.6

    # number of points for Importance Sampling
    nIS = 1000

    # number of points for b calculation
    nB = 1000

    # gamma parameter. Equivalent gamma will be gam/gamma2. gamma2 is not really working too well when it comes to getting the value of F so keep at 1 for now
    gam = 0.3
    gamma2=1

    #c fraction
    cf= 8

    # Name the dataset for the filename
    datasetName = 'breast_cancer'

    # whether to use the QDA classifier
    doQDA = True

    ############################################ PROCESSING ############################################

    match datasetName:
        case 'iris':
            # load the iris dataset
            data = skds.load_iris()
        case 'digits':
            # load the digits dataset
            data = skds.load_digits()
            # apply a PCA to the data
            pca = skd.PCA(n_components=10)
            data.data = pca.fit_transform(data.data)
        case 'wine':
            # load the wine dataset
            data = skds.load_wine()
        case 'breast_cancer':
            # load the breast cancer dataset
            data = skds.load_breast_cancer()

    # split the dataset into training and validation sets
    Xtrain, Xval, ytrain, yval = train_test_split(data.data, data.target, test_size=n, random_state=42, stratify=data.target)
    
    # get the number of classes
    nC = len(np.unique(ytrain))

    # get the number of points in each class
    nT = np.bincount(ytrain)

    # set a list of already searched values of c and gamma
    searched = []

    # check if gamma is a list of values
    if type(gam) == list:
        # check if the length of gamma is the same as the number of classes
        if len(gam) != nC:
            # repeat the first value of gamma until the length of gamma is the same as the number of classes
            gam = [gam[0]]*nC
    else:
        # repeat gamma until the length of gamma is the same as the number of classes
        gam = [gam]*nC

    # check if cf is a list of values
    if type(cf) == list:
        # check if the length of cf is the same as the number of classes
        if len(cf) != nC:
            # repeat the first value of cf until the length of cf is the same as the number of classes
            cf = [cf[0]]*nC
    else:
        # repeat cf until the length of cf is the same as the number of classes
        cf = [cf]*nC
    
    # cf as array
    cf = np.array(cf)

    # get the value of c as the number of points in the training set for each class divided by the fraction
    c = nT/cf

    # create a dissimilarity distribution classifier
    dissimClass = dcl.dissimClas(Xtrain, ytrain, gam, c, nISk=nIS, nBk=nB)

    # calculate cf
    cf = nT/dissimClass.ck

    if doQDA:
        # create a qda classifier
        qda = sklda.QuadraticDiscriminantAnalysis()
        qda.fit(Xtrain, ytrain)

    # get the predictions for the validation set
    with mp.Pool(mp.cpu_count()) as pool:
        ypred = np.array(list(tqdm.tqdm(pool.imap(dissimClass.classify, Xval), total=Xval.shape[0])))

    if doQDA:
        ypredqda = qda.predict(Xval)

    #get the confusion matrices for both classifiers
    cm = confusion_matrix(yval, ypred)

    if doQDA:
        cmqda = confusion_matrix(yval, ypredqda)

    #print the confusion matrices
    print('Dissimilarity Classifier')
    print(cm)

    if doQDA:
        print('QDA Classifier')
        print(cmqda)

    # save the confusion matrices to a file
    np.savetxt('confusionMatrices/' + datasetName + 'DFC_gam' + str(gam) + '_cf' + str(cf) +'.csv', cm, delimiter=',')

    if doQDA:
        np.savetxt('confusionMatrices/' + datasetName + 'QDA.csv', cmqda, delimiter=',')

        
    #normalize the confusion matrix
    cm = cm/cm.sum(axis=1)[:, np.newaxis]

    if doQDA:
        cmqda = cmqda/cmqda.sum(axis=1)[:, np.newaxis]

    # get the likelihood of the training set
    likelihood = dissimClass.getLikelihoodRatio()

    print('cf: ' + str(cf))
    print('gamma: ' + str(gam))
    print('Likelihood: ' + str(likelihood))

    # print some interesting metrics about the classifier performance
    print('Accuracy: ' + str(np.sum(np.diag(cm))/np.sum(cm)))
    print('Precision: ' + str(np.diag(cm)/np.sum(cm, axis=0)))
    print('Recall: ' + str(np.diag(cm)/np.sum(cm, axis=1)))



    if doQDA:
        print('QDA Accuracy: ' + str(np.sum(np.diag(cmqda))/np.sum(cmqda)))
        print('QDA Precision: ' + str(np.diag(cmqda)/np.sum(cmqda, axis=0)))
        print('QDA Recall: ' + str(np.diag(cmqda)/np.sum(cmqda, axis=1)))

    doQDA = False

    frequency = [1000, 1500, 1000]  # Set Frequency To 2500 Hertz
    duration = 200  # Set Duration To 500 ms == 0.5 second

    # Play the beeps
    for i in range(3):
        winsound.Beep(frequency[i], duration)