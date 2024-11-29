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
    nB = 2000

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

    #separate the training set into the different classes
    XtrainSep = [Xtrain[ytrain == i] for i in range(nC)]
    # get a value of sigma for each class
    sigmaK = [np.cov(XtrainSep[i].T) for i in range(nC)]
    #calculate the theoretical value of F when gamma=0 and cf=2 as e^(1/2)/((2*pi)^(d/2)*det(Sigma)^(1/2))
    f0= [np.exp(0.5)/(np.power(2*np.pi, Xtrain.shape[1]/2)*np.sqrt(np.linalg.det(sigmaK[i]))) for i in range(nC)]

    #calculate c0 as the value of c when gamma=0 and cf=2
    c0 = nT/2

    # create a dissimilarity distribution classifier
    dissimClass = dcl.dissimClas(Xtrain, ytrain, gam, c, nISk=nIS, nBk=nB)

    #create a QDA dissimilarity distribution classifier
    qdaDissimClass = dcl.dissimClas(Xtrain, ytrain, 0, c0, Fk=f0)

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

    with mp.Pool(mp.cpu_count()) as pool:
        ypredqdad = np.array(list(tqdm.tqdm(pool.imap(qdaDissimClass.classify, Xval), total=Xval.shape[0])))

    # get the value of F for both dissimilarity classifiers
    print('F: ' + str(dissimClass.Fk))

    print('F QDA: ' + str(qdaDissimClass.Fk))

    #get the confusion matrices for both classifiers
    cm = confusion_matrix(yval, ypred)

    cmqdad = confusion_matrix(yval, ypredqdad)

    if doQDA:
        cmqda = confusion_matrix(yval, ypredqda)

    #print the confusion matrices
    print('Dissimilarity Classifier')
    print(cm)

    if doQDA:
        print('QDA Classifier')
        print(cmqda)

    print('Dissimilarity Classifier QDA')
    print(cmqdad)

    # save the confusion matrices to a file
    #np.savetxt('confusionMatrices/' + datasetName + 'DFC_gam' + str(gam) + '_cf' + str(cf) +'.csv', cm, delimiter=',')

    #if doQDA:
    #    np.savetxt('confusionMatrices/' + datasetName + 'QDA.csv', cmqda, delimiter=',')

        
    #normalize the confusion matrix
    cm = cm/cm.sum(axis=1)[:, np.newaxis]

    if doQDA:
        cmqda = cmqda/cmqda.sum(axis=1)[:, np.newaxis]


    # separate the validation set into the different classes
    XvalSep = [Xval[yval == i] for i in range(nC)]
    # get the likelihood of the validation set
    loglikelihood = dissimClass.getLikelihoodRatio(XvalSep)

    # get the likelihood of the validation set for the QDA classifier
    loglikelihoodqda = qdaDissimClass.getLikelihoodRatio(XvalSep)

    print('cf: ' + str(cf))
    print('gamma: ' + str(gam))
    print('Log Likelihood: ' + str(loglikelihood))
    print('Log Likelihood QDA: ' + str(loglikelihoodqda))

    # undo the log likelihood
    likelihood = np.exp(loglikelihood)
    likelihoodqda = np.exp(loglikelihoodqda)

    print('Likelihood: ' + str(likelihood))
    print('Likelihood QDA: ' + str(likelihoodqda))

    #divide the likelihoods
    likelihoodratio = [loglikelihoodqda[i]/loglikelihood[i] for i in range(nC)]

    print('Likelihood Ratio: ' + str(likelihoodratio))

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