import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp
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

# #use agg when intended for saving the image and not for showing it
# plt.switch_backend('agg')

# ---CONFIGURATION---

#dataset file path
dataset=os.path.dirname(os.path.dirname(__file__))+"\\db\\ZIMdb14151619.csv"

# select training data mode: by year (mixed=False) or mixed (mixed=True)
mixed=True
# keep original class proportions. Only used if mixed=True
keepClassProportions = True
# balance training classes. Only used if keepClassProportions=False. If both are False, training data is selected randomly
balanceTrainingClasses = False

# number of repetitions for mixed=True
nrep=10

# fraction of data used for training if mixed=True
mixedtrains=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# training years if mixed=False
year_train=["2014"]

# apply PCA
dopca=True
# number of components for PCA
ncomp=13

# clasifier types to use. Available types: "lda", "qda", "kriggingfun","krigginglam"
# "lda" and "qda" are the linear and quadratic discriminant analysis from sklearn
# "kriggingfun" and "krigginglam" are the krigging based classifiers from isadoralib
clasifs=["lda","qda","kriggingfun","krigginglam"]

# alpha for krigging 
alphaskrigging=[1]

# print the confusion matrix
printConfusionMatrix=True

# ---END OF CONFIGURATION---

# load the database taking into account that there are 2 index columns
db=pd.read_csv(dataset,index_col=[0,1])

# create an empty list for the mean test accuracy and another for the mean training accuracy
testacc=[]
trainacc=[]

# create an empty list for the maximum test accuracy and another for the maximum training accuracy
testaccmax=[]
trainaccmax=[]

# create an empty list for the minimum test accuracy and another for the minimum training accuracy
testaccmin=[]
trainaccmin=[]
if not mixed:
    nrep=1

# define a function for pararell processing that takes a tuple with krigging alpha, the fraction of data that is used for training and the clasifier type
def process(alphamixedtrain):
    # separate the tuple elements
    alpha,mixedtrain,clasif=alphamixedtrain
    # if training data is mixed
    if mixed:
        # if the original class proportions must be kept
        if keepClassProportions:
            # create a list of dataframes separated by class according to manual classification
            dblist=[db.loc[db["Y"]==clase] for clase in db["Y"].unique()]
            # for each dataframe in the list
            for i in range(len(dblist)):
                # obtain the amount of training data
                n_train=int(dblist[i].shape[0]*mixedtrain)
                # randomly select the training data
                dblist[i]=dblist[i].sample(n=n_train)
            # concatenate the dataframes in the list into a single dataframe
            dbtrain=pd.concat(dblist)
        # if the training classes must be balanced
        elif balanceTrainingClasses:
            # get the number of classes
            nclasses = len(db["Y"].unique())
            
            # create a list of dataframes separated by class
            dblist=[db.loc[db["Y"]==clase] for clase in db["Y"].unique()]

            # get the number of samples of the smallest class
            n_min=min([dblist[i].shape[0] for i in range(len(dblist))])

            # get the number of training samples per class
            n_train=int(n_min*mixedtrain/nclasses)

            # for each dataframe in the list
            for i in range(len(dblist)):
                # randomly select the training data
                dblist[i]=dblist[i].sample(n=n_train)
            # concatenate the dataframes in the list into a single dataframe
            dbtrain=pd.concat(dblist)

        # if the training data must be selected randomly
        else:
            # get the number of training samples
            n_train=int(db.shape[0]*mixedtrain)
            # randomly select the training data
            dbtrain=db.sample(n=n_train)

    # if training data is not mixed
    else:
        # get the year of each data from the second index (yyyy-mm-dd) splitting by "-"
        db["year"]=db.index.get_level_values(1).str.split("-").str[0]

        # select the training data as the data in the year_train list
        dbtrain=db.loc[db["year"].isin(year_train)]

        # remove the year column
        dbtrain.drop(columns=["year"],inplace=True)
        db.drop(columns=["year"],inplace=True)
        
    # order dbtrain by the first index and then by the second
    dbtrain.sort_index(level=[0,1],inplace=True)

    # select the test data as the data that is not in train
    dbtest=db.drop(dbtrain.index)
    # order dbtest by the first index and then by the second
    dbtest.sort_index(level=[0,1],inplace=True)

    # split the training data into X and Y
    Xtrain=dbtrain.iloc[:,:-1]
    Ytrain=dbtrain.iloc[:,-1]

    # split the test data into X and Y
    Xtest=dbtest.iloc[:,:-1]
    Ytest=dbtest.iloc[:,-1]

    # apply PCA if dopca=True
    if dopca:
        pca = skdecomp.PCA(n_components=ncomp)
        pca.fit(Xtrain)
        Xtrain=pca.transform(Xtrain)
        Xtest=pca.transform(Xtest)

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
            
        case _:
            # if the clasifier is not valid, print an error message
            print("unvalid classifier")

    # get the balanced accuracy of the training and test data
    tracc=skmetrics.balanced_accuracy_score(Ytrain,Ytrain_pred)
    teacc=skmetrics.balanced_accuracy_score(Ytest,Ytest_pred)

    # if printConfusionMatrix=True, calculate the confusion matrix to return it
    if printConfusionMatrix:
        trcm = skmetrics.confusion_matrix(Ytrain,Ytrain_pred)
        tecm = skmetrics.confusion_matrix(Ytest,Ytest_pred)
    else:
        trcm = None
        tecm = None

    # return the training and test accuracy and confusion matrices
    return (tracc,teacc,trcm,tecm)

# main class
if __name__ == "__main__":
    # to calculate the number of combinations, the number of non-krigging classifiers is needed
    # create a list with the selected krigging classifiers
    kriggingclasifs=list({"kriggingfun","krigginglam"}.intersection(set(clasifs)))
    # calculate the number of combinations of classifier, alpha and training data fraction
    ncomb = len(kriggingclasifs)*len(alphaskrigging)*len(mixedtrains)+len(set(clasifs).difference(kriggingclasifs))*len(mixedtrains)
    # create a counter to know how many combinations have been done
    count = 0
    for clasif in clasifs:
        # reset the lists
        testacc=[]
        trainacc=[]
        testaccmax=[]
        trainaccmax=[]
        testaccmin=[]
        trainaccmin=[]

        # of the classifier is not krigging, set alpha to 0
        if clasif!="kriggingfun" and clasif!="krigginglam":
            alphas=[0]
        else:
            alphas=alphaskrigging
        for mixedtrain in mixedtrains:
            for alpha in alphas:
                print(count,"/",ncomb)
                count+=1
                # create an empty list for the test accuracy and another for the training accuracy
                testaccalpha=[]
                trainaccalpha=[]

                # create another two empty lists for the confusion matrices
                trcmalpha=[]
                tecmalpha=[]

                # create a process pool
                pool = mp.Pool(mp.cpu_count())

                #create an empty list for the tuples (alpha,mixedtrain,clasif) for each repetition
                alphal=[(alpha,mixedtrain,clasif)]*nrep

                # apply the process function to the list of tuples
                out = tqdm.tqdm(pool.imap(process, alphal), total=nrep)

                # split out into trainaccalpha, testaccalpha, trcmalpha and tecmalpha taking into account that out is a list of tuples (trainacc,testacc,trcm,tecm)
                for trainaccout,testaccout,trcmout,tecmout in out:
                    trainaccalpha.append(trainaccout)
                    testaccalpha.append(testaccout)
                    trcmalpha.append(trcmout)
                    tecmalpha.append(tecmout)
                
                # calculate the mean training and test accuracy
                tracc=np.mean(trainaccalpha)
                teacc=np.mean(testaccalpha)
                
                # add the mean training and test accuracy to the lists
                trainacc.append(tracc)
                testacc.append(teacc)

                # add the maximum training and test accuracy to the lists
                testaccmax.append(max(testaccalpha))
                trainaccmax.append(max(trainaccalpha))

                # add the minimum training and test accuracy to the lists
                testaccmin.append(min(testaccalpha))
                trainaccmin.append(min(trainaccalpha))

                # if printConfusionMatrix=True, calculate the sum of the confusion matrices in trcmalpha and tecmalpha lists
                if printConfusionMatrix:
                    trcm=np.sum(trcmalpha,axis=0)
                    tecm=np.sum(tecmalpha,axis=0)
                    # print data about the classifier, alpha and fraction of data used for training
                    print("clasifier:",clasif,", alpha:",alpha,", fraction of data used for training:",mixedtrain)
                    # print the confusion matrices
                    print("training confusion matrix:")
                    print(trcm)
                    print("test confusion matrix:")
                    print(tecm)
                    # print the normalized confusion matrices
                    print("training normalized confusion matrix:")
                    print(trcm/np.sum(trcm,axis=1)[:,None])
                    print("test normalized confusion matrix:")
                    print(tecm/np.sum(tecm,axis=1)[:,None])
                    # print the balanced accuracy
                    print("training balanced accuracy:",tracc)
                    print("test balanced accuracy:",teacc)

        # substract the mean value from the maximum and minimum values to obtain the error
        testaccmaxdif=np.array(testaccmax)-np.array(testacc)
        trainaccmaxdif=np.array(trainaccmax)-np.array(trainacc)
        testaccmindif=np.array(testaccmin)-np.array(testacc)
        trainaccmindif=np.array(trainaccmin)-np.array(trainacc)

        # combine the maximum and minimum values into an array with shape (2,N)
        testaccerr=np.array([testaccmindif,testaccmaxdif])
        trainaccerr=np.array([trainaccmindif,trainaccmaxdif])

        # get the absolute values of the errors
        testaccerr=np.abs(testaccerr)
        trainaccerr=np.abs(trainaccerr)

        # plot the balanced accuracy vs the fraction of data used for training
        plt.errorbar(mixedtrains,trainacc,yerr=trainaccerr,label=clasif+' train',capsize=5,fmt='o-')
        plt.errorbar(mixedtrains,testacc,yerr=testaccerr,label=clasif+' test',capsize=5, fmt='o-')
    plt.xlabel('fraction of data used for train')
    plt.ylabel('balanced accuracy')
    plt.legend()

    # comment if using agg
    plt.show()

    # # save the plot
    # plt.savefig(os.path.dirname(os.path.dirname(__file__))+"\\ZIMMulticlasifierAccPlot.png")

