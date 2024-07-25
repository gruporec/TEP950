import pandas as pd
import numpy as np
from datetime import time
import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import layers, models
import keras_tuner as kt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

from sklearn.decomposition import PCA

class CNNHypermodel(kt.HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = models.Sequential()
        model.add(layers.Input(shape=self.input_shape))

        input_size = self.input_shape[0]  # Assuming input_shape is a tuple like (length, channels)

        for i in range(hp.Int('num_conv_layers', 1, 10)):  # 1 to 10 convolutional layers
            max_kernel_size = min(3 + (10 - i), input_size)  # Adjust max kernel size based on layer index and current input size
            
            kernel_size = hp.Int('kernel_size_' + str(i), min(3, max_kernel_size), max_kernel_size)
            filters = hp.Int('filters_' + str(i), 32, 256, step=32)
            
            if kernel_size > input_size:
                break  # Skip adding more layers if the kernel size exceeds the current input size
            
            model.add(layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same'
            ))
            # Optional: Add a conditional pooling layer here
            # Update input_size based on the convolution and pooling operations
            model.add(layers.MaxPooling1D(1))            
            input_size = max(1, (input_size - kernel_size) + 1)  # Simplified calculation, adjust if using strides/padding

        
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model


# Databases
#ZIM ['db/ZIMdb14151619raw.csv','db/ZIMdb14151619ZIM80MET4.csv','db/ZIMdb14151619ZIM15MET4.csv']['raw','80ZIM4Meteo','15ZIM4Meteo']
dataFiles=['db/ZIMdb14151619meteoraw.csv']
dataLabels=['meteoraw']

#TDV
# dataFiles=['db/TDVdb14151619raw.csv', 'db/TDVdb14151619meteoraw.csv']
# #dataFiles=['db/TDVdb14151619meteoraw.csv']
# dataLabels=['TDVraw', 'TDVmeteoraw']

# N days of previous data
ndays=[0,3,6,9]

# test on all years or only on test years
testonall=False

#ZIM
# doPCAs=[False,True,True]
# ns_components=[0,13,15]

#TDV
doPCAs=[False]
ns_components=[0]

verbose=True

# Report file in results/ZIM/ of root directory, the name of the file is the name of the classifier
reportFolder='results/ZIM/AnalisysMixedUnbalanced-KNN-LDA-lSVM-RF-CNN-meteoraw/'

# years to be used as training data
#years_train=[['2014'], ['2015'], ['2016'], ['2019']]
train_frac = 0.25
val_frac = 0.25
valancedTrain = False
valancedVal = False

#fix random seed
np.random.seed(42)

# classifiers to be used
classifiers = [
    "CNN",
    LinearDiscriminantAnalysis(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    # SVC(gamma=2, C=1, random_state=42),
    # GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    # DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(max_depth=5, n_estimators=1000, max_features=1, random_state=42),
    # MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    # AdaBoostClassifier(algorithm="SAMME", random_state=42),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
]

# classifier labels
clf_labels = [
    "CNN",
    "LDA",
    "Nearest Neighbors",
    "Linear SVM",
    # "RBF SVM",
    # "Gaussian Process",
    # "Decision Tree",
    "Random Forest",
    # "Neural Net",
    # "AdaBoost",
    # "Naive Bayes",
    # "QDA",
    "BEST"
]

# create the report folder if it does not exist
if not os.path.exists(reportFolder):
    os.makedirs(reportFolder)

#create a dataframe to store the accuracies. The index is the classifier label. The columns names are constructed in each iteration
accpd=pd.DataFrame(index=clf_labels)
# for each database
for idata in range(len(dataFiles)):
    dataFile=dataFiles[idata]
    dataLabel=dataLabels[idata]

    # for each PCA configuration
    for iPCA in range(len(doPCAs)):
        doPCA=doPCAs[iPCA]
        n_components=ns_components[iPCA]

        # for each n days of previous data
        for nday in ndays:

            if verbose:
                print("Database: ", dataLabel)
                print("PCA: ", doPCA)
                print("n_components: ", n_components)
                print()
            #create a name for the column
            colname=dataLabel
            if doPCA:
                colname=colname+str(n_components)+"PCA"

            #add the n days of previous data to the column name
            colname=colname+str(nday)

            #create a list to store the accuracies
            accs=[]

            # best accuracy obtained
            bestacc=0
            # best classifier
            bestclf=''

            # Read data file. use the first two columns as index.
            data = pd.read_csv(dataFile, sep=',', decimal='.', index_col=[0,1])

            #remove nan rows
            data=data.dropna()

            # separate the data in the Y column
            dataY = data["Y"]
            # drop the Y column
            data.drop("Y", axis=1, inplace=True)

            # extend the data with the previous nday days
            # create a temporary dataframe to store the shifted data
            dataToJoin=pd.DataFrame()
            # for each day in the range of nday
            for i in range(1,nday):
                # create a temporary copy of the data and add i to the column names
                dataToShift=data.copy()
                dataToShift.columns=[col+str(i) for col in data.columns]

                dataToJoin=pd.concat([dataToJoin,dataToShift.shift(i)],axis=1)
            # add the shifted data to the original data
            data=pd.concat([data,dataToJoin],axis=1)

            # transform the y column to integer
            dataY=dataY.astype(int)

            # add the Y column back to the data
            data["Y"]=dataY

            #remove nan rows
            data=data.dropna()

            # # 'Fecha' (second index column) is a string with the format 'YYYY-MM-DD', convert it to datetime
            # data.index.set_levels(pd.to_datetime(data.index.levels[1]), level=1)
            # print(data.head())

            #create a test dataset
            dbTest=data.copy()

            #separate the data by class ("Y" column)
            db0=dbTest.loc[dbTest["Y"]==0]
            db1=dbTest.loc[dbTest["Y"]==1]
            db2=dbTest.loc[dbTest["Y"]==2]

            if valancedTrain:
                # calculate the number of training samples for each class as the fraction from the smallest class
                n0 = int(db0.shape[0]*train_frac)
                n1 = int(db1.shape[0]*train_frac)
                n2 = int(db2.shape[0]*train_frac)
                n=min(n0,n1,n2)
                n0=n
                n1=n
                n2=n
            else:
                n0 = int(db0.shape[0]*train_frac)
                n1 = int(db1.shape[0]*train_frac)
                n2 = int(db2.shape[0]*train_frac)
            
            # randomly select training data
            dbtrain=pd.concat([db0.sample(n=n0, random_state=42), db1.sample(n=n1, random_state=42), db2.sample(n=n2, random_state=42)])

            # remove the training data from the original data
            dbTest.drop(dbtrain.index, inplace=True)

            #update the separated data
            db0=dbTest.loc[dbTest["Y"]==0]
            db1=dbTest.loc[dbTest["Y"]==1]
            db2=dbTest.loc[dbTest["Y"]==2]

            # calculate the number of validating samples for each class as the fraction from the smallest class
            if valancedVal:
                n0 = int(db0.shape[0]*val_frac)
                n1 = int(db1.shape[0]*val_frac)
                n2 = int(db2.shape[0]*val_frac)
                n=min(n0,n1,n2)
                n0=n
                n1=n
                n2=n
            else:
                n0 = int(db0.shape[0]*val_frac)
                n1 = int(db1.shape[0]*val_frac)
                n2 = int(db2.shape[0]*val_frac)

            # randomly select validating data
            dbvalid=pd.concat([db0.sample(n=n0, random_state=42), db1.sample(n=n1, random_state=42), db2.sample(n=n2, random_state=42)])

            # remove the validating data from the original data
            dbTest.drop(dbvalid.index, inplace=True)

            

            # extract the y column
            ytrain=dbtrain["Y"]
            # drop the y column
            dbtrain.drop("Y", axis=1, inplace=True)

            # extract the y column
            yvalid=dbvalid["Y"]
            # drop the y column
            dbvalid.drop("Y", axis=1, inplace=True)

            # extract the y column
            ytest=dbTest["Y"]
            # drop the y column
            dbTest.drop("Y", axis=1, inplace=True)

            if doPCA:
                # create the PCA object
                pca = PCA(n_components=n_components)
                # fit the PCA object to the training data
                pca.fit(dbtrain)
                #store the training data indices
                index_train=dbtrain.index
                # transform the training data
                dbtrain = pca.transform(dbtrain)
                # add the index back to the training data
                dbtrain=pd.DataFrame(dbtrain, index=index_train)
                
                # store the validating data indices
                index_valid=dbvalid.index
                # transform the validating data
                dbvalid = pca.transform(dbvalid)
                # add the index back to the validating data
                dbvalid=pd.DataFrame(dbvalid, index=index_valid)

                # store the test data indices
                index_test=dbTest.index
                # transform the test data
                dbTest = pca.transform(dbTest)
                # add the index back to the test data
                dbTest=pd.DataFrame(dbTest, index=index_test)

            # train the classifiers
            for clf in classifiers:

                if verbose:
                    print(clf)

                # if the classifier is the CNN
                if clf_labels[classifiers.index(clf)]=="CNN":
                    #update the input shape
                    input_shape = (dbtrain.shape[1], 1)
                    num_classes = len(ytrain.unique())
                    cnnmodel = CNNHypermodel(input_shape=input_shape, num_classes=num_classes)

                    # Create the tuner
                    tuner = kt.tuners.RandomSearch(
                        hypermodel=cnnmodel,
                        objective='val_accuracy',
                        max_trials=10,
                        directory='results/ZIM/tuner'+dataLabel,
                        project_name='CNN'
                    )
                    tuner.search(dbtrain, ytrain, epochs=10, validation_data=(dbvalid, yvalid))

                    # Create the best model
                    best_hps = tuner.get_best_hyperparameters(num_trials=5)[0]
                    model = cnnmodel.build(best_hps)
                    model.fit(dbtrain, ytrain, epochs=10)
                # else, train the classifier directly
                else:
                    clf.fit(dbtrain, ytrain)

                ytestFull = []
                ypredFull = []

            

                # if the classifier is the CNN, select the predicted class as the one with the highest probability
                if clf_labels[classifiers.index(clf)]=="CNN":
                    ypred=np.argmax(model.predict(dbTest), axis=1)
                    # convert the result into a pandas series with the same index as the ytest
                    ypred=pd.Series(ypred, index=ytest.index)

                # else, predict the classes directly
                else:
                    ypred=clf.predict(dbTest)

                # if it's set to use all years as test data or the year isn't on the training data, save the y values and the predicted y values
                ytestFull.extend(ytest)
                ypredFull.extend(ypred)
                

                # show the predicted values and the real values
                if verbose:
                    print("Real values: ", ytest)
                    print("Predicted values: ", ypred)
                    print()
                    

                curracc=balanced_accuracy_score(ytestFull, ypredFull)

                if verbose:
                    print("Balanced accuracy: ", curracc)
                    print()

                if curracc>bestacc:
                    bestacc=curracc
                    bestclf=clf_labels[classifiers.index(clf)]
                
                accs.append(curracc)

            # add the best accuracy to the list of accuracies
            accs.append(bestacc)

            # add the accuracies to the dataframe
            accpd[colname]=accs
            
            if verbose:
                print("Best accuracy: ", bestacc)
                print("Best classifier: ", bestclf)
                print()
                print("Accuracies: ")
                print(accpd)

    # save the accuracies to a csv file
    accpd.to_csv(reportFolder+"accuracies.csv")