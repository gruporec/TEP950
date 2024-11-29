import pandas as pd
import numpy as np
from datetime import time
import os
import sys
import matplotlib.pyplot as plt
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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

from sklearn.decomposition import PCA

# Databases
#ZIM
dataFiles=['db/ZIMdb14151619raw.csv', 'db/ZIMdb14151619ZIM80MET4.csv','db/ZIMdb14151619ZIM15MET4.csv']
dataLabels=['raw', '80ZIM4Meteo','15ZIM4Meteo']

#TDV
#dataFiles=['db/TDVdb14151619.csv', 'db/TDVdb14151619Meteo.csv', 'db/TDVdb14151619raw.csv', 'db/TDVdb14151619meteoraw.csv']
#dataLabels=['TDV','TDVMeteo', 'TDVraw', 'TDVmeteoraw']

#ZIM
# doPCAs=[False,True,True]
# ns_components=[0,13,15]

#TDV
doPCAs=[False]
ns_components=[0]

verbose=True

# Report file in results/ZIM/ of root directory, the name of the file is the name of the classifier
reportFolder='prediction_results/'

# years to be used as training data
#years_train=[['2014'], ['2015'], ['2016'], ['2019']]
years_train=[['2014']]

# classifiers to be used
classifiers = [
    LinearDiscriminantAnalysis(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(algorithm="SAMME", random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]
# classifier labels
clf_labels = [
    "LDA",
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "BEST"
]
#Predict only one classifier (Linear SVM)
classifiers=[classifiers[2]]
clf_labels=[clf_labels[2]]

# create the report folder if it does not exist
if not os.path.exists(reportFolder):
    os.makedirs(reportFolder)

#create a dataframe to store the accuracies. The index is the classifier label. The columns names are constructed in each iteration
accpd=pd.DataFrame(index=clf_labels)
# for each database
for idata in range(len(dataFiles)):
    dataFile=dataFiles[idata]
    dataLabel=dataLabels[idata]
    for iPCA in range(len(doPCAs)):
        doPCA=doPCAs[iPCA]
        n_components=ns_components[iPCA]
        for year_train in years_train:
            if verbose:
                print("Database: ", dataLabel)
                print("PCA: ", doPCA)
                print("n_components: ", n_components)
                print("Training data: ", year_train)
                print()
            #create a name for the column
            colname=dataLabel
            if doPCA:
                colname=colname+str(n_components)+"PCA"
            colname=colname+"Train"+year_train[0]

            #create a list to store the accuracies
            accs=[]

            # best accuracy obtained
            bestacc=0
            # best classifier
            bestclf=''

            # Read data file. use the first two columns as index
            data = pd.read_csv(dataFile, sep=',', decimal='.', index_col=[0,1])

            #remove nan rows
            data=data.dropna()

            # # 'Fecha' (second index column) is a string with the format 'YYYY-MM-DD', convert it to datetime
            # data.index.set_levels(pd.to_datetime(data.index.levels[1]), level=1)
            # print(data.head())

            # get the year of each data from the second index (yyyy-mm-dd) splitting by "-"
            data["year"]=data.index.get_level_values(1).str.split("-").str[0]

            # select the training data as the data in the year_train list
            dbtrain=data.loc[data["year"].isin(year_train)].copy()
            # drop the year column
            dbtrain.drop("year", axis=1, inplace=True)

            # extract the y column
            ytrain=dbtrain["Y"]
            # drop the y column
            dbtrain.drop("Y", axis=1, inplace=True)



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
                # remove the Y and year columns from the data, storing them temporarily
                y = data["Y"]
                year = data["year"]
                data.drop(["Y", "year"], axis=1, inplace=True)
                # also store the index of the data
                index = data.index
                # transform the remaining data
                data = pca.transform(data)
                # add the index back to the data
                data = pd.DataFrame(data, index=index)
                # add the Y and year columns back to the data
                data["Y"] = y
                data["year"] = year


            # train the classifiers
            for clf in classifiers:

                if verbose:
                    print(clf)
                
                clf.fit(dbtrain, ytrain)

                ytestFull = []
                ypredFull = []

                # for each year in the data
                for year in data["year"].unique():
                    # select the data of the year
                    dbtest=data.loc[data["year"]==year].copy()
                    # drop the year column
                    dbtest.drop("year", axis=1, inplace=True)
                    # extract the y column
                    ytest=dbtest["Y"]
                    # drop the y column
                    dbtest.drop("Y", axis=1, inplace=True)

                    # predict the y values
                    ypred=clf.predict(dbtest)

                    # save the y values and the predicted y values if the year isn't in the training data
                    if year not in year_train:
                        ytestFull.extend(ytest)
                        ypredFull.extend(ypred)

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