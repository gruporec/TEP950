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


# ZIM data file in db/ZIMdb14151619Meteo.csv of root directory
dataFile='db/ZIMdb14151619raw.csv'

sufix="raw15PCA"
doPCA=True
n_components=15

# Report file in results/ZIM/ of root directory, the name of the file is the name of the classifier
reportFolder='results/ZIM/'+sufix+'/'

# years to be used as training data
year_train=['2014']

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
]

# best accuracy obtained
bestacc=0
# best classifier
bestclf=''

#list of accuracies
accs=[]

# Read ZIM data file. use the first two columns as index
data = pd.read_csv(dataFile, sep=',', decimal='.', index_col=[0,1])

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

# create the report folder if it does not exist
if not os.path.exists(reportFolder):
    os.makedirs(reportFolder)

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

        # save the report
        reportString = "Report for year " + year + "\n"
        reportString += "Confusion matrix\n"
        reportString += str(confusion_matrix(ytest, ypred)) + "\n"
        reportString += "Normalized confusion matrix\n"
        reportString += str(confusion_matrix(ytest, ypred, normalize='true')) + "\n"
        reportString += "Accuracy\n"
        reportString += str(balanced_accuracy_score(ytest, ypred)) + "\n"
        reportString += "Classification report\n"
        reportString += str(classification_report(ytest, ypred)) + "\n"

        # save the report to a file
        reportFile = reportFolder + clf_labels[classifiers.index(clf)] + year+sufix + ".txt"
        with open(reportFile, 'w') as f:
            f.write(reportString)

    # Save the report for all years
    reportString = "Report for all years\n"
    reportString += "Confusion matrix\n"
    reportString += str(confusion_matrix(ytestFull, ypredFull)) + "\n"
    reportString += "Normalized confusion matrix\n"
    reportString += str(confusion_matrix(ytestFull, ypredFull, normalize='true')) + "\n"
    reportString += "Accuracy\n"
    reportString += str(balanced_accuracy_score(ytestFull, ypredFull)) + "\n"
    reportString += "Classification report\n"
    reportString += str(classification_report(ytestFull, ypredFull)) + "\n"

    if balanced_accuracy_score(ytestFull, ypredFull)>bestacc:
        bestacc=balanced_accuracy_score(ytestFull, ypredFull)
        bestclf=clf_labels[classifiers.index(clf)]
    
    accs.append(balanced_accuracy_score(ytestFull, ypredFull))

    # save the report to a file
    reportFile = reportFolder + clf_labels[classifiers.index(clf)] +sufix+ ".txt"
    with open(reportFile, 'w') as f:
        f.write(reportString)
    #print the classifier and the accuracy
    print(clf_labels[classifiers.index(clf)], balanced_accuracy_score(ytestFull, ypredFull))

# save the accuracies to a file
reportFile = reportFolder + "accuracies"+sufix+".csv"
with open(reportFile, 'w') as f:
    f.write("Classifier, Accuracy\n")
    for i in range(len(accs)):
        f.write(clf_labels[i] + ", " + str(accs[i]) + "\n")

print("Best accuracy: ", bestacc)
print("Best classifier: ", bestclf)