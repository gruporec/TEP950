import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import seaborn as sns
import multiprocessing as mp
import tqdm
import sys
import os
#add the path to the lib folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
# import the isadoralib library
import isadoralib as isl


# create a function to apply the classifier to the training and test data in parallel
def classify_probs(arg):
    (x,clf)=arg
    ret=clf.classifyProbs(x)
    return ret


if __name__ == '__main__':

    sns.set(rc={'figure.figsize':(11.7,8.27)})

    #use agg when intended for saving the image and not for showing it
    #plt.switch_backend('agg')

    # ---CONFIGURATION---

    #dataset file path
    datasetpath=os.path.dirname(__file__)+"\\dbv2\\"

    # fix the limits of the plot
    plotlims=(-2,2,-2,2)
    #plotlims=None

    # grid step
    gridstep=0.005


    # calibration/training split
    train_split=0.5

    #databases to plot
    files=[3]

    # ---END OF CONFIGURATION---
    for file in files:
        dataset=datasetpath+str(file)+".csv"
        # load the database taking into account that there is no index
        db=pd.read_csv(dataset,index_col=False)

        # split the database into the data and the labels
        Xtrain=db[["a","b"]].to_numpy()
        Ytrain=db["Y"].to_numpy()

        

        # separate the training data into training and calibration data
        Xtrain, Xcal, Ytrain, Ycal = train_test_split(Xtrain, Ytrain, test_size=train_split, random_state=42, stratify=Ytrain)
        
        # Plot the training data for class 0 in grey with shape x
        plt.scatter(Xtrain[Ytrain==0,0],Xtrain[Ytrain==0,1],color="grey",marker="x",s=10)

        # Plot the training data for class 1 in grey with shape o
        plt.scatter(Xtrain[Ytrain==1,0],Xtrain[Ytrain==1,1],color="grey",marker="o",s=10)
        
        # Plot the calibration data for class 0 in black with shape x
        plt.scatter(Xcal[Ycal==0,0],Xcal[Ycal==0,1],color="black",marker="x",s=10)

        # Plot the calibration data for class 1 in black with shape o
        plt.scatter(Xcal[Ycal==1,0],Xcal[Ycal==1,1],color="black",marker="o",s=10)

        # Plot a square with corners (-1,-1) and (1,1) in black with dotted lines
        plt.plot([-1,1],[1,1],color="black",linestyle="--")
        plt.plot([1,1],[1,-1],color="black",linestyle="--")
        plt.plot([1,-1],[-1,-1],color="black",linestyle="--")
        plt.plot([-1,-1],[-1,1],color="black",linestyle="--")

        # Show the plot
        plt.show()