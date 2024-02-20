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

#Classifiers to plot
clasifs=["dissimilarityWckCalHC"]

#Gammas to plot
gammas=[0]

#Train split
train_split=0.5

# Database to plot
file=3

for clasif in clasifs:
    for kr_gamma in gammas:

        # Form the name of the file
        filename="Plots\\borders\\"+clasif
        # If the classifier isn't qda, add the lambda value to the name; put two decimal places and remove the decimal point
        if clasif!="qda":
            filename+="Lambda"+"{:.2f}".format(kr_gamma).replace(".","")

        # If the classifier is a calibrated dissimilarity function, add the calibration split to the name
        if clasif=="dissimilarityWckCal" or clasif=="dissimilarityWckCalHC":
            filename+="Cal"+"{:.2f}".format(train_split).replace(".","")
        # Add the database number to the name
        filename+="_"+str(file)+".csv"

        # Load the probabilities from the file
        Ytest_pred=np.loadtxt(filename,delimiter=",")

        # Plot the data as a imshow
        plt.imshow(Ytest_pred, aspect='auto')
        plt.show()
        sys.exit() 