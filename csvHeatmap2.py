import sys
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sns.set(rc={'figure.figsize':(11.7,8.27)})

df = pd.read_csv("ignore/analisisPCALDA/resultadosPCALDA3.csv", index_col=[0,1,2,3],header=[0,1], skipinitialspace=True)
accuracies=df['total accuracy'].droplevel(3,'index').unstack(level=1).droplevel(0,'columns').swaplevel(0,1,'index').sort_index('index',0)
for PCAcomp, accuracy in accuracies.groupby(level=0):
    fig=plt.figure()
    accuracy=accuracy.droplevel(0,'index')
    accuracy=accuracy.dropna('columns','all')
    print(accuracy)
    mask=accuracy.isnull()
    sns.heatmap(accuracy, mask=mask,cmap=sns.light_palette("#2c4978", reverse=False, as_cmap=True),annot=True,vmin=0.65,vmax=0.71)
    plt.xlabel('Number of samples per meteorological data')
    plt.ylabel('Number of daily ZIM probe samples')
    fig.savefig('ignore/analisisPCALDA/results PCA components '+str(PCAcomp)+'.png')

