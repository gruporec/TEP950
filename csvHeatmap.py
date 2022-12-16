import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sns.set(rc={'figure.figsize':(11.7,8.27)})

df = pd.read_csv("ignore/analisisPCALDA/resultadosPCALDA.csv", index_col=[0,1,2,3],header=[0,1], skipinitialspace=True)
accuracies=df['total accuracy'].droplevel(2,'index').unstack(level=1).droplevel(0,'columns').swaplevel(0,1,'index').sort_index('index',0)
for fraction, accuracy in accuracies.groupby(level=0):
    fig=plt.figure()
    accuracy=accuracy.droplevel(0,'index')
    accuracy=accuracy.dropna('columns','all')
    print(accuracy)
    mask=accuracy.isnull()
    sns.heatmap(accuracy, mask=mask,cmap=sns.light_palette("#2c4978", reverse=False, as_cmap=True),annot=True,vmin=0.65,vmax=0.71)
    fig.savefig('ignore/analisisPCALDA/results PCA fraction '+str(fraction)+'.png')
