import sys
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
nSamples=500
#sns.set(rc={'figure.figsize':(11.7,8.27)})

df = pd.read_csv("ignore/analisisPCALDA/resultadosPCALDA3.csv", index_col=[0,1,2,3],header=[0,1], skipinitialspace=True)
accuracies=df['total accuracy'].droplevel(3,'index').unstack(level=1).droplevel(0,'columns').stack(level=0).sort_values(ascending=False)
acc=accuracies.head(nSamples).reset_index()

fig=plt.figure()
acc.iloc[:,0].hist(bins=10)
fig.savefig('ignore/analisisPCALDA/LTPhistogram'+str(nSamples)+'.png')

fig=plt.figure()
labels, counts = np.unique(acc.iloc[:,1], return_counts=True)
plt.bar(labels, counts, align='center')
plt.grid(True)
plt.gca().set_xticks(labels)
fig.savefig('ignore/analisisPCALDA/PCAhistogram'+str(nSamples)+'.png')

fig=plt.figure()
labels, counts = np.unique(acc.iloc[:,2], return_counts=True)
plt.bar(labels, counts, align='center')
plt.grid(True)
plt.gca().set_xticks(labels)
fig.savefig('ignore/analisisPCALDA/Meteohistogram'+str(nSamples)+'.png')

