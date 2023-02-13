import sys
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sns.set(rc={'figure.figsize':(11.7,8.27)})

df = pd.read_csv("ignore/analisisPCALDA/resultadosPCALDAnoMeteo.csv", index_col=[0,1,2,3],header=[0,1], skipinitialspace=True)
print(df)
accuracies=df['total accuracy'].droplevel(3,'index').unstack(level=2).droplevel(0,'columns').swaplevel(0,1,'index').sort_index('index',0).loc[0]

print(accuracies)
fig=plt.figure()
accuracy=accuracies.dropna('columns','all')
print(accuracy)
mask=accuracy.isnull()
sns.heatmap(accuracy, mask=mask,cmap=sns.light_palette("#2c4978", reverse=False, as_cmap=True),annot=True,vmin=0.65,vmax=0.71)
plt.xlabel('Number of PCA components')
plt.ylabel('Number of daily ZIM probe samples')
fig.savefig('ignore/analisisPCALDA/results PCA no meteo.png')

