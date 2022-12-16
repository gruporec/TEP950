import sys
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


sns.set(rc={'figure.figsize':(11.7,8.27)})

df = pd.read_csv("ignore/analisisPCALDA/resultadosPCALDA3.csv", index_col=[0,1,2,3],header=[0,1], skipinitialspace=True)
accuracies=df['total accuracy'].droplevel(3,'index').unstack(level=1).droplevel(0,'columns').swaplevel(0,1,'index').sort_index('index',0)
print(accuracies.quantile([0, 0.25, .5, 0.75, 1]))
acc=accuracies.quantile([0, 0.25, .5, 0.75, 1]).transpose()
#ax = accuracies.quantile([0, 0.25, .5, 0.75, 1]).transpose().plot()
fig, ax = plt.subplots()
grad1=[(x*(0.75-0.1725)+0.1725,x*(0.75-0.2863)+0.2863,x*(0.75-0.4706)+0.4706) for x in list(np.linspace(1,0,5))]
grad2=[(x*(0.75-0.1725)+0.1725,x*(0.75-0.2863)+0.2863,x*(0.75-0.4706)+0.4706) for x in list(np.linspace(1,0,4))]
acc.plot(kind='line', color=grad1, linewidth=3, ax=ax,marker='o' ,zorder=2,legend=True)
for i in range(0,4):
    plt.fill_between(acc.index, acc[acc.columns[i+1]],acc[acc.columns[i]], interpolate=True, color=grad2[i], alpha=0.5)
print(acc.columns)
plt.legend(['0 (minimum)', '0.25', '0.5 (median)', '0.75', '1 (maximum)'], loc='upper left', title='Quantile')
plt.xticks(acc.index)
plt.ylabel('Average accuracy')
ax.set_xlabel('Number of samples per meteorological data')
#fig=ax.get_figure()
fig.savefig('ignore/analisisPCALDA/quantile.png')


# for PCAcomp, accuracy in accuracies.groupby(level=0):
#     fig=plt.figure()
#     accuracy=accuracy.droplevel(0,'index')
#     accuracy=accuracy.dropna('columns','all')
#     print(accuracy)
#     mask=accuracy.isnull()
#     sns.heatmap(accuracy, mask=mask,cmap=sns.light_palette("#2c4978", reverse=False, as_cmap=True),annot=True,vmin=0.65,vmax=0.71)
#     fig.savefig('ignore/analisisPCALDA/results PCA components '+str(PCAcomp)+'.png')

