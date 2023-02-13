import sys
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


sns.set(rc={'figure.figsize':(11.7,8.27)})

#df = pd.read_csv("ignore/analisisPCALDA/resultadosPCALDAZIM.csv", index_col=[0],header=[0,1,2], skipinitialspace=True)
df = pd.read_csv("ignore/analisisPCALDA/resultadosPCALDAPCA.csv", index_col=[0,1,2,3],header=[0,1], skipinitialspace=True)
print(df)
#accuracies=df['total accuracy'].droplevel(3,'index').unstack(level=2).droplevel(0,'columns').sort_index('index',0)
accuracies=df['total accuracy'].droplevel(3,'index').unstack(level=0).droplevel(0,'columns').sort_index('index',0).droplevel(0,'index')
print(accuracies)
acc=accuracies
#ax = accuracies.quantile([0, 0.25, .5, 0.75, 1]).transpose().plot()
fig, ax = plt.subplots()
acc.plot(kind='line', color="#2c4978", linewidth=3, ax=ax,marker='.' ,zorder=2,legend=False)
print(acc.columns)
#plt.xticks(acc.index)
plt.ylabel('Average accuracy')
ax.set_xlabel('Number of PCA components')
#fig=ax.get_figure()
plt.show()
fig.savefig('ignore/analisisPCALDA/PCA Line.png')