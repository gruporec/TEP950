import sys
from time import time
from matplotlib.markers import MarkerStyle
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import time
import isadoralib as isl
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp

year="2014"
alphaValue=1
# Ejecuta cargaRaw.py si no existe rawDiarios.csv o rawMinutales.csv
if not os.path.isfile("rawDiarios"+year+".csv") or not os.path.isfile("rawMinutales"+year+".csv"):
    os.system("python3 cargaRaw.py")

# Carga de datos

tdv,ltp,meteo,valdatapd=isl.cargaDatosTDV(year,"")

valdatacols = list(valdatapd.columns)
valdataind = valdatapd.index

valdatapd=valdatapd.groupby(valdatapd.index.date).max()

tdv=tdv[valdatacols]

tdvavg24 = tdv.rolling(window=24*60,center=True).mean()
# calcula el valor medio de tdv para cada dia
tdv_medio = tdv.groupby(tdv.index.date).mean()
# calcula el valor de la desviación estándar de tdv para cada dia
tdv_std = tdv.groupby(tdv.index.date).std()

tdv_max = tdv.groupby(tdv.index.date).max()
tdv_min = tdv.groupby(tdv.index.date).min()
tdv_amp = tdv_max - tdv_min

tdv_cummax = tdv_medio.cummax()

tdv_de = tdv_medio - tdv_cummax

meteo_mean=meteo.groupby(meteo.index.date).mean()

print(valdatapd)
print(tdv_amp.loc[valdataind])
print(tdv_de.loc[valdataind])

tdv_dataset = [valdatapd,tdv_amp.loc[valdataind],tdv_de.loc[valdataind]]

#añade las columnas de la desviación de tdv de los dias anteriores
for i in range(1,14,1):
    tdv_de_prev=tdv_de.copy()
    tdv_de_prev.index = pd.to_datetime(tdv_de_prev.index)
    tdv_de_prev.index = tdv_de_prev.index + pd.Timedelta(days=i)
    tdv_de_prev=tdv_de_prev.groupby(tdv_de_prev.index.date).max()
    
    tdv_dataset.append(tdv_de_prev.loc[tdv_de_prev.index.intersection(valdatapd.index)])

#añade las columnas del incremento de la desviación de tdv de los dias anteriores
for i in range(1,14,1):
    tdv_de_prev=tdv_de.copy()
    tdv_de_prev.index = pd.to_datetime(tdv_de_prev.index)
    tdv_de_prev.index = tdv_de_prev.index + pd.Timedelta(days=i)
    tdv_de_prev=tdv_de_prev.groupby(tdv_de_prev.index.date).max()

    tdv_de_prev=tdv_de-tdv_de_prev

    tdv_dataset.append(tdv_de_prev.loc[tdv_de_prev.index.intersection(valdatapd.index)])

#añade las columnas del incremento de tdv de los dias anteriores
for i in range(1,14,1):
    tdv_diff=tdv_medio.diff(i)
    tdv_dataset.append(tdv_diff.loc[tdv_diff.index.intersection(valdatapd.index)])

tdv_data=pd.concat(tdv_dataset, axis=1, keys=['stress level', 'Amp', 'Dev']+['Dev ' + str(s) + 'd' for s in [*range(1,14,1)]]+['Dev Inc ' + str(s) + 'd' for s in [*range(1,14,1)]]+['Inc ' + str(s) + 'd' for s in [*range(1,14,1)]])

print(tdv_data)
print(meteo_mean)
#meteo_sensor=pd.MultiIndex.from_product([valdatapd.columns, meteo_mean],names=['sensor', 'meteo'])
#meteo_sensor=meteo_sensor.transpose().merge(meteo_mean,how='cross')
meteo_sensor=pd.concat([meteo_mean]*len(valdatapd.columns), axis=1, keys=valdatapd.columns)
print(meteo_sensor)

#invierte el orden de los indices de columnas de meteo_sensor
meteo_sensor.columns = meteo_sensor.columns.swaplevel(0,1)
print(meteo_sensor)

#añade las columnas de meteo a tdv_data donde coincidan en el segundo índice
tdv_data=tdv_data.merge(meteo_sensor,how='left',left_index=True,right_index=True)
print(tdv_data)

#añade las columnas de meteo de dias anteriores
for i in range(1,14,1):
    meteo_prev=meteo_mean.copy()
    meteo_prev.index = pd.to_datetime(meteo_prev.index)
    meteo_prev.index = meteo_prev.index + pd.Timedelta(days=i)
    meteo_prev=meteo_prev.groupby(meteo_prev.index.date).max()
    meteo_prev=pd.concat([meteo_prev]*len(valdatapd.columns), axis=1, keys=valdatapd.columns)
    meteo_prev.columns = meteo_prev.columns.swaplevel(0,1)
    tdv_data=tdv_data.merge(meteo_prev,how='left',left_index=True,right_index=True,suffixes=('',' '+str(i)+'d'))

tdv_data=tdv_data.stack(1)

tdv_data=tdv_data.dropna(how='any')

tdv_1 = tdv_data.loc[tdv_data['stress level'] == 1].dropna(how='all').drop('stress level',axis=1)
tdv_2 = tdv_data.loc[tdv_data['stress level'] == 2].dropna(how='all').drop('stress level',axis=1)
tdv_3 = tdv_data.loc[tdv_data['stress level'] == 3].dropna(how='all').drop('stress level',axis=1)


tdv_1 = tdv_1.stack().unstack(1).reset_index().drop('level_0',axis=1).set_index('level_1')
tdv_2 = tdv_2.stack().unstack(1).reset_index().drop('level_0',axis=1).set_index('level_1')
tdv_3 = tdv_3.stack().unstack(1).reset_index().drop('level_0',axis=1).set_index('level_1')

tdv_1["Last"] = np.nan
tdv_2["Last"] = np.nan
tdv_3["Last"] = np.nan

print(tdv_1)

fig, (axlpt1,axlpt2,axlpt3) = plt.subplots(3, 1)
# grafica tdv_1 poniendo la hora normalizada en el eje x
axlpt1.plot(tdv_1,ls='none',marker='o',color='black',alpha=alphaValue*50/len(tdv_1),MarkerSize=5)
#axlpt1.set_xlim(0,24)
#axlpt1.set_ylim(-2.5,2.5)
axlpt1.set_title('TDV_1')
# grafica tdv_2 poniendo la hora normalizada en el eje x
axlpt2.plot(tdv_2,ls='none',marker='o',color='black',alpha=alphaValue*50/len(tdv_2),MarkerSize=5)
#axlpt2.set_xlim(0,24)
#axlpt2.set_ylim(-2.5,2.5)
axlpt2.set_title('TDV_2')
# grafica tdv_3 poniendo la hora normalizada en el eje x
axlpt3.plot(tdv_3,ls='none',marker='o',color='black',alpha=alphaValue*50/len(tdv_3),MarkerSize=5)
#axlpt3.set_xlim(0,24)
#axlpt3.set_ylim(-2.5,2.5)
axlpt3.set_title('TDV_3')

# calcula las lineas medias de los 3 tdv
# agrupa las columnas de tdv_1, tdv_2 y tdv_3 en una sola haciendo la media
tdv_1_media=tdv_1.reset_index().set_index(['level_1'], append=True).mean(axis=1).unstack().mean(axis=0)
tdv_2_media=tdv_2.reset_index().set_index(['level_1'], append=True).mean(axis=1).unstack().mean(axis=0)
tdv_3_media=tdv_3.reset_index().set_index(['level_1'], append=True).mean(axis=1).unstack().mean(axis=0)

#obtiene los máximos
tdv_1_std_sup=tdv_1.reset_index().set_index(['level_1'], append=True).max(axis=1).unstack().max(axis=0)
tdv_2_std_sup=tdv_2.reset_index().set_index(['level_1'], append=True).max(axis=1).unstack().max(axis=0)
tdv_3_std_sup=tdv_3.reset_index().set_index(['level_1'], append=True).max(axis=1).unstack().max(axis=0)

#obtiene los mínimos
tdv_1_std_inf=tdv_1.reset_index().set_index(['level_1'], append=True).min(axis=1).unstack().min(axis=0)
tdv_2_std_inf=tdv_2.reset_index().set_index(['level_1'], append=True).min(axis=1).unstack().min(axis=0)
tdv_3_std_inf=tdv_3.reset_index().set_index(['level_1'], append=True).min(axis=1).unstack().min(axis=0)


# crea una nueva figura
fig2,(tdvavg)=plt.subplots(1,1,sharex=True)
# grafica las lineas medias de los 3 tdv juntas con leyenda
tdvavg.plot(tdv_1_media,ls='-',label='TDV_1')
tdvavg.plot(tdv_2_media,ls='-',label='TDV_2')
tdvavg.plot(tdv_3_media,ls='-',label='TDV_3')
tdvavg.plot(tdv_1_std_sup,ls='--',color='blue')
tdvavg.plot(tdv_1_std_inf,ls='--',color='blue')
tdvavg.plot(tdv_2_std_sup,ls='--',color='orange')
tdvavg.plot(tdv_2_std_inf,ls='--',color='orange')
tdvavg.plot(tdv_3_std_sup,ls='--',color='green')
tdvavg.plot(tdv_3_std_inf,ls='--',color='green')
tdvavg.legend()

#plt.show()

x=tdv_data.drop('stress level',axis=1)
y=tdv_data['stress level']

print(np.shape(x))
print(np.shape(y))


pca = skdecomp.PCA(n_components=15)
pca.fit(x)
x = pca.transform(x)

clf = sklda.LinearDiscriminantAnalysis(solver='svd')
# entrena el modelo
clf.fit(x,y)

x_new=clf.transform(x)

ypred=clf.predict(x)

bcm=skmetrics.confusion_matrix(y, ypred,normalize='true')
print(bcm)
accuracy = skmetrics.balanced_accuracy_score(y, ypred)
print(accuracy)

rep_new=pd.concat([y.reset_index().drop(['level_0','level_1'],axis=1), pd.DataFrame(x_new)], axis=1)

def dfScatter(df, xcol=0, ycol=1, catcol='stress level'):
    fig, ax = plt.subplots()
    categories = np.unique(df[catcol])
    colors = np.linspace(0, 1, len(categories))
    colordict = dict(zip(categories, colors))  

    df["Color"] = df[catcol].apply(lambda x: colordict[x])
    ax.scatter(df[xcol], df[ycol], c=df.Color)
    return fig

fig = dfScatter(rep_new)
plt.show()
