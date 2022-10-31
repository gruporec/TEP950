import sys
from time import time
from matplotlib.markers import MarkerStyle
import matplotlib
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import time
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp
import isadoralib as isl

year_train="2019"
year_data="2019"
sufix="rht"
comp=4
fltp=4
fmeteo=4
saveFolder="ignore/figures/PCALDAMETEOresults/"+year_train+"-"+year_data+"-"+sufix+"/"
n_dias_print=5
matplotlib.use("Agg")

#pd.set_option('display.max_rows', None)

# Carga de datos de entrenamiento
tdvT,ltpT,meteoT,trdatapd=isl.cargaDatos(year_train,sufix)

# Carga de datos de predicción
tdvP,ltpP,meteoP,valdatapd=isl.cargaDatos(year_data,sufix)

# guarda la información raw para plots
ltpPlot = ltpP.copy()
meteoPlot = meteoP.copy()

# añade meteo a ltpT
ltpT=ltpT.join(meteoT)
# elimina los valores NaN de ltp
ltpT = ltpT.dropna(axis=1,how='all')

# añade meteo a ltpP
ltpP=ltpP.join(meteoP)

# elimina los valores NaN de ltp
ltpP = ltpP.dropna(axis=1,how='all')
# rellena los valores NaN de ltp con el valor anterior
ltpP = ltpP.fillna(method='ffill')
ltpT = ltpT.fillna(method='ffill')
# rellena los valores NaN de ltp con el valor siguiente
ltpP = ltpP.fillna(method='bfill')
ltpT = ltpT.fillna(method='bfill')

# aplica un filtro de media móvil a ltp
ltpP = ltpP.rolling(window=240,center=True).mean()
ltpT = ltpT.rolling(window=240,center=True).mean()

# calcula el valor medio de ltp para cada dia
ltp_medioP = ltpP.groupby(ltpP.index.date).mean()
ltp_medioT = ltpT.groupby(ltpT.index.date).mean()

# calcula el valor de la desviación estándar de ltp para cada dia
ltp_stdP = ltpP.groupby(ltpP.index.date).std()
ltp_stdT = ltpT.groupby(ltpT.index.date).std()

# cambia el índice a datetime
ltp_medioP.index = pd.to_datetime(ltp_medioP.index)
ltp_medioT.index = pd.to_datetime(ltp_medioT.index)
ltp_stdP.index = pd.to_datetime(ltp_stdP.index)
ltp_stdT.index = pd.to_datetime(ltp_stdT.index)

# remuestrea ltp_medio y ltp_std a minutal
ltp_medioP = ltp_medioP.resample('T').pad()
ltp_medioT = ltp_medioT.resample('T').pad()
ltp_stdP = ltp_stdP.resample('T').pad()
ltp_stdT = ltp_stdT.resample('T').pad()

# normaliza ltp para cada dia

ltpP = (ltpP - ltp_medioP) / ltp_stdP
ltpT = (ltpT - ltp_medioT) / ltp_stdT

# obtiene todos los cambios de signo de R_Neta_Avg en el dataframe meteo
signosP = np.sign(meteoP.loc[:,meteoP.columns.str.startswith('R_Neta_Avg')]).diff()
signosT = np.sign(meteoT.loc[:,meteoT.columns.str.startswith('R_Neta_Avg')]).diff()
# obtiene los cambios de signo de positivo a negativo
signos_pnP = signosP<0
signos_pnT = signosT<0
# elimina los valores falsos (que no sean cambios de signo)
signos_pnP = signos_pnP.replace(False,np.nan).dropna()
signos_pnT = signos_pnT.replace(False,np.nan).dropna()
# obtiene los cambios de signo de negativo a positivo
signos_npP = signosP>0
signos_npT = signosT>0
# elimina los valores falsos (que no sean cambios de signo)
signos_npP = signos_npP.replace(False,np.nan).dropna()
signos_npT = signos_npT.replace(False,np.nan).dropna()

# duplica el índice de signos np como una columna más en signos_np
signos_npP['Hora'] = signos_npP.index
signos_npT['Hora'] = signos_npT.index
# recorta signos np al primer valor de cada día
signos_npP = signos_npP.resample('D').first()
signos_npT = signos_npT.resample('D').first()

#elimina los dias en los que no haya cambio de signo
signos_npP=signos_npP.dropna()
signos_npT=signos_npT.dropna()

# duplica el índice de signos pn como una columna más en signos_pn
signos_pnP['Hora'] = signos_pnP.index
signos_pnT['Hora'] = signos_pnT.index
# recorta signos pn al último valor de cada día
signos_pnP = signos_pnP.resample('D').last()
signos_pnT = signos_pnT.resample('D').last()

#elimina los días en los que no haya cambio de signo
signos_pnP = signos_pnP.dropna()
signos_pnT = signos_pnT.dropna()

# recoge los valores del índice de ltp donde la hora es 00:00
ltp_00P = ltpP.index.time == time.min
ltp_00T = ltpT.index.time == time.min
# recoge los valores del índice de ltp donde la hora es la mayor de cada día
ltp_23P = ltpP.index.time == time(23,59)
ltp_23T = ltpT.index.time == time(23,59)

# crea una columna en ltp que vale 0 a las 00:00
ltpP.loc[ltp_00P,'Hora_norm'] = 0
ltpT.loc[ltp_00T,'Hora_norm'] = 0
# iguala Hora_norm a 6 en los índices de signos np
ltpP.loc[signos_npP['Hora'],'Hora_norm'] = 6
ltpT.loc[signos_npT['Hora'],'Hora_norm'] = 6
# iguala Hora_norm a 18 en los índices de signos pn
ltpP.loc[signos_pnP['Hora'],'Hora_norm'] = 18
ltpT.loc[signos_pnT['Hora'],'Hora_norm'] = 18
# iguala Hora_norm a 24 en el último valor de cada día
ltpP.loc[ltp_23P,'Hora_norm'] = 24
ltpT.loc[ltp_23T,'Hora_norm'] = 24
# iguala el valor en la última fila de Hora_norm a 24
ltpP.loc[ltpP.index[-1],'Hora_norm'] = 24
ltpT.loc[ltpT.index[-1],'Hora_norm'] = 24
# interpola Hora_norm en ltp
ltpP.loc[:,'Hora_norm'] = ltpP.loc[:,'Hora_norm'].interpolate()
ltpT.loc[:,'Hora_norm'] = ltpT.loc[:,'Hora_norm'].interpolate()

# almacena los valores antes de recortar
ltpPBase=ltpP

# recorta ltp a los tramos de 6 a 18 de hora_norm
ltpP = ltpP.loc[ltpP['Hora_norm']>=6,:]
ltpT = ltpT.loc[ltpT['Hora_norm']>=6,:]
ltpP = ltpP.loc[ltpP['Hora_norm']<=18,:]
ltpT = ltpT.loc[ltpT['Hora_norm']<=18,:]


# añade la hora normalizada al índice de ltp
ltpP.index = [ltpP.index.strftime('%Y-%m-%d'),ltpP['Hora_norm']]
ltpT.index = [ltpT.index.strftime('%Y-%m-%d'),ltpT['Hora_norm']]

#crea el índice de ltpPBase
print('indice de ltpBase')
print(ltpPBase)
ltpPBase['Hora_norm']=ltpPBase['Hora_norm'].apply(pd.to_timedelta,unit='H')
ltpPBase['dia_norm'] = ltpPBase.index.strftime('%Y-%m-%d')
ltpPBase.index = [ltpPBase['dia_norm'].apply(pd.to_datetime,format='%Y-%m-%d'),ltpPBase['Hora_norm']]
ltpPBase=ltpPBase.drop('Hora_norm',axis=1)
ltpPBase=ltpPBase.drop('dia_norm',axis=1)
ltpPBase=ltpPBase.unstack(level=0)

valdatapd.index = valdatapd.index.strftime('%Y-%m-%d')
trdatapd.index = trdatapd.index.strftime('%Y-%m-%d')

print('fin indice')

#obtiene el índice interseccion de valdatapd y el primer nivel del índice de ltp
ltpPdates = ltpP.index.get_level_values(0)
ltpTdates = ltpT.index.get_level_values(0)

valdatapd_ltp = valdatapd.index.intersection(ltpPdates)
trdatapd_ltp = trdatapd.index.intersection(ltpTdates)

# vuelve a separar los valores de meteo de ltp
meteoP_norm=ltpP.drop(ltpP.columns[ltpP.columns.str.startswith('LTP')], axis=1)
meteoT_norm=ltpT.drop(ltpT.columns[ltpT.columns.str.startswith('LTP')], axis=1)

# elimina los valores de ltp que no estén en valdatapd
ltpv = ltpP.loc[valdatapd_ltp,valdatapd.columns]
# elimina los valores de ltp que no estén en trdatapd
ltpt = ltpT.loc[trdatapd_ltp,trdatapd.columns]

# unstackea meteoP_norm y meteoT_norm
meteoP_norm = meteoP_norm.unstack(level=0)
meteoT_norm = meteoT_norm.unstack(level=0)

# unstackea ltpv
ltpv = ltpv.unstack(level=0)
# unstackea ltpt
ltpt = ltpt.unstack(level=0)

# crea un índice para ajustar frecuencias
ltpv_index_float=pd.Int64Index(np.floor(ltpv.index*1000000000))
ltpt_index_float=pd.Int64Index(np.floor(ltpt.index*1000000000))
meteoP_index_float=pd.Int64Index(np.floor(meteoP_norm.index*1000000000))
meteoT_index_float=pd.Int64Index(np.floor(meteoT_norm.index*1000000000))

# convierte el indice a datetime para ajustar frecuencias
ltpv.index = pd.to_datetime(ltpv_index_float)
ltpv=ltpv.resample(str(fltp)+'S').mean()
ltpt.index = pd.to_datetime(ltpt_index_float)
ltpt=ltpt.resample(str(fltp)+'S').mean()
meteoP_norm.index = pd.to_datetime(meteoP_index_float)
meteoP_norm=meteoP_norm.resample(str(fmeteo)+'S').mean()
meteoT_norm.index = pd.to_datetime(meteoT_index_float)
meteoT_norm=meteoT_norm.resample(str(fmeteo)+'S').mean()

# conserva los valores de 1970-01-01 00:00:06.000 a 1970-01-01 00:00:17.900
ltpv = ltpv.loc[ltpv.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
ltpt = ltpt.loc[ltpt.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
ltpv = ltpv.loc[ltpv.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]
ltpt = ltpt.loc[ltpt.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]
meteoP_norm = meteoP_norm.loc[meteoP_norm.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
meteoP_norm = meteoP_norm.loc[meteoP_norm.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]
meteoT_norm = meteoT_norm.loc[meteoT_norm.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
meteoT_norm = meteoT_norm.loc[meteoT_norm.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]


# Crea una serie para restaurar el índice
norm_index=pd.Series(np.arange(6,18,fltp))
#recorta norm_index para que coincida con el tamano de ltpt si se ha producido un desajuste al calcular el dataframe
norm_index=norm_index.loc[norm_index.index<len(ltpt)]
# Ajusta el índice de ltpv a la serie
ltpv.index=norm_index
# Ajusta el índice de ltpt a la serie
ltpt.index=norm_index
# Crea una serie para restaurar el índice
norm_index=pd.Series(np.arange(6,18,fmeteo))
#recorta norm_index para que coincida con el tamano de meteoP_norm si se ha producido un desajuste al calcular el dataframe
norm_index=norm_index.loc[norm_index.index<len(meteoT_norm)]
# Ajusta el índice de meteoP_norm a la serie
meteoP_norm.index=norm_index
# Ajusta el índice de meteoT_norm a la serie
meteoT_norm.index=norm_index

# dropea la columna Hora_norm de meteo
meteoP_norm = meteoP_norm.drop('Hora_norm',axis=1)
meteoT_norm = meteoT_norm.drop('Hora_norm',axis=1)

# stackea meteoP_norm y meteoT_norm
meteoP_norm = meteoP_norm.stack(level=0)
meteoT_norm = meteoT_norm.stack(level=0)

#intercambia los niveles del índice de meteo
meteoP_norm.index = meteoP_norm.index.swaplevel(0,1)
meteoT_norm.index = meteoT_norm.index.swaplevel(0,1)

meteoP_norm=meteoP_norm.dropna(axis=1,how='all')
meteoT_norm=meteoT_norm.dropna(axis=1,how='all')

#combina los dos índices de meteo
meteoP_norm.index = meteoP_norm.index.map('{0[1]}/{0[0]}'.format)
meteoT_norm.index = meteoT_norm.index.map('{0[1]}/{0[0]}'.format)

#elimina los indices no comunes de meteo
meteoP_norm = meteoP_norm.loc[meteoP_norm.index.isin(meteoT_norm.index)]
meteoT_norm = meteoT_norm.loc[meteoT_norm.index.isin(meteoP_norm.index)]

#crea un array de numpy en blanco
array_ltpv=np.empty((len(ltpv)+len(meteoP_norm),0))
array_ltpt=np.empty((len(ltpt)+len(meteoT_norm),0))

#por cada elemento en el primer índice de columnas de ltp
for i in ltpv.columns.levels[0]:
    ltpv_col=ltpv.loc[:,i]
    # elimina los valores de meteo que no estén en ltp_col
    meteo_ltp = ltpv_col.columns.intersection(meteoP_norm.columns)
    meteoP_col = meteoP_norm.loc[:,meteo_ltp]

    # combina los valores de ltpv con los de meteo
    merge_ltp_meteo = pd.merge(ltpv.loc[:,i],meteoP_col,how='outer')
    # añade la unión al array de numpy
    array_ltpv=np.append(array_ltpv,merge_ltp_meteo.values,axis=1)

#por cada elemento en el primer índice de columnas de ltp
for i in ltpt.columns.levels[0]:
    ltpt_col=ltpt.loc[:,i]
    # elimina los valores de meteo que no estén en ltp_col
    meteo_ltp = ltpt_col.columns.intersection(meteoT_norm.columns)
    meteoT_col = meteoT_norm.loc[:,meteo_ltp]

    # combina los valores de ltpv con los de meteo
    merge_ltp_meteo = pd.merge(ltpt.loc[:,i],meteoT_col,how='outer')
    # añade la unión al array de numpy
    array_ltpt=np.append(array_ltpt,merge_ltp_meteo.values,axis=1)
print("ltpt")
print(ltpt)
print("meteo_ltp")
print(meteo_ltp)
print("meteoT_norm")
print(meteoT_norm)
print("ltpt_col")
print(ltpt_col)

# crea los valores X e y para el modelo
Xtr=array_ltpt.transpose()
Ytr=trdatapd.unstack().values
Xv=array_ltpv.transpose()
Yv=valdatapd.unstack().values

#print(np.shape(Xtr))
#print(np.shape(Xv))

# elimina los valores NaN de Xtr y Xv
XtrBase = np.nan_to_num(Xtr)
XvBase = np.nan_to_num(Xv)

# crea un array de tamaño MaxComp
#aplica PCA
pca = skdecomp.PCA(n_components=comp+1)
pca.fit(XtrBase)
Xtr = pca.transform(XtrBase)
Xv = pca.transform(XvBase)

# crea el modelo
clf = sklda.LinearDiscriminantAnalysis(solver='svd')
# entrena el modelo
clf.fit(Xtr,Ytr)
# predice los valores de Yv
Ypred=clf.predict(Xv)

# predice las probabilidades de Yv
Yprob=clf.predict_proba(Xv)

# calcula la matriz de confusion
confusion_matrix = skmetrics.confusion_matrix(Yv, Ypred)

print(confusion_matrix)

bcm=skmetrics.confusion_matrix(Yv, Ypred,normalize='true')

print(bcm)

# calcula el porcentaje de acierto
accuracy = skmetrics.balanced_accuracy_score(Yv, Ypred)
print('Porcentaje de acierto: '+str(accuracy*100)+'%')


#print(pca.inverse_transform(clf.coef_))

ltpPBase=ltpPBase.resample('5T').mean()
# for column in range(ltpPBase.shape[1]):
#     print(ltpPBase.columns.values[column])
#     print(ltpPBase.columns.values[column][1])
#     print(ltpPBase.iloc[:,column].dropna())
# sys.exit()

# #Datos de partida
# for column in range(ltpPBase.shape[1]):

#     #crea la carpeta si no existe
#     if not os.path.exists(saveFolder+'/'+ltpPBase.columns.values[column][0]):
#         os.makedirs(saveFolder+'/'+ltpPBase.columns.values[column][0])
#     plt.figure()
#     ax=ltpPBase.iloc[:,column].dropna().plot()
#     #añade lineas verticales al amanecer y atardecer
#     x_ns = pd.Timedelta(seconds=60*60*6) / pd.Timedelta(1,'ns') #Seconds to nanoseconds
#     ax.axvline(x=x_ns, color='r', linestyle='dashed', linewidth=2)
#     x_ns = pd.Timedelta(seconds=60*60*18) / pd.Timedelta(1,'ns') #Seconds to nanoseconds
#     ax.axvline(x=x_ns, color='r', linestyle='dashed', linewidth=2)


#     plt.savefig(saveFolder+'/'+ltpPBase.columns.values[column][0]+'/'+ltpPBase.columns.values[column][1].strftime("%Y-%m-%d")+'.png')
#     plt.close()

res=pd.DataFrame()
res['estado real']=valdatapd.unstack()
res['estado estimado']=pd.DataFrame(Ypred,index=res.index)
res['error']=res['estado real']-res['estado estimado']
res['probabilidad estado 1']=Yprob[:,0]
res['probabilidad estado 2']=Yprob[:,1]
res['probabilidad estado 3']=Yprob[:,2]
res.to_csv('resClasPCALDAMeteo'+year_train+year_data+sufix+'2.csv')

res = res.transpose()
for i in range(len(Ypred)):
    if str(res.iloc[:,i]['estado real'])!=str(res.iloc[:,i]['estado estimado']):
        maxProb=max(res.iloc[:,i]['probabilidad estado 1'],res.iloc[:,i]['probabilidad estado 2'],res.iloc[:,i]['probabilidad estado 3'])
        currFolder=saveFolder+'/prob '+f'{maxProb:.2f}'+' re '+str(res.iloc[:,i]['estado real'])+' est '+str(res.iloc[:,i]['estado estimado'])+'/'+res.iloc[:,i].name[0]+'/'+res.iloc[:,i].name[1]
        if not os.path.exists(currFolder):
            os.makedirs(currFolder)

            # crea una figura con dos subplots
            fig = plt.figure(figsize=(10,10))
            plt.subplot(2,1,1)
            plt.plot(range(len(XvBase[i])),XvBase[i],label='XvBase')

            plt.plot(range(len(pca.inverse_transform(clf.coef_)[0])),pca.inverse_transform(clf.coef_)[0],label='Estrés 1')
            plt.plot(range(len(pca.inverse_transform(clf.coef_)[1])),pca.inverse_transform(clf.coef_)[1],label='Estrés 2')
            plt.plot(range(len(pca.inverse_transform(clf.coef_)[2])),pca.inverse_transform(clf.coef_)[2],label='Estrés 3')
            
            #agrega una leyenda
            plt.legend()
            #agrega una etiqueta a los ejes
            plt.xlabel('Característica número')
            plt.ylabel('Valor')

            
            plt.subplot(2,1,2)
            plt.plot(range(len(Xv[i])),Xv[i],label='Xv')
            plt.plot(range(len(clf.coef_[0])),clf.coef_[0],label='Estrés 1')
            plt.plot(range(len(clf.coef_[1])),clf.coef_[1],label='Estrés 2')
            plt.plot(range(len(clf.coef_[2])),clf.coef_[2],label='Estrés 3')
            
            #agrega una leyenda
            plt.legend()
            #agrega una etiqueta a los ejes
            plt.xlabel('Característica número')
            plt.ylabel('Valor')

            plt.savefig(currFolder+'/DatosIA.png')
            plt.close()
            #f = open(currFolder+'/informe y notas.txt', "w")
            #prob1=res.iloc[:,i]['probabilidad estado 1']*100
            #prob2=res.iloc[:,i]['probabilidad estado 2']*100
            #prob3=res.iloc[:,i]['probabilidad estado 3']*100
            #informe='Aparece como '+str(res.iloc[:,i]['estado real'])+' __ en las notas del IRNAS.\nDías anteriores y posteriores __.\n\nLa IA devuelve un estado 1 con una probabilidad del '+f'{prob1:.2f}'+'%, estado 2 con una probabilidad del '+f'{prob2:.2f}'+'% y estado 3 con una probabilidad del '+f'{prob3:.2f}'+'%.\n\nEn observación manual de la curva, se observa ___.'
            #f.write(informe)
            #f.close()
            
            plotSensor=res.iloc[:,i].name[0]
            plotDate=pd.to_datetime(res.iloc[:,i].name[1])
            meteod=meteoPlot[plotDate-pd.Timedelta(n_dias_print-1,'d'):plotDate+pd.Timedelta(1,'d')]
            ltpd=ltpPlot[plotSensor][plotDate-pd.Timedelta(n_dias_print-1,'d'):plotDate+pd.Timedelta(1,'d')]

            fig, [ax1,ax2,ax3,ax4,ax5] = plt.subplots(5,1)

            ax1.plot(ltpd)
            ax1.set(xlabel='Hora', ylabel='LTP', title=plotSensor+' '+str(plotDate))
            ax1.fill_between(meteod.index, ltpd.min(), ltpd.max(), where=(meteod['R_Neta_Avg'] < 0), alpha=0.5,color=(232/255, 222/255, 164/255, 0.5))
            ax1.grid()

            ax2.plot(meteod['T_Amb_Avg'])
            ax2.set(xlabel='Hora', ylabel='T amb (ºC)')
            ax2.fill_between(meteod.index, meteod['T_Amb_Avg'].min(), meteod['T_Amb_Avg'].max(), where=(meteod['R_Neta_Avg'] < 0), alpha=0.5,color=(232/255, 222/255, 164/255, 0.5))
            ax2.grid()

            ax3.plot(meteod['H_Relat_Avg'])
            ax3.set(xlabel='Hora', ylabel='H rel (%)')
            ax3.fill_between(meteod.index, meteod['H_Relat_Avg'].min(), meteod['H_Relat_Avg'].max(), where=(meteod['R_Neta_Avg'] < 0), alpha=0.5,color=(232/255, 222/255, 164/255, 0.5))
            ax3.grid()

            ax4.plot(meteod['VPD_Avg'])
            ax4.set(xlabel='Hora', ylabel='VPD')
            ax4.fill_between(meteod.index, meteod['VPD_Avg'].min(), meteod['VPD_Avg'].max(), where=(meteod['R_Neta_Avg'] < 0), alpha=0.5,color=(232/255, 222/255, 164/255, 0.5))
            ax4.grid()

            ax5.plot(meteod['R_Neta_Avg'])
            ax5.set(xlabel='Hora', ylabel='Rad Neta')
            ax5.fill_between(meteod.index, meteod['R_Neta_Avg'].min(), meteod['R_Neta_Avg'].max(), where=(meteod['R_Neta_Avg'] < 0), alpha=0.5,color=(232/255, 222/255, 164/255, 0.5))
            ax5.grid()
            plt.savefig(currFolder+'/DatosRaw.png')
            plt.close()

sys.exit()

#crea una figura
fig = plt.figure(figsize=(10,10))
#representa la Yv y Ypred
plt.scatter(range(len(Yv)),Yv,color='blue',label='Datos de validación',s=10)
plt.scatter(range(len(Ypred)),Ypred,color='red',label='Datos predichos',s=5)
#agrega una leyenda
plt.legend()
#agrega una etiqueta a los ejes
plt.xlabel('Muestra número')
plt.ylabel('Estado hídrico')

#crea otra figura
fig2 = plt.figure(figsize=(10,10))
#representa el error entre Yv y Ypred
plt.plot(range(len(Yv)),Yv-Ypred,color='blue',label='Error')
#agrega una etiqueta a los ejes
plt.xlabel('Muestra número')
plt.ylabel('Error cometido')
plt.show()



# guarda res en un archivo csv
res.to_csv('resClasPCALDAMeteo'+year_train+year_data+sufix+'.csv')
# crea un dataframe con la fila estado de res
res_estado = res.loc['estado',:]
res_valdata = res.loc['valdata',:]

# unstackea res_estado
res_estado = res_estado.unstack(level=0)
# unstackea res_valdata
res_valdata = res_valdata.unstack(level=0)

res_estado=res_estado.astype(np.float64)
res_valdata=res_valdata.astype(np.float64)

#renombra el indice de res_estado a Fecha
res_estado.index = res_estado.index.rename('Fecha')
#renombra el indice de res_valdata a Fecha
res_valdata.index = res_valdata.index.rename('Fecha')

# convierte el indice de res_estado a datetime
res_estado.index = pd.to_datetime(res_estado.index)
# convierte el indice de res_valdata a datetime
res_valdata.index = pd.to_datetime(res_valdata.index)
