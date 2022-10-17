import sys
from time import time
from matplotlib.markers import MarkerStyle
import matplotlib
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import time
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics

year_train="2014"
year_data="2019"
nDiasMax=7
fltp=4

#matplotlib.use("Agg")

#pd.set_option('display.max_rows', None)

# Carga de datos de entrenamiento
dfT = pd.read_csv("rawMinutales"+year_train+".csv",na_values='.')
dfT.loc[:,"Fecha"]=pd.to_datetime(dfT.loc[:,"Fecha"])# Fecha como datetime
dfT=dfT.drop_duplicates(subset="Fecha")
dfT.dropna(subset = ["Fecha"], inplace=True)
dfT=dfT.set_index("Fecha")
dfT=dfT.apply(pd.to_numeric, errors='coerce')

# separa dfT en tdv y ltp en función del principio del nombre de cada columna y guarda el resto en meteoT
tdvT = dfT.loc[:,dfT.columns.str.startswith('TDV')]
ltpT = dfT.loc[:,dfT.columns.str.startswith('LTP')]
meteoT = dfT.drop(dfT.columns[dfT.columns.str.startswith('TDV')], axis=1)
meteoT = meteoT.drop(meteoT.columns[meteoT.columns.str.startswith('LTP')], axis=1)

# Carga de datos de predicción
dfP = pd.read_csv("rawMinutales"+year_data+".csv",na_values='.')
dfP.loc[:,"Fecha"]=pd.to_datetime(dfP.loc[:,"Fecha"])# Fecha como datetime
dfP=dfP.drop_duplicates(subset="Fecha")
dfP.dropna(subset = ["Fecha"], inplace=True)
dfP=dfP.set_index("Fecha")
dfP=dfP.apply(pd.to_numeric, errors='coerce')

# separa dfP en tdv y ltp en función del principio del nombre de cada columna y guarda el resto en meteoP
tdvP = dfP.loc[:,dfP.columns.str.startswith('TDV')]
ltpP = dfP.loc[:,dfP.columns.str.startswith('LTP')]
meteoP = dfP.drop(dfP.columns[dfP.columns.str.startswith('TDV')], axis=1)
meteoP = meteoP.drop(meteoP.columns[meteoP.columns.str.startswith('LTP')], axis=1)

# Carga datos de validacion
valdatapd=pd.read_csv("validacion"+year_data+".csv")
valdatapd.dropna(inplace=True)
valdatapd['Fecha'] = pd.to_datetime(valdatapd['Fecha'])
valdatapd.set_index('Fecha',inplace=True)

# Carga datos de resultados de entrenamiento
trdatapd=pd.read_csv("validacion"+year_train+".csv")
trdatapd.dropna(inplace=True)
trdatapd['Fecha'] = pd.to_datetime(trdatapd['Fecha'])
trdatapd.set_index('Fecha',inplace=True)

# elimina los valores NaN de ltp
ltpP = ltpP.dropna(axis=1,how='all')
ltpT = ltpT.dropna(axis=1,how='all')
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

# duplica el índice de signos pn como una columna más en signos_pn
signos_pnP['Hora'] = signos_pnP.index
signos_pnT['Hora'] = signos_pnT.index
# recorta signos pn al último valor de cada día
signos_pnP = signos_pnP.resample('D').last()
signos_pnT = signos_pnT.resample('D').last()

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
# recorta ltp a los tramos de 6 a 18 de hora_norm
ltpP = ltpP.loc[ltpP['Hora_norm']>=6,:]
ltpT = ltpT.loc[ltpT['Hora_norm']>=6,:]
ltpPbase = ltpP.loc[ltpP['Hora_norm']<=18,:]
ltpTbase = ltpT.loc[ltpT['Hora_norm']<=18,:]

valdatapd.index = valdatapd.index.strftime('%Y-%m-%d')
trdatapd.index = trdatapd.index.strftime('%Y-%m-%d')

valdatapd_orig = valdatapd.copy()
trdatapd_orig = trdatapd.copy()

# crea un array para guardar los valores del porcentaje de acierto
acierto = np.zeros(nDiasMax)
for nDias in range(nDiasMax):
    # crea ltp_1 copiando ltp
    ltpP_1 = ltpPbase.copy()
    ltpT_1 = ltpTbase.copy()
    ltpP = ltpPbase.copy()
    ltpT = ltpTbase.copy()

    valdatapd=valdatapd_orig.copy()
    trdatapd=trdatapd_orig.copy()

    for i in range(nDias+1):
        # resta 24 a Hora_norm en ltp_1
        ltpP_1.loc[:,'Hora_norm'] = ltpP_1.loc[:,'Hora_norm'] - 24
        ltpT_1.loc[:,'Hora_norm'] = ltpT_1.loc[:,'Hora_norm'] - 24

        # suma 1 dia al índice de ltp_1
        ltpP_1.index = ltpP_1.index + pd.Timedelta(days=1)
        ltpT_1.index = ltpT_1.index + pd.Timedelta(days=1)
        # combina ltp, ltp_1 y ltp_2
        ltpP = pd.concat([ltpP,ltpP_1])
        ltpT = pd.concat([ltpT,ltpT_1])

    # añade la hora normalizada al índice de ltp
    ltpP.index = [ltpP.index.strftime('%Y-%m-%d'),ltpP['Hora_norm']]
    ltpT.index = [ltpT.index.strftime('%Y-%m-%d'),ltpT['Hora_norm']]


    #obtiene el índice interseccion de valdatapd y el primer nivel del índice de ltp
    ltpPdates = ltpP.index.get_level_values(0)
    ltpTdates = ltpT.index.get_level_values(0)

    valdatapd_ltp = valdatapd.index.intersection(ltpPdates)
    trdatapd_ltp = trdatapd.index.intersection(ltpTdates)

    # elimina los valores de ltp que no estén en valdatapd
    ltpv = ltpP.loc[valdatapd_ltp,valdatapd.columns]
    # elimina los valores de ltp que no estén en trdatapd
    ltpt = ltpT.loc[trdatapd_ltp,trdatapd.columns]

    # unstackea ltpv
    ltpv = ltpv.unstack(level=0)
    # unstackea ltpt
    ltpt = ltpt.unstack(level=0)

    # sustituye el indice de ltpv por el valor de la columna Hora de ltpv_index
    ltpv_index_float=pd.Int64Index(np.floor(ltpv.index*1000000000))

    # sustituye el indice de ltpt por el valor de la columna Hora de ltpt_index
    ltpt_index_float=pd.Int64Index(np.floor(ltpt.index*1000000000))

    # convierte el indice a datetime para ajustar frecuencias
    ltpv.index = pd.to_datetime(ltpv_index_float)
    ltpv=ltpv.resample(str(fltp)+'S').mean()
    ltpt.index = pd.to_datetime(ltpt_index_float)
    ltpt=ltpt.resample(str(fltp)+'S').mean()

    ltpv=ltpv.dropna(how='all')
    ltpt=ltpt.dropna(how='all')

    # conserva las filas del menor de los dos índices
    ltpv = ltpv.loc[ltpv.index.isin(ltpt.index),:]
    ltpt = ltpt.loc[ltpt.index.isin(ltpv.index),:]

    # Crea una serie para restaurar el índice
    norm_index=pd.Series(np.arange(0,len(ltpv.index)))
    # Ajusta el índice de ltpv a la serie
    ltpv.index=norm_index
    # Ajusta el índice de ltpt a la serie
    ltpt.index=norm_index

    #muestra todas las filas de los dataframes
    pd.set_option('display.max_rows', None)

    # crea los valores X e y para el modelo
    Xtr=ltpt.values.transpose()
    Ytr=trdatapd.unstack().values
    Xv=ltpv.values.transpose()
    Yv=valdatapd.unstack().values

    # crea el modelo
    clf = sklda.LinearDiscriminantAnalysis(solver='svd')#,shrinkage='auto')
    # entrena el modelo
    clf.fit(Xtr,Ytr)
    # predice los valores de Yv
    Ypred=clf.predict(Xv)


    # calcula la matriz de confusion
    confusion_matrix = skmetrics.confusion_matrix(Yv, Ypred)
    print('Numero de dias: '+str(nDias+1))
    print('Matriz de confusion:')
    print(confusion_matrix)

    # calcula el porcentaje de acierto
    accuracy = skmetrics.balanced_accuracy_score(Yv, Ypred)
    print('Porcentaje de acierto: '+str(accuracy*100)+'%')
    acierto[nDias]=accuracy

# grafica el porcentaje de acierto
plt.plot(acierto)
plt.xlabel('Numero de dias en memoria')
plt.ylabel('Porcentaje de acierto')
plt.show()