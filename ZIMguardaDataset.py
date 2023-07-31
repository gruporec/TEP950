import sys
from matplotlib.markers import MarkerStyle
import matplotlib
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import time
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp
import isadoralib as isl
import time as tm
import seaborn as sns
import scipy.optimize as opt

sns.set(rc={'figure.figsize':(11.7,8.27)})

year_datas=["2014","2015","2016","2019"]
sufix="rht"

n_dias_print=5

ltpitems=80
meteoitems=4

#pd.set_option('display.max_rows', None)

# crea un dataframe vacío para guardar los datos de X y otro para Y
savedfX = pd.DataFrame()
savedfY = pd.DataFrame()

for year_data in year_datas:
    # Carga de datos de predicción
    tdvP,ltpP,meteoP,valdatapd=isl.cargaDatos(year_data,sufix)

    # guarda la información raw para plots
    ltpPlot = ltpP.copy()
    meteoPlot = meteoP.copy()

    # añade meteo a ltpP
    ltpP=ltpP.join(meteoP)

    # elimina los valores NaN de ltp
    ltpP = ltpP.dropna(axis=1,how='all')
    # rellena los valores NaN de ltp con el valor anterior
    ltpP = ltpP.fillna(method='ffill')
    # rellena los valores NaN de ltp con el valor siguiente
    ltpP = ltpP.fillna(method='bfill')

    # aplica un filtro de media móvil a ltp
    ltpP = ltpP.rolling(window=240,center=True).mean()

    # calcula el valor medio de ltp para cada dia
    ltp_medioP = ltpP.groupby(ltpP.index.date).mean()

    # calcula el valor de la desviación estándar de ltp para cada dia
    ltp_stdP = ltpP.groupby(ltpP.index.date).std()

    # cambia el índice a datetime
    ltp_medioP.index = pd.to_datetime(ltp_medioP.index)
    ltp_stdP.index = pd.to_datetime(ltp_stdP.index)

    # remuestrea ltp_medio y ltp_std a minutal
    ltp_medioP = ltp_medioP.resample('T').pad()
    ltp_stdP = ltp_stdP.resample('T').pad()

    # normaliza ltp para cada dia

    ltpP = (ltpP - ltp_medioP) / ltp_stdP

    # obtiene todos los cambios de signo de R_Neta_Avg en el dataframe meteo
    signosP = np.sign(meteoP.loc[:,meteoP.columns.str.startswith('R_Neta_Avg')]).diff()
    # obtiene los cambios de signo de positivo a negativo
    signos_pnP = signosP<0
    # elimina los valores falsos (que no sean cambios de signo)
    signos_pnP = signos_pnP.replace(False,np.nan).dropna()
    # obtiene los cambios de signo de negativo a positivo
    signos_npP = signosP>0
    # elimina los valores falsos (que no sean cambios de signo)
    signos_npP = signos_npP.replace(False,np.nan).dropna()

    # duplica el índice de signos np como una columna más en signos_np
    signos_npP['Hora'] = signos_npP.index
    # recorta signos np al primer valor de cada día
    signos_npP = signos_npP.resample('D').first()

    #elimina los dias en los que no haya cambio de signo
    signos_npP=signos_npP.dropna()

    # duplica el índice de signos pn como una columna más en signos_pn
    signos_pnP['Hora'] = signos_pnP.index
    # recorta signos pn al último valor de cada día
    signos_pnP = signos_pnP.resample('D').last()

    #elimina los días en los que no haya cambio de signo
    signos_pnP = signos_pnP.dropna()

    # recoge los valores del índice de ltp donde la hora es 00:00
    ltp_00P = ltpP.index.time == time.min
    # recoge los valores del índice de ltp donde la hora es la mayor de cada día
    ltp_23P = ltpP.index.time == time(23,59)

    # crea una columna en ltp que vale 0 a las 00:00
    ltpP.loc[ltp_00P,'Hora_norm'] = 0
    # iguala Hora_norm a 6 en los índices de signos np
    ltpP.loc[signos_npP['Hora'],'Hora_norm'] = 6
    # iguala Hora_norm a 18 en los índices de signos pn
    ltpP.loc[signos_pnP['Hora'],'Hora_norm'] = 18
    # iguala Hora_norm a 24 en el último valor de cada día
    ltpP.loc[ltp_23P,'Hora_norm'] = 24
    # iguala el valor en la última fila de Hora_norm a 24
    ltpP.loc[ltpP.index[-1],'Hora_norm'] = 24
    # interpola Hora_norm en ltp
    ltpP.loc[:,'Hora_norm'] = ltpP.loc[:,'Hora_norm'].interpolate()

    # almacena los valores antes de recortar
    ltpPBase=ltpP

    # recorta ltp a los tramos de 6 a 18 de hora_norm
    ltpP = ltpP.loc[ltpP['Hora_norm']>=6,:]
    ltpP = ltpP.loc[ltpP['Hora_norm']<=18,:]


    # añade la hora normalizada al índice de ltp
    ltpP.index = [ltpP.index.strftime('%Y-%m-%d'),ltpP['Hora_norm']]

    #crea el índice de ltpPBase
    ltpPBase['Hora_norm']=ltpPBase['Hora_norm'].apply(pd.to_timedelta,unit='H')
    ltpPBase['dia_norm'] = ltpPBase.index.strftime('%Y-%m-%d')
    ltpPBase.index = [ltpPBase['dia_norm'].apply(pd.to_datetime,format='%Y-%m-%d'),ltpPBase['Hora_norm']]
    ltpPBase=ltpPBase.drop('Hora_norm',axis=1)
    ltpPBase=ltpPBase.drop('dia_norm',axis=1)
    ltpPBase=ltpPBase.unstack(level=0)

    valdatapd.index = valdatapd.index.strftime('%Y-%m-%d')

    #obtiene el índice interseccion de valdatapd y el primer nivel del índice de ltp
    ltpPdates = ltpP.index.get_level_values(0)

    valdatapd_ltp = valdatapd.index.intersection(ltpPdates)

    # vuelve a separar los valores de meteo de ltp
    meteoP_norm=ltpP.drop(ltpP.columns[ltpP.columns.str.startswith('LTP')], axis=1)

    # elimina los valores de ltp que no estén en valdatapd
    ltpv = ltpP.loc[valdatapd_ltp,valdatapd.columns]

    # unstackea meteoP_norm
    meteoP_norm = meteoP_norm.unstack(level=0)

    # unstackea ltpv
    ltpv = ltpv.unstack(level=0)

    # crea un índice para ajustar frecuencias
    ltpv_index_float=pd.Int64Index(np.floor(ltpv.index*1000000000))
    meteoP_index_float=pd.Int64Index(np.floor(meteoP_norm.index*1000000000))

    ltpv.index = pd.to_datetime(ltpv_index_float)
    meteoP_norm.index = pd.to_datetime(meteoP_index_float)

    ltpv_orig=ltpv.copy()
    meteoP_norm_orig=meteoP_norm.copy()

    fltp=12/ltpitems
    if meteoitems>0:
        fmeteo=12/meteoitems
    else:
        fmeteo=0
    # convierte el indice a datetime para ajustar frecuencias
    ltpv=ltpv_orig.resample(str(int(fltp*1000))+'L').mean()
    if meteoitems>0:
        meteoP_norm=meteoP_norm_orig.resample(str(int(fmeteo*1000))+'L').mean()

    # conserva los valores de 1970-01-01 00:00:06.000 a 1970-01-01 00:00:17.900
    ltpv = ltpv.loc[ltpv.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
    ltpv = ltpv.loc[ltpv.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]

    if meteoitems>0:
        meteoP_norm = meteoP_norm.loc[meteoP_norm.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
        meteoP_norm = meteoP_norm.loc[meteoP_norm.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]


    # Crea una serie para restaurar el índice
    norm_index=pd.Series(np.arange(6,18,fltp))
    # Ajusta el índice de ltpv a la serie
    ltpv.index=norm_index

    if meteoitems>0:
        # Crea una serie para restaurar el índice
        norm_index=pd.Series(np.arange(6,18,fmeteo))
        #recorta norm_index para que coincida con el tamano de meteoP_norm si se ha producido un desajuste al calcular el dataframe
        norm_index=norm_index.loc[norm_index.index<len(meteoP_norm)]
        # Ajusta el índice de meteoP_norm a la serie
        meteoP_norm.index=norm_index

        # dropea la columna Hora_norm de meteo
        meteoP_norm = meteoP_norm.drop('Hora_norm',axis=1)

        # stackea meteoP_norm y meteoT_norm
        meteoP_norm = meteoP_norm.stack(level=0)

        #intercambia los niveles del índice de meteo
        meteoP_norm.index = meteoP_norm.index.swaplevel(0,1)

        meteoP_norm=meteoP_norm.dropna(axis=1,how='all')

        #combina los dos índices de meteo
        meteoP_norm.index = meteoP_norm.index.map('{0[1]}/{0[0]}'.format)

    else:
        meteoP_norm = pd.DataFrame()

    #crea un array de numpy en blanco
    array_ltpv=np.empty((len(ltpv)+len(meteoP_norm),0))

    #por cada elemento en el primer índice de columnas de ltp
    for i in ltpv.columns.levels[0]:
        ltpv_col=ltpv.loc[:,i]
        if meteoitems>0:
            # elimina los valores de meteo que no estén en ltp_col
            meteo_ltp = ltpv_col.columns.intersection(meteoP_norm.columns)
            meteoP_col = meteoP_norm.loc[:,meteo_ltp]

            # combina los valores de ltpv con los de meteo
            merge_ltp_meteo = pd.merge(ltpv.loc[:,i],meteoP_col,how='outer')
        else:
            merge_ltp_meteo = ltpv.loc[:,i]
        # añade la unión al array de numpy
        array_ltpv=np.append(array_ltpv,merge_ltp_meteo.values,axis=1)

    
    # print("ltpt")
    # print(ltpt)
    # print("meteo_ltp")
    # print(meteo_ltp)
    # print("meteoT_norm")
    # print(meteoT_norm)
    # print("ltpt_col")
    # print(ltpt_col)

    Xv=array_ltpv.transpose()
    Yv=valdatapd.unstack()

    # convierte Xv a dataframe recuperando el índice de Yv
    Xv=pd.DataFrame(Xv,index=Yv.index)

    # añade los dataframes de Xv e Yv a sus respectivos dataframes
    savedfX=pd.concat([savedfX,Xv])
    savedfY=pd.concat([savedfY,Yv])

#añade savedfY a savedfX como columna con el nombre 'Y', restando 1 para que los valores de Y estén entre 0 y 2
savedfX['Y']=savedfY-1

#crea un string con los dos últimos dígitos de cada año en year_datas
year_datas_str = ''.join(year_datas[-2:] for year_datas in year_datas)
# guarda savedfX en un csv con un nombre compuesto por 'db' seguido por las dos ultimas cifras de cada año en year_datas
savedfX.to_csv('db'+year_datas_str+'.csv')