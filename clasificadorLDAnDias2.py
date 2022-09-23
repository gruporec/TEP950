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
fltpMax=4
nDias=5

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
# recorta ltp a los tramos de 6 a 18 de hora_norm
ltpP = ltpP.loc[ltpP['Hora_norm']>=6,:]
ltpT = ltpT.loc[ltpT['Hora_norm']>=6,:]
ltpP = ltpP.loc[ltpP['Hora_norm']<=18,:]
ltpT = ltpT.loc[ltpT['Hora_norm']<=18,:]
# crea ltp_1 copiando ltp
ltpP_1 = ltpP.copy()
ltpT_1 = ltpT.copy()

for i in range(nDias):
    print('Dia '+str(i+1))
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

valdatapd.index = valdatapd.index.strftime('%Y-%m-%d')
trdatapd.index = trdatapd.index.strftime('%Y-%m-%d')

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

ltpvini=ltpv.copy()
ltptini=ltpt.copy()

res=np.zeros(int(fltpMax*10))
fres=np.zeros(int(fltpMax*10))

for fltpi in range(1,int(fltpMax*10+1)):
    fltp=fltpi/10
    ltpv=ltpvini.copy()
    ltpt=ltptini.copy()

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

    # conserva los valores de 1970-01-01 00:00:06.000 a 1970-01-01 00:00:17.900
    ltpv = ltpv.loc[ltpv.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
    ltpt = ltpt.loc[ltpt.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
    ltpv = ltpv.loc[ltpv.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]
    ltpt = ltpt.loc[ltpt.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]

    # Crea una serie para restaurar el índice
    norm_index=pd.Series(np.arange(0,len(ltpv.index)))
    # Ajusta el índice de ltpv a la serie
    ltpv.index=norm_index
    # Ajusta el índice de ltpt a la serie
    ltpt.index=norm_index

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
    print(fltp)

    print(confusion_matrix)

    # calcula el porcentaje de acierto
    accuracy = skmetrics.balanced_accuracy_score(Yv, Ypred)
    print('Porcentaje de acierto: '+str(accuracy*100)+'%')
    res[fltpi-1]=accuracy
    fres[fltpi-1]=fltp



# grafica el resultado
plt.plot(fres,res)
plt.xlabel('Periodo de muestreo (h)')
plt.ylabel('Porcentaje de acierto')
plt.show()
sys.exit()

res=pd.DataFrame()
res['valdata']=valdatapd.unstack()
res['estado']=pd.DataFrame(Ypred,index=res.index)
res = res.transpose()

# abre una figura f
f = plt.figure()
# Function to map the colors as a list from the input list of x variables
def pltcolor(lst):
    cols=[]
    for l in lst:
        if l==1:
            cols.append('red')
        elif l==2:
            cols.append('blue')
        else:
            cols.append('green')
    return cols
# Create the colors list using the function above
cols=pltcolor(res.loc['valdata'].values)

# guarda res en un archivo csv
res.to_csv('resClasLDA'+year_train+year_data+'.csv')
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

# crea un dataframe con la matriz de confusión
conf_matrix=pd.DataFrame(index=['est LTP_1','est LTP_2','est LTP_3'],columns=['LTP_1','LTP_2','LTP_3'])
# inicializa la matriz de confusión con 0
conf_matrix.iloc[:,:]=0

# carga de secuencias tipo
sec=pd.read_csv("ltp_media"+year_train+".csv",index_col=0)
# asegura que el tipo de dato sea float
sec=sec.astype(float)
# recorta entre el índice de valor 6 y el de valor 18
sec=sec.loc[6:18]

# convierte las columnas LTP_1, LTP_2 y LTP_3 de sec a vectores de numpy
ltp_1 = sec['LTP_1'].values
ltp_2 = sec['LTP_2'].values
ltp_3 = sec['LTP_3'].values

#si no existen, crea las carpetas 1_1 a 3_3 en figurasClasificadorLDA
if not os.path.exists('figurasClasificadorLDA'+year_train+year_data):
    os.makedirs('figurasClasificadorLDA'+year_train+year_data)
    os.makedirs('figurasClasificadorLDA'+year_train+year_data+'/1_1')
    os.makedirs('figurasClasificadorLDA'+year_train+year_data+'/1_2')
    os.makedirs('figurasClasificadorLDA'+year_train+year_data+'/1_3')
    os.makedirs('figurasClasificadorLDA'+year_train+year_data+'/2_1')
    os.makedirs('figurasClasificadorLDA'+year_train+year_data+'/2_2')
    os.makedirs('figurasClasificadorLDA'+year_train+year_data+'/2_3')
    os.makedirs('figurasClasificadorLDA'+year_train+year_data+'/3_1')
    os.makedirs('figurasClasificadorLDA'+year_train+year_data+'/3_2')
    os.makedirs('figurasClasificadorLDA'+year_train+year_data+'/3_3')

# recorre las filas de res_estado
for i in range(len(res_estado)):
    # recorre las columnas de res_estado
    for j in range(len(res_estado.columns)):
        conf_matrix.iloc[int(res_estado.iloc[i,j]-1),int(res_valdata.iloc[i,j]-1)] += 1
        
        # abre una figura f
        f = plt.figure()
        # plotea en f el vector de referencia
        plt.plot(ltp_1,'r--')
        plt.plot(ltp_2,'g--')
        plt.plot(ltp_3,'b--')
        # plotea en f el vector de estimación
        plt.plot(ltpv.loc[:,res_estado.columns[j]].loc[:,res_estado.index[i].strftime('%Y-%m-%d')].values)
        # añade una leyenda a f
        plt.legend(['LTP_1','LTP_2','LTP_3','valores medidos'])
        # guarda la figura f en la carpeta figurasClasificadorConvolucion con el nombre de la columna de res_estado, indice de res_estado y valor de res_estado y valdatapd
        f.savefig('figurasClasificadorLDA'+year_train+year_data+'/'+str(int(res_estado.iloc[i,j]))+'_'+str(int(res_valdata.iloc[i,j]))+'/'+res_estado.columns[j]+'_'+res_estado.index[i].strftime('%Y-%m-%d')+'.png')
        # cierra la figura f
        plt.close(f)

print(conf_matrix)