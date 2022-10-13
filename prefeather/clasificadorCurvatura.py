import sys
from time import time
from matplotlib.markers import MarkerStyle
import matplotlib
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import time

year_train="2014"
year_data="2019"

matplotlib.use("Agg")
# carga de secuencias tipo
sec=pd.read_csv("ltp_media"+year_train+".csv",index_col=0)
# asegura que el tipo de dato sea float
sec=sec.astype(float)
# recorta entre el índice de valor 6 y el de valor 18
sec=sec.loc[6:18]

# Carga de datos
dfT = pd.read_csv("rawMinutales"+year_data+".csv",na_values='.')
dfT.loc[:,"Fecha"]=pd.to_datetime(dfT.loc[:,"Fecha"])# Fecha como datetime
dfT=dfT.drop_duplicates(subset="Fecha")
dfT.dropna(subset = ["Fecha"], inplace=True)
dfT=dfT.set_index("Fecha")
dfT=dfT.apply(pd.to_numeric, errors='coerce')

# separa dfT en tdv y ltp en función del principio del nombre de cada columna y guarda el resto en meteo
tdv = dfT.loc[:,dfT.columns.str.startswith('TDV')]
ltp = dfT.loc[:,dfT.columns.str.startswith('LTP')]
meteo = dfT.drop(dfT.columns[dfT.columns.str.startswith('TDV')], axis=1)
meteo = meteo.drop(meteo.columns[meteo.columns.str.startswith('LTP')], axis=1)

# Carga datos de  validacion
valdatapd=pd.read_csv("validacion"+year_data+".csv")
valdatapd.dropna(inplace=True)
valdatapd['Fecha'] = pd.to_datetime(valdatapd['Fecha'])
valdatapd.set_index('Fecha',inplace=True)

# elimina los valores NaN de ltp
ltp = ltp.dropna(axis=1,how='all')
# rellena los valores NaN de ltp con el valor anterior
ltp = ltp.fillna(method='ffill')
# rellena los valores NaN de ltp con el valor siguiente
ltp = ltp.fillna(method='bfill')

# aplica un filtro de media móvil a ltp
ltp = ltp.rolling(window=240,center=True).mean()

# calcula el valor medio de ltp para cada dia
ltp_medio = ltp.groupby(ltp.index.date).mean()
# calcula el valor de la desviación estándar de ltp para cada dia
ltp_std = ltp.groupby(ltp.index.date).std()
# cambia el índice a datetime
ltp_medio.index = pd.to_datetime(ltp_medio.index)
ltp_std.index = pd.to_datetime(ltp_std.index)

# remuestrea ltp_medio y ltp_std a minutal
ltp_medio = ltp_medio.resample('T').pad()
ltp_std = ltp_std.resample('T').pad()

# normaliza ltp para cada dia
ltp = (ltp - ltp_medio) / ltp_std

# calcula la pendiente de ltp
ltp_diff = ltp.diff()

# aplica un filtro de media móvil a ltp_diff
ltp_diff = ltp_diff.rolling(window=240,center=True).mean()

# obtiene todos los cambios de signo de R_Neta_Avg en el dataframe meteo
signos = np.sign(meteo.loc[:,meteo.columns.str.startswith('R_Neta_Avg')]).diff()
# obtiene los cambios de signo de positivo a negativo
signos_pn = signos<0
# elimina los valores falsos (que no sean cambios de signo)
signos_pn = signos_pn.replace(False,np.nan).dropna()
# obtiene los cambios de signo de negativo a positivo
signos_np = signos>0
# elimina los valores falsos (que no sean cambios de signo)
signos_np = signos_np.replace(False,np.nan).dropna()

# duplica el índice de signos np como una columna más en signos_np
signos_np['Hora'] = signos_np.index
# recorta signos np al primer valor de cada día
signos_np = signos_np.resample('D').first()

# duplica el índice de signos pn como una columna más en signos_pn
signos_pn['Hora'] = signos_pn.index
# recorta signos pn al último valor de cada día
signos_pn = signos_pn.resample('D').last()

# recoge los valores del índice de ltp donde la hora es 00:00
ltp_00 = ltp.index.time == time.min
# recoge los valores del índice de ltp donde la hora es la mayor de cada día
ltp_23 = ltp.index.time == time(23,59)

# crea una columna en ltp que vale 0 a las 00:00
ltp.loc[ltp_00,'Hora_norm'] = 0
# iguala Hora_norm a 6 en los índices de signos np
ltp.loc[signos_np['Hora'],'Hora_norm'] = 6
# iguala Hora_norm a 18 en los índices de signos pn
ltp.loc[signos_pn['Hora'],'Hora_norm'] = 18
# iguala Hora_norm a 24 en el último valor de cada día
ltp.loc[ltp_23,'Hora_norm'] = 24
# iguala el valor en la última fila de Hora_norm a 24
ltp.loc[ltp.index[-1],'Hora_norm'] = 24
# interpola Hora_norm en ltp
ltp.loc[:,'Hora_norm'] = ltp.loc[:,'Hora_norm'].interpolate()
# recorta ltp a los tramos de 6 a 18 de hora_norm
ltp = ltp.loc[ltp['Hora_norm']>=6,:]
ltp = ltp.loc[ltp['Hora_norm']<=18,:]

# remuestrea valdatapd a minutal con el valor de cada día
valdatapdT = valdatapd.resample('T').pad()
# obtiene el índice interseccion de valdatapd y ltp
valdatapd_ltp = valdatapdT.index.intersection(ltp.index)
# recorta valdatapd según el índice interseccion
valdatapdT = valdatapdT.loc[valdatapd_ltp,:]

# elimina los valores de ltp que no estén en valdatapd
ltpv = ltp.loc[valdatapd_ltp,valdatapdT.columns]
# elimina los valores de diff que no estén en valdatapd
ltp_diffv = ltp_diff.loc[valdatapd_ltp,valdatapdT.columns]


# separa el índice de ltpv en dos columnas de fecha y hora y añade una columna con el valor de la hora normalizada
ltpv_index = ltpv.index.strftime('%Y-%m-%d %H:%M').str.split(' ',expand=True)
# aplica el nuevo indice a ltpv
ltpv.index = ltpv_index

# separa el índice de ltp_diffv en dos columnas de fecha y hora
ltp_diffv_index = ltp_diffv.index.strftime('%Y-%m-%d %H:%M').str.split(' ',expand=True)
# aplica el nuevo indice a ltp_diffv
ltp_diffv.index = ltp_diffv_index

# unstackea ltpv
ltpv = ltpv.unstack(level=0)
# crea un dataframe para generar el indice de ltpv
ltpv_index = pd.DataFrame(ltpv.index.get_level_values(0))
# separa ltpv_index en dos columnas de hora y minuto
ltpv_index = ltpv_index[0].str.split(':',expand=True)
#divide los minutos por 60
ltpv_index[1] = ltpv_index[1].astype(int)/60
# suma los minutos y horas
ltpv_index['Hora'] = ltpv_index[1] + ltpv_index[0].astype(int)
# sustituye el indice de ltpv por el valor de la columna Hora de ltpv_index
ltpv.index = ltpv_index['Hora']
ltpv_index_float=pd.Int64Index(np.floor(ltpv.index*1000000000))
# convierte el indice a datetime para ajustar frecuencias
ltpv.index = pd.to_datetime(ltpv_index_float)
ltpv=ltpv.resample('0.1S').mean()
# Crea una serie de 0.01 a 24 para restaurar el índice
norm_index=pd.Series(np.arange(6,18,0.1))
# Ajusta el índice de ltpv a la serie de 0.01 a 24
ltpv.index=norm_index

# convierte las columnas LTP_1, LTP_2 y LTP_3 de sec a vectores de numpy
ltp_1 = sec['LTP_1'].values
ltp_2 = sec['LTP_2'].values
ltp_3 = sec['LTP_3'].values

# obtiene la aproximación por mínimos cuadrados de la parábola que mejor se ajusta a los valores de ltp_1
p1 = np.polyfit(range(len(ltp_1)),ltp_1,2)
# obtiene la aproximación por mínimos cuadrados de la parábola que mejor se ajusta a los valores de ltp_2
p2 = np.polyfit(range(len(ltp_2)),ltp_2,2)
# obtiene la aproximación por mínimos cuadrados de la parábola que mejor se ajusta a los valores de ltp_3
p3 = np.polyfit(range(len(ltp_3)),ltp_3,2)

#crea un dataframe con los datos de valdatapd y la forma de ltpv
res = pd.DataFrame()
res['valdata'] = valdatapd.unstack()
res = res.transpose()

# recorre las columnas de ltpv en un bucle
for i in range(len(ltpv.columns)):
    #convierte la columna de ltpv en un vector de numpy
    ltpv_col = ltpv.iloc[:,i].values
    # obtiene la aproximación por mínimos cuadrados de la parábola que mejor se ajusta a los valores de ltpv_col
    p = np.polyfit(range(len(ltpv_col)),ltpv_col,2)
    # añade a res_valdatapd el primer valor de la parábola
    res.loc['lsqr2_1',ltpv.columns[i]] = p[0]
    # añade a res_valdatapd el segundo valor de la parábola
    res.loc['lsqr2_2',ltpv.columns[i]] = p[2]

    # añade un indice a la columna correspondiente de res con el valor 1, 2 o 3 según el valor de la curvatura
    if p[2] < -1:
        res.loc['estado',ltpv.columns[i]] = 1
    elif p[2] > 1:
        res.loc['estado',ltpv.columns[i]] = 3
    else:
        res.loc['estado',ltpv.columns[i]] = 2

res=res.dropna(axis=1,how='any')
print(res)

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

#plotea los datos de res_valdatapd con res.loc['valdata'] en el eje x y res.loc['lsqr2'] en el eje y usando puntos y color en función de res.loc['valdata']
plt.scatter(x=res.loc['lsqr2_2'],y=res.loc['lsqr2_1'],s=1,c=cols,alpha=0.5)
#plt.plot(res.loc['valdata'],res.loc['lsqr2'], 'o')

# añade nombres a los ejes
plt.ylabel('Término cuadrático de la aproximación por mínimos cuadrados')
plt.xlabel('Término independiente de la aproximación por mínimos cuadrados')
# guarda la figura f en la carpeta figurasClasificadorConvolucion con el nombre de la columna de res_estado, indice de res_estado y valor de res_estado y valdatapd
f.savefig('figurasClasificadorCurvatura/blobs.png')
# cierra la figura f
plt.close(f)

# guarda res en un archivo csv
res.to_csv('resClasErr.csv')
# crea un dataframe con la fila estado de res
res_estado = res.loc['estado',:]

# unstackea res_estado
res_estado = res_estado.unstack(level=0)

res_estado=res_estado.astype(np.float64)

#renombra el indice de res_estado a Fecha
res_estado.index = res_estado.index.rename('Fecha')
# convierte el indice de res_estado a datetime
res_estado.index = pd.to_datetime(res_estado.index)
# elimina las filas no comunes de res_estado y valdatapd
# obtiene el indice de intersección de res_estado y valdatapd
res_estado_valdatapd = res_estado.index.intersection(valdatapd.index)
# elimina las filas no comunes de res_estado y valdatapd
res_estado = res_estado.loc[res_estado_valdatapd,:]
valdatapd = valdatapd.loc[res_estado_valdatapd,:]

# crea un dataframe con la matriz de confusión
conf_matrix=pd.DataFrame(index=['est LTP_1','est LTP_2','est LTP_3'],columns=['LTP_1','LTP_2','LTP_3'])
# inicializa la matriz de confusión con 0
conf_matrix.iloc[:,:]=0

# recorre las filas de res_estado
for i in range(len(res_estado)):
    # recorre las columnas de res_estado
    for j in range(len(res_estado.columns)):
        conf_matrix.iloc[int(res_estado.iloc[i,j]-1),int(valdatapd.iloc[i,j]-1)] += 1

        
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
        f.savefig('figurasClasificadorCurvatura/'+str(int(res_estado.iloc[i,j]))+'_'+str(int(valdatapd.iloc[i,j]))+'/'+res_estado.columns[j]+'_'+res_estado.index[i].strftime('%Y-%m-%d')+'.png')
        # cierra la figura f
        plt.close(f)

print(conf_matrix)