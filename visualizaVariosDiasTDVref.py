import sys
from time import time
from matplotlib.markers import MarkerStyle
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import time
import isadoralib as isl

year = "2016"
nDias = 4
alphaValue = 0.05
# Ejecuta cargaRaw.py si no existe rawDiarios.csv o rawMinutales.csv
if not os.path.isfile("rawDiarios"+year+".csv") or not os.path.isfile("rawMinutales"+year+".csv"):
    os.system("python3 cargaRaw.py")

# Carga de datos
tdv,ltp,meteo,data=isl.cargaDatosTDV(year,"rht")

#elimina los datos nan de tdv
tdv = tdv.dropna()

#separa el índice de tdv en dos columnas de fecha y hora
tdv.index = pd.to_datetime(tdv.index)

#remuestrea tdv a 1 hora
tdv = tdv.resample('1H').mean()

tdv['Fecha'] = tdv.index.date
tdv['Hora'] = tdv.index.time

#convierte el índice de data a fecha
data.index = pd.to_datetime(data.index)
data.index = data.index.date

#cambia el índice de tdv por la columna de fecha
tdv = tdv.set_index('Fecha')

#convierte la columna de hora a un valor numérico de minutos teniendo en cuenta que es un objeto datetime.time
tdv['Hora'] = tdv['Hora'].apply(lambda x: x.hour*60 + x.minute)

#copia tdv en un dataframe nuevo al que añadir los datos de dias anteriores
tdv_prev = tdv.copy()
#por cada valor entre 1 y nDias, crea un dataframe temporal con los datos de tdv, restando 24*i horas al valor de la columna hora y añadiendo un dia a cada valor del índice
for i in range(1,nDias,1):
    tdv_temp = tdv.copy()
    #resta 24*i horas al valor de la columna hora
    tdv_temp['Hora'] = tdv_temp['Hora'] - 24*60*i
    #añade i días al índice
    tdv_temp.index = tdv_temp.index + pd.Timedelta(days=i)
    #añade el dataframe temporal al dataframe de dias anteriores
    tdv_prev = tdv_prev.append(tdv_temp)
#añade la columna hora al índice en un segundo nivel
tdv_prev = tdv_prev.set_index('Hora',append=True)
print('1')
#stackea las columnas de tdv_prev
tdv_prev = tdv_prev.stack()
#unstackea la columna de hora
tdv_prev = tdv_prev.unstack('Hora')
print('2')

#stackea las columnas de data
data = data.stack()
print('3')

#elimina los valores de data que no estén en tdv_prev
data = data[data.index.isin(tdv_prev.index)]
print('4')
#elimina los valores de tdv_prev que no estén en data
tdv_prev = tdv_prev[tdv_prev.index.isin(data.index)]
print('5')

# #calcula la media de cada fila de tdv_prev
# tdv_prev_mean = tdv_prev.mean(axis=1)
# #calcula la desviación estándar de cada fila de tdv_prev
# tdv_prev_std = tdv_prev.std(axis=1)

# print('6')
# #resta a cada fila su media
# tdv_prev = tdv_prev.sub(tdv_prev_mean,axis=0)
# #divide cada fila entre su desviación estándar
# tdv_prev = tdv_prev.div(tdv_prev_std,axis=0)

# # suma a cada fila su minimo
# tdv_prev = tdv_prev.sub(tdv_prev.min(axis=1),axis=0)
# #divide cada fila entre su maximo
# tdv_prev = tdv_prev.div(tdv_prev.max(axis=1),axis=0)
# #suma 1 a cada fila
# tdv_prev = tdv_prev.add(1,axis=0)

#unstackea el segundo nivel del índice de tdv_prev
tdv_prev = tdv_prev.unstack(1)
#stackea el primer nivel del índice de tdv_prev
tdv_prev = tdv_prev.stack(0)



#extrae la primera columna de tdv_prev que contiene la palabra "Control"
tdv_prev_ref = tdv_prev[tdv_prev.columns[tdv_prev.columns.str.contains('Control')]]
#limita ref a 1 columna
tdv_prev_ref = tdv_prev_ref.iloc[:,0]

#divide tdv_prev entre tdv_prev_ref
#tdv_prev = tdv_prev.div(tdv_prev_ref,axis=0)
#resta tdv_prev_ref a tdv_prev
tdv_prev = tdv_prev.sub(tdv_prev_ref,axis=0)

#unstackea el segundo nivel del índice de tdv_prev
tdv_prev = tdv_prev.unstack(1)
#stackea el primer nivel del índice de tdv_prev
tdv_prev = tdv_prev.stack(0)

#calcula la media de cada fila de tdv_prev
tdv_prev_mean = tdv_prev.mean(axis=1)
#calcula la desviación estándar de cada fila de tdv_prev
tdv_prev_std = tdv_prev.std(axis=1)

print('6')
#resta a cada fila su media
tdv_prev = tdv_prev.sub(tdv_prev_mean,axis=0)
# #divide cada fila entre su desviación estándar
# tdv_prev = tdv_prev.div(tdv_prev_std,axis=0)

print('7')

#añade la columna de data a tdv_prev
tdv_prev['data'] = data

print(tdv_prev)
print(data)

#separa la serie tdv_prev en 3 series según el valor de data entre 1 y 3
tdv_prev_1 = tdv_prev[tdv_prev['data']==1]
tdv_prev_2 = tdv_prev[tdv_prev['data']==2]
tdv_prev_3 = tdv_prev[tdv_prev['data']==3]

#drop de la columna data
tdv_prev_1 = tdv_prev_1.drop('data',axis=1)
tdv_prev_2 = tdv_prev_2.drop('data',axis=1)
tdv_prev_3 = tdv_prev_3.drop('data',axis=1)

#transpone los dataframes
tdv_prev_1 = tdv_prev_1.T
tdv_prev_2 = tdv_prev_2.T
tdv_prev_3 = tdv_prev_3.T

#Crea una figura con 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True,sharey=True)
#en el primer subplot, pinta tdv_prev_1 en negro con transparencia alphaValue dividido por el número de filas de tdv_prev_1, sin leyenda
ax1.plot(tdv_prev_1,ls='none',marker='o',color='black',alpha=alphaValue/(len(tdv_prev_1)/(60)),MarkerSize=1)
#en el segundo subplot, pinta tdv_prev_2 en negro con transparencia alphaValue dividido por el número de filas de tdv_prev_2, sin leyenda
ax2.plot(tdv_prev_2,ls='none',marker='o',color='black',alpha=alphaValue/(len(tdv_prev_2)/(60)),MarkerSize=1)
#en el tercer subplot, pinta tdv_prev_3 en negro con transparencia alphaValue dividido por el número de filas de tdv_prev_3, sin leyenda
ax3.plot(tdv_prev_3,ls='none',marker='o',color='black',alpha=alphaValue/(len(tdv_prev_3)/(60)),MarkerSize=1)

#crea una figura nueva
fig2 = plt.figure()

#en la nueva figura, dibuja las medias de las columnas de tdv_prev_1, tdv_prev_2 y tdv_prev_3 en rojo, azul y verde respectivamente
plt.plot(tdv_prev_1.mean(axis=1),color='red')
plt.plot(tdv_prev_2.mean(axis=1),color='blue')
plt.plot(tdv_prev_3.mean(axis=1),color='green')

#añade nuevas lineas discontinuas con la media más y menos 1 desviación estándar de las columnas de tdv_prev_1, tdv_prev_2 y tdv_prev_3 en rojo, azul y verde respectivamente
plt.plot(tdv_prev_1.mean(axis=1)+tdv_prev_1.std(axis=1),ls='--',color='red')
plt.plot(tdv_prev_1.mean(axis=1)-tdv_prev_1.std(axis=1),ls='--',color='red')
plt.plot(tdv_prev_2.mean(axis=1)+tdv_prev_2.std(axis=1),ls='--',color='blue')
plt.plot(tdv_prev_2.mean(axis=1)-tdv_prev_2.std(axis=1),ls='--',color='blue')
plt.plot(tdv_prev_3.mean(axis=1)+tdv_prev_3.std(axis=1),ls='--',color='green')
plt.plot(tdv_prev_3.mean(axis=1)-tdv_prev_3.std(axis=1),ls='--',color='green')

#añade lineas punteadas con el valor máximo y mínimo de las columnas de tdv_prev_1, tdv_prev_2 y tdv_prev_3 en rojo, azul y verde respectivamente
plt.plot(tdv_prev_1.max(axis=1),ls=':',color='red')
plt.plot(tdv_prev_1.min(axis=1),ls=':',color='red')
plt.plot(tdv_prev_2.max(axis=1),ls=':',color='blue')
plt.plot(tdv_prev_2.min(axis=1),ls=':',color='blue')
plt.plot(tdv_prev_3.max(axis=1),ls=':',color='green')
plt.plot(tdv_prev_3.min(axis=1),ls=':',color='green')

#añade una leyenda que diga '1', '2' y '3' en rojo, azul y verde respectivamente
plt.legend(['1','2','3'])

plt.show()
