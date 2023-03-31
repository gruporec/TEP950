from time import time
from matplotlib.markers import MarkerStyle
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import time
import isadoralib as isl
year="2019"
alphaValue=0.4
# Ejecuta cargaRaw.py si no existe rawDiarios.csv o rawMinutales.csv
if not os.path.isfile("rawDiarios"+year+".csv") or not os.path.isfile("rawMinutales"+year+".csv"):
    os.system("python3 cargaRaw.py")

# Carga de datos

tdv,ltp,meteo,valdatapd=isl.cargaDatosTDV(year,"")


# elimina los valores NaN de tdv
tdv = tdv.dropna(axis=1,how='all')
# rellena los valores NaN de tdv con el valor anterior
tdv = tdv.fillna(method='ffill')
# rellena los valores NaN de tdv con el valor siguiente
tdv = tdv.fillna(method='bfill')

# aplica un filtro de media móvil a tdv
tdv = tdv.rolling(window=24*60,center=True).mean()

# calcula el valor medio de tdv para cada dia
tdv_medio = tdv.groupby(tdv.index.date).mean()
# calcula el valor de la desviación estándar de tdv para cada dia
tdv_std = tdv.groupby(tdv.index.date).std()
# cambia el índice a datetime
tdv_medio.index = pd.to_datetime(tdv_medio.index)
tdv_std.index = pd.to_datetime(tdv_std.index)

# remuestrea tdv_medio y tdv_std a minutal
tdv_medio = tdv_medio.resample('T').pad()
tdv_std = tdv_std.resample('T').pad()

# normaliza tdv para cada dia
tdv = (tdv - tdv_medio) / tdv_std

# calcula la pendiente de tdv
tdv_diff = tdv.diff()

# aplica un filtro de media móvil a tdv_diff
tdv_diff = tdv_diff.rolling(window=240,center=True).mean()

# calcula la segunda derivada de tdv
tdv_diff2 = tdv_diff.diff()

#aplica un filtro de media móvil a tdv_diff2
tdv_diff2 = tdv_diff2.rolling(window=240,center=True).mean()

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

#elimina los dias en los que no haya cambio de signo
signos_np=signos_np.dropna()

# duplica el índice de signos pn como una columna más en signos_pn
signos_pn['Hora'] = signos_pn.index
# recorta signos pn al último valor de cada día
signos_pn = signos_pn.resample('D').last()

#elimina los días en los que no haya cambio de signo
signos_pn = signos_pn.dropna()

# recoge los valores del índice de tdv donde la hora es 00:00
tdv_00 = tdv.index.time == time.min
# recoge los valores del índice de tdv donde la hora es la mayor de cada día
tdv_23 = tdv.index.time == time(23,59)

# crea una columna en tdv que vale 0 a las 00:00
tdv.loc[tdv_00,'Hora_norm'] = 0
# iguala Hora_norm a 6 en los índices de signos np
tdv.loc[signos_np['Hora'],'Hora_norm'] = 6
# iguala Hora_norm a 18 en los índices de signos pn
tdv.loc[signos_pn['Hora'],'Hora_norm'] = 18
# iguala Hora_norm a 24 en el último valor de cada día
tdv.loc[tdv_23,'Hora_norm'] = 24
# iguala el valor en la última fila de Hora_norm a 24
tdv.loc[tdv.index[-1],'Hora_norm'] = 24
# interpola Hora_norm en tdv
tdv.loc[:,'Hora_norm'] = tdv.loc[:,'Hora_norm'].interpolate()

# remuestrea valdatapd a minutal con el valor de cada día
valdatapd = valdatapd.resample('T').pad()

# elimina los valores de tdv que no estén en valdatapd
tdvv = tdv.loc[valdatapd.index,valdatapd.columns]
# elimina los valores de diff que no estén en valdatapd
tdv_diffv = tdv_diff.loc[valdatapd.index,valdatapd.columns]
# elimina los valores de diff2 que no estén en valdatapd
tdv_diff2v = tdv_diff2.loc[valdatapd.index,valdatapd.columns]

# recoge los valores del índice de tdvv donde la hora es 00:00
tdv_00 = tdvv.index.time == time.min

# elimina el primer valor de cada día de tdvv
tdvv.loc[tdv_00]=np.nan
#tdvv=tdvv.dropna(axis=0,how='all')

# elimina el primer valor de cada día de tdv_diffv
tdv_diffv.loc[tdv_00]=np.nan
#tdv_diffv=tdv_diffv.dropna(axis=0,how='all')

# elimina el primer valor de cada día de tdv_diff2v
tdv_diff2v.loc[tdv_00]=np.nan

# aplica un filtro de media a tdv_diffv
#tdv_diffv = tdv_diffv.rolling(window=60,center=False).mean()
# crea un dataframe con los valores de tdv cuando valdatapd es 1
tdv_1 = tdvv[valdatapd==1].dropna(how='all')
# crea un dataframe con los valores de tdv cuando valdatapd es 2
tdv_2 = tdvv[valdatapd==2].dropna(how='all')
# crea un dataframe con los valores de tdv cuando valdatapd es 3
tdv_3 = tdvv[valdatapd==3].dropna(how='all')

#crea un dataframe con los valored de diff cuando valdatapd es 1
diff_1 = tdv_diffv[valdatapd==1].dropna(how='all')
#crea un dataframe con los valored de diff cuando valdatapd es 2
diff_2 = tdv_diffv[valdatapd==2].dropna(how='all')
#crea un dataframe con los valored de diff cuando valdatapd es 3
diff_3 = tdv_diffv[valdatapd==3].dropna(how='all')

#crea un dataframe con los valored de diff2 cuando valdatapd es 1
diff2_1 = tdv_diff2v[valdatapd==1].dropna(how='all')
#crea un dataframe con los valored de diff2 cuando valdatapd es 2
diff2_2 = tdv_diff2v[valdatapd==2].dropna(how='all')
#crea un dataframe con los valored de diff2 cuando valdatapd es 3
diff2_3 = tdv_diff2v[valdatapd==3].dropna(how='all')

# añade la columna Hora_norm a tdv_1
tdv_1['Hora_norm'] = tdv['Hora_norm']
# añade la columna Hora_norm a tdv_2
tdv_2['Hora_norm'] = tdv['Hora_norm']
# añade la columna Hora_norm a tdv_3
tdv_3['Hora_norm'] = tdv['Hora_norm']

# añade la columna Hora_norm a diff_1
diff_1['Hora_norm'] = tdv['Hora_norm']
# añade la columna Hora_norm a diff_2
diff_2['Hora_norm'] = tdv['Hora_norm']
# añade la columna Hora_norm a diff_3
diff_3['Hora_norm'] = tdv['Hora_norm']

# añade la columna Hora_norm a diff2_1
diff2_1['Hora_norm'] = tdv['Hora_norm']
# añade la columna Hora_norm a diff2_2
diff2_2['Hora_norm'] = tdv['Hora_norm']
# añade la columna Hora_norm a diff2_3
diff2_3['Hora_norm'] = tdv['Hora_norm']

#cambia el índice de tdv_1 a hora_norm
tdv_1.set_index('Hora_norm',inplace=True)
#cambia el índice de tdv_2 a hora_norm
tdv_2.set_index('Hora_norm',inplace=True)
#cambia el índice de tdv_3 a hora_norm
tdv_3.set_index('Hora_norm',inplace=True)

#cambia el índice de diff_1 a hora_norm
diff_1.set_index('Hora_norm',inplace=True)
#cambia el índice de diff_2 a hora_norm
diff_2.set_index('Hora_norm',inplace=True)
#cambia el índice de diff_3 a hora_norm
diff_3.set_index('Hora_norm',inplace=True)

#cambia el índice de diff2_1 a hora_norm
diff2_1.set_index('Hora_norm',inplace=True)
#cambia el índice de diff2_2 a hora_norm
diff2_2.set_index('Hora_norm',inplace=True)
#cambia el índice de diff2_3 a hora_norm
diff2_3.set_index('Hora_norm',inplace=True)

# # graficas a color
# # grafica tdv_1 poniendo la hora normalizada en el eje x
# axlpt1=tdv_1.plot(ls='none',marker='o',color='cyan',alpha=60*24/len(tdv_1),MarkerSize=1,legend=False)
# axlpt1.set_xlim(0,24)
# axlpt1.set_ylim(-2.5,2.5)
# # grafica tdv_2 poniendo la hora normalizada en el eje x
# axlpt2=tdv_2.plot(ls='none',marker='o',color='magenta',alpha=60*24/len(tdv_2),MarkerSize=1,legend=False)
# axlpt2.set_xlim(0,24)
# axlpt2.set_ylim(-2.5,2.5)
# # grafica tdv_3 poniendo la hora normalizada en el eje x
# axlpt3=tdv_3.plot(ls='none',marker='o',color='yellow',alpha=60*24/len(tdv_3),MarkerSize=1,legend=False)
# axlpt3.set_xlim(0,24)
# axlpt3.set_ylim(-2.5,2.5)
# # grafica diff_1 poniendo la hora normalizada en el eje x
# axdiff1=diff_1.plot(ls='none',marker='o',color='cyan',alpha=60*24/len(diff_1),MarkerSize=1,legend=False)
# axdiff1.set_xlim(0,24)
# axdiff1.set_ylim(-0.025,0.025)
# # grafica diff_2 poniendo la hora normalizada en el eje x
# axdiff2=diff_2.plot(ls='none',marker='o',color='magenta',alpha=60*24/len(diff_2),MarkerSize=1,legend=False)
# axdiff2.set_xlim(0,24)
# axdiff2.set_ylim(-0.025,0.025)
# # grafica diff_3 poniendo la hora normalizada en el eje x
# axdiff3=diff_3.plot(ls='none',marker='o',color='yellow',alpha=60*24/len(diff_3),MarkerSize=1,legend=False)
# axdiff3.set_xlim(0,24)
# axdiff3.set_ylim(-0.025,0.025)

# graficas en negro
# crea una figura con 9 subplots
fig, ((axlpt1,axlpt2,axlpt3),(axdiff1,axdiff2,axdiff3),(axdiff21,axdiff22,axdiff23)) = plt.subplots(3, 3)
# grafica tdv_1 poniendo la hora normalizada en el eje x
print(tdv_1)
axlpt1.plot(tdv_1,ls='none',marker='o',color='black',alpha=alphaValue*60*24/len(tdv_1),MarkerSize=1)
axlpt1.set_xlim(0,24)
axlpt1.set_ylim(-2.5,2.5)
axlpt1.set_title('TDV_1')
# grafica tdv_2 poniendo la hora normalizada en el eje x
axlpt2.plot(tdv_2,ls='none',marker='o',color='black',alpha=alphaValue*60*24/len(tdv_2),MarkerSize=1)
axlpt2.set_xlim(0,24)
axlpt2.set_ylim(-2.5,2.5)
axlpt2.set_title('TDV_2')
# grafica tdv_3 poniendo la hora normalizada en el eje x
axlpt3.plot(tdv_3,ls='none',marker='o',color='black',alpha=alphaValue*60*24/len(tdv_3),MarkerSize=1)
axlpt3.set_xlim(0,24)
axlpt3.set_ylim(-2.5,2.5)
axlpt3.set_title('TDV_3')
# grafica diff_1 poniendo la hora normalizada en el eje x
axdiff1.plot(diff_1,ls='none',marker='o',color='black',alpha=alphaValue*60*24/len(diff_1),MarkerSize=1)
axdiff1.set_xlim(0,24)
axdiff1.set_ylim(-0.025,0.025)
axdiff1.set_title('DIFF_1')
# grafica diff_2 poniendo la hora normalizada en el eje x
axdiff2.plot(diff_2,ls='none',marker='o',color='black',alpha=alphaValue*60*24/len(diff_2),MarkerSize=1)
axdiff2.set_xlim(0,24)
axdiff2.set_ylim(-0.025,0.025)
axdiff2.set_title('DIFF_2')
# grafica diff_3 poniendo la hora normalizada en el eje x
axdiff3.plot(diff_3,ls='none',marker='o',color='black',alpha=alphaValue*60*24/len(diff_3),MarkerSize=1)
axdiff3.set_xlim(0,24)
axdiff3.set_ylim(-0.025,0.025)
axdiff3.set_title('DIFF_3')
# grafica diff2_1 poniendo la hora normalizada en el eje x
axdiff21.plot(diff2_1,ls='none',marker='o',color='black',alpha=alphaValue*60*24/len(diff2_1),MarkerSize=1)
axdiff21.set_xlim(0,24)
#axdiff21.set_ylim(-0.025,0.025)
axdiff21.set_title('DIFF2_1')
# grafica diff2_2 poniendo la hora normalizada en el eje x
axdiff22.plot(diff2_2,ls='none',marker='o',color='black',alpha=alphaValue*60*24/len(diff2_2),MarkerSize=1)
axdiff22.set_xlim(0,24)
#axdiff22.set_ylim(-0.025,0.025)
axdiff22.set_title('DIFF2_2')
# grafica diff2_3 poniendo la hora normalizada en el eje x
axdiff23.plot(diff2_3,ls='none',marker='o',color='black',alpha=alphaValue*60*24/len(diff2_3),MarkerSize=1)
axdiff23.set_xlim(0,24)
#axdiff23.set_ylim(-0.025,0.025)
axdiff23.set_title('DIFF2_3')

# calcula las lineas medias de los 3 tdv
# agrupa las columnas de tdv_1, tdv_2 y tdv_3 en una sola haciendo la media
tdv_1_media=tdv_1.mean(axis=1)
tdv_2_media=tdv_2.mean(axis=1)
tdv_3_media=tdv_3.mean(axis=1)
# ordena las columnas de tdv_1_media, tdv_2_media y tdv_3_media según la hora normalizada
tdv_1_media.sort_index(inplace=True)
tdv_2_media.sort_index(inplace=True)
tdv_3_media.sort_index(inplace=True)
# convierte la hora normalizada a datetime para ajustar frecuencias
tdv_1_media.index=pd.to_datetime(tdv_1_media.index*1000000000)
tdv_2_media.index=pd.to_datetime(tdv_2_media.index*1000000000)
tdv_3_media.index=pd.to_datetime(tdv_3_media.index*1000000000)
# remuestrea las columnas de tdv_1_media, tdv_2_media y tdv_3_media a una frecuencia de 0.001 segundos haciendo la media
tdv_1_media=tdv_1_media.resample('0.1S').mean()
tdv_2_media=tdv_2_media.resample('0.1S').mean()
tdv_3_media=tdv_3_media.resample('0.1S').mean()
# Crea una serie de 0.01 a 24 para restaurar el índice
norm_index=pd.Series(np.arange(0,24.1,0.1))
# Ajusta el índice de tdv_1_media, tdv_2_media y tdv_3_media a la serie de 0.1 a 24
tdv_1_media.index=norm_index
tdv_2_media.index=norm_index
tdv_3_media.index=norm_index
# añade las lineas media a los gráficos
axlpt1.plot(tdv_1_media,ls='-',color='red')
axlpt2.plot(tdv_2_media,ls='-',color='red')
axlpt3.plot(tdv_3_media,ls='-',color='red')

# calcula la varizanza de los 3 tdv entre sensores
# agrupa las columnas de tdv_1, tdv_2 y tdv_3 en una sola haciendo la desviación estándar
tdv_1_std=tdv_1.std(axis=1)*2
tdv_2_std=tdv_2.std(axis=1)*2
tdv_3_std=tdv_3.std(axis=1)*2
# ordena las columnas de tdv_1_var, tdv_2_var y tdv_3_var según la hora normalizada
tdv_1_std.sort_index(inplace=True)
tdv_2_std.sort_index(inplace=True)
tdv_3_std.sort_index(inplace=True)
# convierte la hora normalizada a datetime para ajustar frecuencias
tdv_1_std.index=pd.to_datetime(tdv_1_std.index*1000000000)
tdv_2_std.index=pd.to_datetime(tdv_2_std.index*1000000000)
tdv_3_std.index=pd.to_datetime(tdv_3_std.index*1000000000)
# remuestrea las columnas de tdv_1_var, tdv_2_var y tdv_3_var a una frecuencia de 0.001 segundos haciendo la media
tdv_1_std=tdv_1_std.resample('0.1S').mean()
tdv_2_std=tdv_2_std.resample('0.1S').mean()
tdv_3_std=tdv_3_std.resample('0.1S').mean()
# Crea una serie de 0.01 a 24 para restaurar el índice
norm_index=pd.Series(np.arange(0,24.1,0.1))
# Ajusta el índice de tdv_1_var, tdv_2_var y tdv_3_var a la serie de 0.1 a 24
tdv_1_std.index=norm_index
tdv_2_std.index=norm_index
tdv_3_std.index=norm_index
# suma y resta las desviaciones estándar de los 3 tdv a la media de cada uno
tdv_1_std_sup=tdv_1_std+tdv_1_media
tdv_2_std_sup=tdv_2_std+tdv_2_media
tdv_3_std_sup=tdv_3_std+tdv_3_media
tdv_1_std_inf=tdv_1_media-tdv_1_std
tdv_2_std_inf=tdv_2_media-tdv_2_std
tdv_3_std_inf=tdv_3_media-tdv_3_std
# añade las lineas desviación estándar a los gráficos
axlpt1.plot(tdv_1_std_sup,ls='-',color='green')
axlpt1.plot(tdv_1_std_inf,ls='-',color='green')
axlpt2.plot(tdv_2_std_sup,ls='-',color='green')
axlpt2.plot(tdv_2_std_inf,ls='-',color='green')
axlpt3.plot(tdv_3_std_sup,ls='-',color='green')
axlpt3.plot(tdv_3_std_inf,ls='-',color='green')

# añade una línea vertical en 6 y en 18 que represente el amanecer y el anochecer
axlpt1.axvline(x=6,color='blue')
axlpt1.axvline(x=18,color='blue')
axlpt2.axvline(x=6,color='blue')
axlpt2.axvline(x=18,color='blue')
axlpt3.axvline(x=6,color='blue')
axlpt3.axvline(x=18,color='blue')

# calcula las lineas medias de los 3 diff
# agrupa las columnas de diff_1, diff_2 y diff_3 en una sola haciendo la media
diff_1_media=diff_1.mean(axis=1)
diff_2_media=diff_2.mean(axis=1)
diff_3_media=diff_3.mean(axis=1)
# ordena las columnas de diff_1_media, diff_2_media y diff_3_media según la hora normalizada
diff_1_media.sort_index(inplace=True)
diff_2_media.sort_index(inplace=True)
diff_3_media.sort_index(inplace=True)
# convierte la hora normalizada a datetime para ajustar frecuencias
diff_1_media.index=pd.to_datetime(diff_1_media.index*1000000000)
diff_2_media.index=pd.to_datetime(diff_2_media.index*1000000000)
diff_3_media.index=pd.to_datetime(diff_3_media.index*1000000000)
# remuestrea las columnas de diff_1_media, diff_2_media y diff_3_media a una frecuencia de 0.001 segundos haciendo la media
diff_1_media=diff_1_media.resample('0.1S').mean()
diff_2_media=diff_2_media.resample('0.1S').mean()
diff_3_media=diff_3_media.resample('0.1S').mean()
# Ajusta el índice de diff_1_media, diff_2_media y diff_3_media a la serie de 0.1 a 24
diff_1_media.index=norm_index
diff_2_media.index=norm_index
diff_3_media.index=norm_index
# añade las lineas media a los gráficos
axdiff1.plot(diff_1_media,ls='-',color='red')
axdiff2.plot(diff_2_media,ls='-',color='red')
axdiff3.plot(diff_3_media,ls='-',color='red')

# calcula las desviaciones estándar de los 3 diff
# agrupa las columnas de diff_1, diff_2 y diff_3 en una sola haciendo la desviación estándar
diff_1_std=diff_1.std(axis=1)*2
diff_2_std=diff_2.std(axis=1)*2
diff_3_std=diff_3.std(axis=1)*2
# ordena las columnas de diff_1_std, diff_2_std y diff_3_std según la hora normalizada
diff_1_std.sort_index(inplace=True)
diff_2_std.sort_index(inplace=True)
diff_3_std.sort_index(inplace=True)
# convierte la hora normalizada a datetime para ajustar frecuencias
diff_1_std.index=pd.to_datetime(diff_1_std.index*1000000000)
diff_2_std.index=pd.to_datetime(diff_2_std.index*1000000000)
diff_3_std.index=pd.to_datetime(diff_3_std.index*1000000000)
# remuestrea las columnas de diff_1_std, diff_2_std y diff_3_std a una frecuencia de 0.001 segundos haciendo la desviación estándar
diff_1_std=diff_1_std.resample('0.1S').mean()
diff_2_std=diff_2_std.resample('0.1S').mean()
diff_3_std=diff_3_std.resample('0.1S').mean()
# Ajusta el índice de diff_1_std, diff_2_std y diff_3_std a la serie de 0.1 a 24
diff_1_std.index=norm_index
diff_2_std.index=norm_index
diff_3_std.index=norm_index
# suma y resta las desviaciones estándar de los 3 diff a la media de cada uno
diff_1_std_sup=diff_1_std+diff_1_media
diff_2_std_sup=diff_2_std+diff_2_media
diff_3_std_sup=diff_3_std+diff_3_media
diff_1_std_inf=diff_1_media-diff_1_std
diff_2_std_inf=diff_2_media-diff_2_std
diff_3_std_inf=diff_3_media-diff_3_std
# añade las lineas desviación estándar a los gráficos
axdiff1.plot(diff_1_std_sup,ls='-',color='green')
axdiff1.plot(diff_1_std_inf,ls='-',color='green')
axdiff2.plot(diff_2_std_sup,ls='-',color='green')
axdiff2.plot(diff_2_std_inf,ls='-',color='green')
axdiff3.plot(diff_3_std_sup,ls='-',color='green')
axdiff3.plot(diff_3_std_inf,ls='-',color='green')


# añade una línea vertical en 6 y en 18 que represente el amanecer y el anochecer
axdiff1.axvline(x=6,color='blue')
axdiff1.axvline(x=18,color='blue')
axdiff2.axvline(x=6,color='blue')
axdiff2.axvline(x=18,color='blue')
axdiff3.axvline(x=6,color='blue')
axdiff3.axvline(x=18,color='blue') 

# calcula las lineas medias de los 3 diff2
# agrupa las columnas de diff2_1, diff2_2 y diff2_3 en una sola haciendo la media
diff2_1_media=diff2_1.mean(axis=1)
diff2_2_media=diff2_2.mean(axis=1)
diff2_3_media=diff2_3.mean(axis=1)
# ordena las columnas de diff2_1_media, diff2_2_media y diff2_3_media según la hora normalizada
diff2_1_media.sort_index(inplace=True)
diff2_2_media.sort_index(inplace=True)
diff2_3_media.sort_index(inplace=True)
# convierte la hora normalizada a datetime para ajustar frecuencias
diff2_1_media.index=pd.to_datetime(diff2_1_media.index*1000000000)
diff2_2_media.index=pd.to_datetime(diff2_2_media.index*1000000000)
diff2_3_media.index=pd.to_datetime(diff2_3_media.index*1000000000)
# remuestrea las columnas de diff2_1_media, diff2_2_media y diff2_3_media a una frecuencia de 0.001 segundos haciendo la media
diff2_1_media=diff2_1_media.resample('0.1S').mean()
diff2_2_media=diff2_2_media.resample('0.1S').mean()
diff2_3_media=diff2_3_media.resample('0.1S').mean()
# Ajusta el índice de diff2_1_media, diff2_2_media y diff2_3_media a la serie de 0.1 a 24
diff2_1_media.index=norm_index
diff2_2_media.index=norm_index
diff2_3_media.index=norm_index
# añade las lineas media a los gráficos
axdiff21.plot(diff2_1_media,ls='-',color='red')
axdiff22.plot(diff2_2_media,ls='-',color='red')
axdiff23.plot(diff2_3_media,ls='-',color='red')

# añade una línea vertical en 6 y en 18 que represente el amanecer y el anochecer
axdiff21.axvline(x=6,color='blue')
axdiff21.axvline(x=18,color='blue')
axdiff22.axvline(x=6,color='blue')
axdiff22.axvline(x=18,color='blue')
axdiff23.axvline(x=6,color='blue')
axdiff23.axvline(x=18,color='blue')

# crea una nueva figura con 3 subplots
fig2,(tdvavg,diffavg,diff2avg)=plt.subplots(3,1,sharex=True)
# grafica las lineas medias de los 3 tdv juntas con leyenda
tdvavg.plot(tdv_1_media,ls='-',label='TDV_1')
tdvavg.plot(tdv_2_media,ls='-',label='TDV_2')
tdvavg.plot(tdv_3_media,ls='-',label='TDV_3')
# añade las desviaciones estándar de los 3 tdv como lineas discontinuas azul, naranja y verde respectivamente
tdvavg.plot(tdv_1_std_sup,ls='--',color='blue')
tdvavg.plot(tdv_1_std_inf,ls='--',color='blue')
tdvavg.plot(tdv_2_std_sup,ls='--',color='orange')
tdvavg.plot(tdv_2_std_inf,ls='--',color='orange')
tdvavg.plot(tdv_3_std_sup,ls='--',color='green')
tdvavg.plot(tdv_3_std_inf,ls='--',color='green')
tdvavg.legend()
tdvavg.axvline(x=6,color='black')
tdvavg.axvline(x=18,color='black')
# grafica las lineas medias de los 3 diff juntas con leyenda
diffavg.plot(diff_1_media,ls='-',label='DIFF_1')
diffavg.plot(diff_2_media,ls='-',label='DIFF_2')
diffavg.plot(diff_3_media,ls='-',label='DIFF_3')
# añade las desviaciones estándar de los 3 diff como lineas discontinuas azul, naranja y verde respectivamente
diffavg.plot(diff_1_std_sup,ls='--',color='blue')
diffavg.plot(diff_1_std_inf,ls='--',color='blue')
diffavg.plot(diff_2_std_sup,ls='--',color='orange')
diffavg.plot(diff_2_std_inf,ls='--',color='orange')
diffavg.plot(diff_3_std_sup,ls='--',color='green')
diffavg.plot(diff_3_std_inf,ls='--',color='green')
diffavg.legend()
diffavg.axvline(x=6,color='black')
diffavg.axvline(x=18,color='black')
# grafica las lineas medias de los 3 diff2 juntas con leyenda
diff2avg.plot(diff2_1_media,ls='-',label='DIFF2_1')
diff2avg.plot(diff2_2_media,ls='-',label='DIFF2_2')
diff2avg.plot(diff2_3_media,ls='-',label='DIFF2_3')
diff2avg.legend()
diff2avg.axvline(x=6,color='black')
diff2avg.axvline(x=18,color='black')

# show
plt.show()

# guarda las gráficas en un archivo
fig.savefig('figuras/figuras dias/TDVdiariosf24h'+year+'.png')
fig2.savefig('figuras/figuras dias/TDVmediasf24h'+year+'.png')
