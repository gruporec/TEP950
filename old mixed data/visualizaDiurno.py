import sys
from time import time
from matplotlib.markers import MarkerStyle
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import time

# Ejecuta cargaRaw.py si no existe rawDiarios.csv o rawMinutales.csv
if not os.path.isfile("rawDiarios.csv") or not os.path.isfile("rawMinutales.csv"):
    os.system("python3 cargaRaw.py")

# Carga de datos
dfT = pd.read_csv("rawMinutales.csv",na_values='.')
dfT.loc[:,"Fecha"]=pd.to_datetime(dfT.loc[:,"Fecha"])# Fecha como datetime
dfT=dfT.drop_duplicates(subset="Fecha")
dfT.dropna(subset = ["Fecha"], inplace=True)
dfT=dfT.set_index("Fecha")

dfd = pd.read_csv("rawDiarios.csv",na_values='.')
dfd.loc[:,"Fecha"]=pd.to_datetime(dfd.loc[:,"Fecha"])# Fecha como datetime
dfd=dfd.drop_duplicates(subset="Fecha")
dfd.dropna(subset = ["Fecha"], inplace=True)
dfd=dfd.set_index("Fecha")

# dropea las columnas de dfd que empiezan por estado
dfd=dfd.drop(dfd.columns[dfd.columns.str.startswith('Estado')], axis=1)

# separa dfT en tdv y ltp en función del principio del nombre de cada columna y guarda el resto en meteo
tdv = dfT.loc[:,dfT.columns.str.startswith('TDV')]
ltp = dfT.loc[:,dfT.columns.str.startswith('LTP')]
meteo = dfT.drop(dfT.columns[dfT.columns.str.startswith('TDV')], axis=1)
meteo = meteo.drop(meteo.columns[meteo.columns.str.startswith('LTP')], axis=1)

# Carga datos de  validacion
valdatapd=pd.read_csv("validacion.csv")
valdatapd.dropna(inplace=True)
valdatapd['Fecha'] = pd.to_datetime(valdatapd['Fecha'])
valdatapd.set_index('Fecha',inplace=True)

# elimina los valores NaN de ltp
ltp = ltp.dropna(axis=1,how='all')
# rellena los valores NaN de ltp con el valor anterior
ltp = ltp.fillna(method='ffill')
# rellena los valores NaN de ltp con el valor siguiente
ltp = ltp.fillna(method='bfill')

# recoge los valores del índice de ltp donde la hora es 00:00
ltp_00 = ltp.index.time == time.min
# recoge los valores del índice de ltp donde la hora es la mayor de cada día
ltp_23 = ltp.index.time == time(23,59)

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

# conserva los valores con hora normalizada entre 6 y 18
ltp = ltp[ltp.Hora_norm >= 6]
ltp = ltp[ltp.Hora_norm <= 18]

# aplica un filtro de media móvil a ltp
ltp = ltp.rolling(window=240,center=True).mean()
ltp=ltp.dropna()

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
hora_n=ltp['Hora_norm']
ltp = (ltp - ltp_medio) / ltp_std
ltp['Hora_norm']=hora_n
ltp=ltp.dropna()

# calcula la pendiente de ltp
ltp_diff = ltp.diff()
ltp_diff['Hora_norm']=ltp['Hora_norm']

# aplica un filtro de media móvil a ltp_diff
ltp_diff = ltp_diff.rolling(window=240,center=True).mean()
ltp_diff=ltp_diff.dropna()

# calcula la segunda derivada de ltp
ltp_diff2 = ltp_diff.diff()
ltp_diff2['Hora_norm']=ltp_diff['Hora_norm']

#aplica un filtro de media móvil a ltp_diff2
ltp_diff2 = ltp_diff2.rolling(window=240,center=True).mean()
ltp_diff2=ltp_diff2.dropna()

# remuestrea valdatapd a minutal con el valor de cada día
valdatapd = valdatapd.resample('T').pad()

# elimina los valores de ltp que no estén en valdatapd
ltpv = ltp[ltp.index.isin(valdatapd.index)]
ltpv = ltpv.loc[:,ltpv.columns.isin(valdatapd.columns)]
# elimina los valores de diff que no estén en valdatapd
ltp_diffv = ltp_diff[ltp_diff.index.isin(valdatapd.index)]
ltp_diffv = ltp_diff.loc[:,ltp_diff.columns.isin(valdatapd.columns)]
# elimina los valores de diff2 que no estén en valdatapd
ltp_diff2v = ltp_diff2[ltp_diff2.index.isin(valdatapd.index)]
ltp_diff2v = ltp_diff2.loc[:,ltp_diff2.columns.isin(valdatapd.columns)]

# crea un dataframe con los valores de ltp cuando valdatapd es 1
ltp_1 = ltpv[valdatapd==1].dropna(how='all')
# crea un dataframe con los valores de ltp cuando valdatapd es 2
ltp_2 = ltpv[valdatapd==2].dropna(how='all')
# crea un dataframe con los valores de ltp cuando valdatapd es 3
ltp_3 = ltpv[valdatapd==3].dropna(how='all')

#crea un dataframe con los valored de diff cuando valdatapd es 1
diff_1 = ltp_diffv[valdatapd==1].dropna(how='all')
#crea un dataframe con los valored de diff cuando valdatapd es 2
diff_2 = ltp_diffv[valdatapd==2].dropna(how='all')
#crea un dataframe con los valored de diff cuando valdatapd es 3
diff_3 = ltp_diffv[valdatapd==3].dropna(how='all')

#crea un dataframe con los valored de diff2 cuando valdatapd es 1
diff2_1 = ltp_diff2v[valdatapd==1].dropna(how='all')
#crea un dataframe con los valored de diff2 cuando valdatapd es 2
diff2_2 = ltp_diff2v[valdatapd==2].dropna(how='all')
#crea un dataframe con los valored de diff2 cuando valdatapd es 3
diff2_3 = ltp_diff2v[valdatapd==3].dropna(how='all')

# añade la columna Hora_norm a ltp_1
ltp_1['Hora_norm'] = ltp['Hora_norm']
# añade la columna Hora_norm a ltp_2
ltp_2['Hora_norm'] = ltp['Hora_norm']
# añade la columna Hora_norm a ltp_3
ltp_3['Hora_norm'] = ltp['Hora_norm']

# añade la columna Hora_norm a diff_1
diff_1['Hora_norm'] = ltp['Hora_norm']
# añade la columna Hora_norm a diff_2
diff_2['Hora_norm'] = ltp['Hora_norm']
# añade la columna Hora_norm a diff_3
diff_3['Hora_norm'] = ltp['Hora_norm']

# añade la columna Hora_norm a diff2_1
diff2_1['Hora_norm'] = ltp['Hora_norm']
# añade la columna Hora_norm a diff2_2
diff2_2['Hora_norm'] = ltp['Hora_norm']
# añade la columna Hora_norm a diff2_3
diff2_3['Hora_norm'] = ltp['Hora_norm']

#cambia el índice de ltp_1 a hora_norm
ltp_1.set_index('Hora_norm',inplace=True)
#cambia el índice de ltp_2 a hora_norm
ltp_2.set_index('Hora_norm',inplace=True)
#cambia el índice de ltp_3 a hora_norm
ltp_3.set_index('Hora_norm',inplace=True)

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


# graficas en negro
# crea una figura con 9 subplots
fig, ((axlpt1,axlpt2,axlpt3),(axdiff1,axdiff2,axdiff3),(axdiff21,axdiff22,axdiff23)) = plt.subplots(3, 3)
# grafica ltp_1 poniendo la hora normalizada en el eje x
axlpt1.plot(ltp_1,ls='none',marker='o',color='black',alpha=60*24/len(ltp_1),MarkerSize=1)
axlpt1.set_xlim(6,18)
axlpt1.set_ylim(-2.5,2.5)
axlpt1.set_title('LTP_1')
# grafica ltp_2 poniendo la hora normalizada en el eje x
axlpt2.plot(ltp_2,ls='none',marker='o',color='black',alpha=60*24/len(ltp_2),MarkerSize=1)
axlpt2.set_xlim(6,18)
axlpt2.set_ylim(-2.5,2.5)
axlpt2.set_title('LTP_2')
# grafica ltp_3 poniendo la hora normalizada en el eje x
axlpt3.plot(ltp_3,ls='none',marker='o',color='black',alpha=60*24/len(ltp_3),MarkerSize=1)
axlpt3.set_xlim(6,18)
axlpt3.set_ylim(-2.5,2.5)
axlpt3.set_title('LTP_3')
# grafica diff_1 poniendo la hora normalizada en el eje x
axdiff1.plot(diff_1,ls='none',marker='o',color='black',alpha=60*24/len(diff_1),MarkerSize=1)
axdiff1.set_xlim(6,18)
axdiff1.set_ylim(-0.025,0.025)
axdiff1.set_title('DIFF_1')
# grafica diff_2 poniendo la hora normalizada en el eje x
axdiff2.plot(diff_2,ls='none',marker='o',color='black',alpha=60*24/len(diff_2),MarkerSize=1)
axdiff2.set_xlim(6,18)
axdiff2.set_ylim(-0.025,0.025)
axdiff2.set_title('DIFF_2')
# grafica diff_3 poniendo la hora normalizada en el eje x
axdiff3.plot(diff_3,ls='none',marker='o',color='black',alpha=60*24/len(diff_3),MarkerSize=1)
axdiff3.set_xlim(6,18)
axdiff3.set_ylim(-0.025,0.025)
axdiff3.set_title('DIFF_3')
# grafica diff2_1 poniendo la hora normalizada en el eje x
axdiff21.plot(diff2_1,ls='none',marker='o',color='black',alpha=60*24/len(diff2_1),MarkerSize=1)
axdiff21.set_xlim(6,18)
#axdiff21.set_ylim(-0.025,0.025)
axdiff21.set_title('DIFF2_1')
# grafica diff2_2 poniendo la hora normalizada en el eje x
axdiff22.plot(diff2_2,ls='none',marker='o',color='black',alpha=60*24/len(diff2_2),MarkerSize=1)
axdiff22.set_xlim(6,18)
#axdiff22.set_ylim(-0.025,0.025)
axdiff22.set_title('DIFF2_2')
# grafica diff2_3 poniendo la hora normalizada en el eje x
axdiff23.plot(diff2_3,ls='none',marker='o',color='black',alpha=60*24/len(diff2_3),MarkerSize=1)
axdiff23.set_xlim(6,18)
#axdiff23.set_ylim(-0.025,0.025)
axdiff23.set_title('DIFF2_3')

# # calcula las lineas medias de los 3 ltp
# # agrupa las columnas de ltp_1, ltp_2 y ltp_3 en una sola haciendo la media
# ltp_1_media=ltp_1.mean(axis=1)
# ltp_2_media=ltp_2.mean(axis=1)
# ltp_3_media=ltp_3.mean(axis=1)
# # ordena las columnas de ltp_1_media, ltp_2_media y ltp_3_media según la hora normalizada
# ltp_1_media.sort_index(inplace=True)
# ltp_2_media.sort_index(inplace=True)
# ltp_3_media.sort_index(inplace=True)

# # agrupa los valores cercanos haciendo la media
# ltp_1_media.groupby(np.arange(100)).mean()
# ltp_2_media.groupby(np.arange(100)).mean()
# ltp_3_media.groupby(np.arange(100)).mean()

# # añade las lineas media a los gráficos
# axlpt1.plot(ltp_1_media,ls='-',color='red')
# axlpt2.plot(ltp_2_media,ls='-',color='red')
# axlpt3.plot(ltp_3_media,ls='-',color='red')

# # calcula la varizanza de los 3 ltp entre sensores
# # agrupa las columnas de ltp_1, ltp_2 y ltp_3 en una sola haciendo la desviación estándar
# ltp_1_std=ltp_1.std(axis=1)*2
# ltp_2_std=ltp_2.std(axis=1)*2
# ltp_3_std=ltp_3.std(axis=1)*2
# # ordena las columnas de ltp_1_var, ltp_2_var y ltp_3_var según la hora normalizada
# ltp_1_std.sort_index(inplace=True)
# ltp_2_std.sort_index(inplace=True)
# ltp_3_std.sort_index(inplace=True)
# # suma y resta las desviaciones estándar de los 3 ltp a la media de cada uno
# ltp_1_std_sup=ltp_1_std+ltp_1_media
# ltp_2_std_sup=ltp_2_std+ltp_2_media
# ltp_3_std_sup=ltp_3_std+ltp_3_media
# ltp_1_std_inf=ltp_1_media-ltp_1_std
# ltp_2_std_inf=ltp_2_media-ltp_2_std
# ltp_3_std_inf=ltp_3_media-ltp_3_std
# # añade las lineas desviación estándar a los gráficos
# axlpt1.plot(ltp_1_std_sup,ls='-',color='green')
# axlpt1.plot(ltp_1_std_inf,ls='-',color='green')
# axlpt2.plot(ltp_2_std_sup,ls='-',color='green')
# axlpt2.plot(ltp_2_std_inf,ls='-',color='green')
# axlpt3.plot(ltp_3_std_sup,ls='-',color='green')
# axlpt3.plot(ltp_3_std_inf,ls='-',color='green')

# añade una línea vertical en 6 y en 18 que represente el amanecer y el anochecer
axlpt1.axvline(x=6,color='blue')
axlpt1.axvline(x=18,color='blue')
axlpt2.axvline(x=6,color='blue')
axlpt2.axvline(x=18,color='blue')
axlpt3.axvline(x=6,color='blue')
axlpt3.axvline(x=18,color='blue')

# # calcula las lineas medias de los 3 diff
# # agrupa las columnas de diff_1, diff_2 y diff_3 en una sola haciendo la media
# diff_1_media=diff_1.mean(axis=1)
# diff_2_media=diff_2.mean(axis=1)
# diff_3_media=diff_3.mean(axis=1)
# # ordena las columnas de diff_1_media, diff_2_media y diff_3_media según la hora normalizada
# diff_1_media.sort_index(inplace=True)
# diff_2_media.sort_index(inplace=True)
# diff_3_media.sort_index(inplace=True)
# # agrupa los valores cercanos haciendo la media
# diff_1_media.groupby(np.arange(100)).mean()
# diff_2_media.groupby(np.arange(100)).mean()
# diff_3_media.groupby(np.arange(100)).mean()

# # añade las lineas media a los gráficos
# axdiff1.plot(diff_1_media,ls='-',color='red')
# axdiff2.plot(diff_2_media,ls='-',color='red')
# axdiff3.plot(diff_3_media,ls='-',color='red')

# # calcula las desviaciones estándar de los 3 diff
# # agrupa las columnas de diff_1, diff_2 y diff_3 en una sola haciendo la desviación estándar
# diff_1_std=diff_1.std(axis=1)*2
# diff_2_std=diff_2.std(axis=1)*2
# diff_3_std=diff_3.std(axis=1)*2
# # ordena las columnas de diff_1_std, diff_2_std y diff_3_std según la hora normalizada
# diff_1_std.sort_index(inplace=True)
# diff_2_std.sort_index(inplace=True)
# diff_3_std.sort_index(inplace=True)


# # suma y resta las desviaciones estándar de los 3 diff a la media de cada uno
# diff_1_std_sup=diff_1_std+diff_1_media
# diff_2_std_sup=diff_2_std+diff_2_media
# diff_3_std_sup=diff_3_std+diff_3_media
# diff_1_std_inf=diff_1_media-diff_1_std
# diff_2_std_inf=diff_2_media-diff_2_std
# diff_3_std_inf=diff_3_media-diff_3_std
# # añade las lineas desviación estándar a los gráficos
# axdiff1.plot(diff_1_std_sup,ls='-',color='green')
# axdiff1.plot(diff_1_std_inf,ls='-',color='green')
# axdiff2.plot(diff_2_std_sup,ls='-',color='green')
# axdiff2.plot(diff_2_std_inf,ls='-',color='green')
# axdiff3.plot(diff_3_std_sup,ls='-',color='green')
# axdiff3.plot(diff_3_std_inf,ls='-',color='green')


# añade una línea vertical en 6 y en 18 que represente el amanecer y el anochecer
axdiff1.axvline(x=6,color='blue')
axdiff1.axvline(x=18,color='blue')
axdiff2.axvline(x=6,color='blue')
axdiff2.axvline(x=18,color='blue')
axdiff3.axvline(x=6,color='blue')
axdiff3.axvline(x=18,color='blue') 

# # calcula las lineas medias de los 3 diff2
# # agrupa las columnas de diff2_1, diff2_2 y diff2_3 en una sola haciendo la media
# diff2_1_media=diff2_1.mean(axis=1)
# diff2_2_media=diff2_2.mean(axis=1)
# diff2_3_media=diff2_3.mean(axis=1)
# # ordena las columnas de diff2_1_media, diff2_2_media y diff2_3_media según la hora normalizada
# diff2_1_media.sort_index(inplace=True)
# diff2_2_media.sort_index(inplace=True)
# diff2_3_media.sort_index(inplace=True)

# # añade las lineas media a los gráficos
# axdiff21.plot(diff2_1_media,ls='-',color='red')
# axdiff22.plot(diff2_2_media,ls='-',color='red')
# axdiff23.plot(diff2_3_media,ls='-',color='red')

# añade una línea vertical en 6 y en 18 que represente el amanecer y el anochecer
axdiff21.axvline(x=6,color='blue')
axdiff21.axvline(x=18,color='blue')
axdiff22.axvline(x=6,color='blue')
axdiff22.axvline(x=18,color='blue')
axdiff23.axvline(x=6,color='blue')
axdiff23.axvline(x=18,color='blue')

# crea una nueva figura con 3 subplots
# fig2,(ltpavg,diffavg,diff2avg)=plt.subplots(3,1,sharex=True)
# grafica las lineas medias de los 3 ltp juntas con leyenda
# ltpavg.plot(ltp_1_media,ls='-',label='LTP_1')
# ltpavg.plot(ltp_2_media,ls='-',label='LTP_2')
# ltpavg.plot(ltp_3_media,ls='-',label='LTP_3')
# # añade las desviaciones estándar de los 3 ltp como lineas discontinuas azul, naranja y verde respectivamente
# ltpavg.plot(ltp_1_std_sup,ls='--',color='blue')
# ltpavg.plot(ltp_1_std_inf,ls='--',color='blue')
# ltpavg.plot(ltp_2_std_sup,ls='--',color='orange')
# ltpavg.plot(ltp_2_std_inf,ls='--',color='orange')
# ltpavg.plot(ltp_3_std_sup,ls='--',color='green')
# ltpavg.plot(ltp_3_std_inf,ls='--',color='green')
# ltpavg.legend()
# ltpavg.axvline(x=6,color='black')
# ltpavg.axvline(x=18,color='black')
# grafica las lineas medias de los 3 diff juntas con leyenda
# diffavg.plot(diff_1_media,ls='-',label='DIFF_1')
# diffavg.plot(diff_2_media,ls='-',label='DIFF_2')
# diffavg.plot(diff_3_media,ls='-',label='DIFF_3')
# # añade las desviaciones estándar de los 3 diff como lineas discontinuas azul, naranja y verde respectivamente
# diffavg.plot(diff_1_std_sup,ls='--',color='blue')
# diffavg.plot(diff_1_std_inf,ls='--',color='blue')
# diffavg.plot(diff_2_std_sup,ls='--',color='orange')
# diffavg.plot(diff_2_std_inf,ls='--',color='orange')
# diffavg.plot(diff_3_std_sup,ls='--',color='green')
# diffavg.plot(diff_3_std_inf,ls='--',color='green')
# diffavg.legend()
# diffavg.axvline(x=6,color='black')
# diffavg.axvline(x=18,color='black')
# grafica las lineas medias de los 3 diff2 juntas con leyenda
# diff2avg.plot(diff2_1_media,ls='-',label='DIFF2_1')
# diff2avg.plot(diff2_2_media,ls='-',label='DIFF2_2')
# diff2avg.plot(diff2_3_media,ls='-',label='DIFF2_3')
# diff2avg.legend()
# diff2avg.axvline(x=6,color='black')
# diff2avg.axvline(x=18,color='black')

# show
plt.show()

# # guarda las gráficas en un archivo
# axlpt1.get_figure().savefig('figuras dias/ltp_1.png')
# axlpt2.get_figure().savefig('figuras dias/ltp_2.png')
# axlpt3.get_figure().savefig('figuras dias/ltp_3.png')
# axdiff1.get_figure().savefig('figuras dias/diff_1.png')
# axdiff2.get_figure().savefig('figuras dias/diff_2.png')
# axdiff3.get_figure().savefig('figuras dias/diff_3.png')
