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

#matplotlib.use("Agg")

#pd.set_option('display.max_rows', None)

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

# duplica el índice de signos pn como una columna más en signos_pn
signos_pnP['Hora'] = signos_pnP.index
# recorta signos pn al último valor de cada día
signos_pnP = signos_pnP.resample('D').last()

# recoge los valores del índice de ltp donde la hora es 00:00
ltp_00P = ltpP.index.time == time.min
# recoge los valores del índice de ltp donde la hora es la mayor de cada día
ltp_23P = ltpP.index.time == time(23,59)

#elimina los dias en los que no haya cambio de signo
signos_npP=signos_npP.dropna()

#elimina los días en los que no haya cambio de signo
signos_pnP = signos_pnP.dropna()

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
# recorta ltp a los tramos de 6 a 18 de hora_norm
ltpP = ltpP.loc[ltpP['Hora_norm']>=6,:]
ltpP = ltpP.loc[ltpP['Hora_norm']<=18,:]

# añade la hora normalizada al índice de ltp
ltpP.index = [ltpP.index.strftime('%Y-%m-%d'),ltpP['Hora_norm']]

valdatapd.index = valdatapd.index.strftime('%Y-%m-%d')

#obtiene el índice interseccion de valdatapd y el primer nivel del índice de ltp
ltpPdates = ltpP.index.get_level_values(0)

valdatapd_ltp = valdatapd.index.intersection(ltpPdates)

# elimina los valores de ltp que no estén en valdatapd
ltpv = ltpP.loc[valdatapd_ltp,valdatapd.columns]

# unstackea ltpv
ltpv = ltpv.unstack(level=0)

# sustituye el indice de ltpv por el valor de la columna Hora de ltpv_index
ltpv_index_float=pd.Int64Index(np.floor(ltpv.index*1000000000))

# convierte el indice a datetime para ajustar frecuencias
ltpv.index = pd.to_datetime(ltpv_index_float)
ltpv=ltpv.resample('0.1S').mean()

# conserva los valores de 1970-01-01 00:00:06.000 a 1970-01-01 00:00:17.900
ltpv = ltpv.loc[ltpv.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
ltpv = ltpv.loc[ltpv.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]

# Crea una serie de 0.01 a 24 para restaurar el índice
norm_index=pd.Series(np.arange(6,18,0.1))
# Ajusta el índice de ltpv a la serie de 0.01 a 24
ltpv.index=norm_index

# crea los valores X e y para el modelo
Xv=ltpv.values.transpose()
Yv=valdatapd.unstack().values

#crea un vector para las predicciones
Ypred=np.zeros(Yv.shape)
#crea una matriz de 3 filas para las posiciones de los máximos de convolución
max_pos=np.zeros((3,Yv.shape[0]))
#crea una matriz de 3 filas para los valores de los máximos de convolución
max_val=np.zeros((3,Yv.shape[0]))

# invierte el orden de los valores de ltp_1
ltp_1_flip = ltp_1[::-1]
# invierte el orden de los valores de ltp_2
ltp_2_flip = ltp_2[::-1]
# invierte el orden de los valores de ltp_3
ltp_3_flip = ltp_3[::-1]
#recorre las filas de Xv
for i in range(Xv.shape[0]):
    # calcula la convolución de Xv[:,i] con ltp_1
    conv_1=np.convolve(Xv[i,:],ltp_1_flip)
    # calcula la convolución de Xv[:,i] con ltp_2
    conv_2=np.convolve(Xv[i,:],ltp_2_flip)
    # calcula la convolución de Xv[:,i] con ltp_3
    conv_3=np.convolve(Xv[i,:],ltp_3_flip)
    
    #calcula el punto máximo de las convoluciones
    max_1=np.max(conv_1)
    max_2=np.max(conv_2)
    max_3=np.max(conv_3)

    #calcula la posición del punto máximo de las convoluciones
    max_1_pos=np.argmax(conv_1)
    max_2_pos=np.argmax(conv_2)
    max_3_pos=np.argmax(conv_3)

    # calcula el desplazamiento del punto máximo de las convoluciones
    max_1_pos_shift=max_1_pos-len(ltp_1_flip)
    max_2_pos_shift=max_2_pos-len(ltp_2_flip)
    max_3_pos_shift=max_3_pos-len(ltp_3_flip)

    #guarda los máximos de las convoluciones en la matriz max_pos
    max_pos[0,i]=max_1_pos_shift
    max_pos[1,i]=max_2_pos_shift
    max_pos[2,i]=max_3_pos_shift
    #guarda los valores de las convoluciones en la matriz max_val
    max_val[0,i]=max_1
    max_val[1,i]=max_2
    max_val[2,i]=max_3

    # clasifica la curva en función del mínimo desplazamiento del punto máximo en valor absoluto
    if abs(max_1_pos_shift)<abs(max_2_pos_shift) and abs(max_1_pos_shift)<abs(max_3_pos_shift):
        Ypred[i]=1
    elif abs(max_3_pos_shift)<abs(max_1_pos_shift) and abs(max_3_pos_shift)<abs(max_2_pos_shift):
        Ypred[i]=3
    else:
        Ypred[i]=2
# calcula la matriz de confusion
confusion_matrix = skmetrics.confusion_matrix(Yv, Ypred)

print(confusion_matrix)

# calcula el porcentaje de acierto
accuracy = skmetrics.balanced_accuracy_score(Yv, Ypred)
print('Porcentaje de acierto: '+str(accuracy*100)+'%')
sys.exit()

res=pd.DataFrame()
res['valdata']=valdatapd.unstack()
res['estado']=pd.DataFrame(Ypred,index=res.index)
res['max_pos1']=pd.DataFrame(max_pos[0,:],index=res.index)
res['max_pos2']=pd.DataFrame(max_pos[1,:],index=res.index)
res['max_pos3']=pd.DataFrame(max_pos[2,:],index=res.index)
res['max_val1']=pd.DataFrame(max_val[0,:],index=res.index)
res['max_val2']=pd.DataFrame(max_val[1,:],index=res.index)
res['max_val3']=pd.DataFrame(max_val[2,:],index=res.index)
res = res.transpose()

# Function to map the colors as a list from the input list of x variables
def pltcolor(lst):
    cols=[]
    for l in lst:
        if l==1:
            cols.append('red')
        elif l==2:
            cols.append('green')
        else:
            cols.append('blue')
    return cols
# Create the colors list using the function above
cols=pltcolor(res.loc['valdata'].values)

#crea una figura con tres subplots
fig, axs = plt.subplots(3, 1, figsize=(10,10))
# en el primer subplot representa una gráfica de puntos con la posición del máximo 1 en el eje x, el valor del máximo 1 en el eje y y el color del punto según el estado
axs[0].scatter(res.loc['max_pos1'].values,res.loc['max_val1'].values,s=1,color=cols)
# añade un título al subplot
axs[0].set_title('Convolución con la primera señal tipo')
# añade titulo a los ejes
axs[0].set_xlabel('Posición del máximo')
axs[0].set_ylabel('Valor del máximo')
# en el segundo subplot representa una gráfica de puntos con la posición del máximo 2 en el eje x, el valor del máximo 2 en el eje y y el color del punto según el estado
axs[1].scatter(res.loc['max_pos2'].values,res.loc['max_val2'].values,s=1,color=cols)
# añade un título al subplot
axs[1].set_title('Convolución con la segunda señal tipo')
# añade titulo a los ejes
axs[1].set_xlabel('Posición del máximo')
axs[1].set_ylabel('Valor del máximo')
# en el tercer subplot representa una gráfica de puntos con la posición del máximo 3 en el eje x, el valor del máximo 3 en el eje y y el color del punto según el estado
axs[2].scatter(res.loc['max_pos3'].values,res.loc['max_val3'].values,s=1,color=cols)
# añade un título al subplot
axs[2].set_title('Convolución con la tercera señal tipo')
# añade titulo a los ejes
axs[2].set_xlabel('Posición del máximo')
axs[2].set_ylabel('Valor del máximo')
plt.show()