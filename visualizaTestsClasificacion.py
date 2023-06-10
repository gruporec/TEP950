import sys
from time import time
import matplotlib
from matplotlib import patches as mpatches
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

matplotlib.use("Agg")

year="2016"
desfase_estres=1
n=7
alfa=0.25
save_folder = 'ignore/resultadosTDV/batch/Caracteristicas TDV/'+year+'/'

# si no existe la carpeta, la crea
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Ejecuta cargaRaw.py si no existe rawDiarios.csv o rawMinutales.csv
if not os.path.isfile("rawDiarios"+year+".csv") or not os.path.isfile("rawMinutales"+year+".csv"):
    os.system("python3 cargaRaw.py")

# Carga de datos
tdv,ltp,meteo,valdatapd=isl.cargaDatosTDV(year,"")

#elimina los valores nan
tdv=tdv.dropna()
ltp=ltp.dropna()
meteo=meteo.dropna()
valdatapd=valdatapd.dropna()

# obtiene los valores de tdv entre las 5 y las 8 de cada dia
tdv_5_8 = tdv.between_time(time(5,0),time(8,0))
# calcula el máximo de tdv entre las 5 y las 8 de cada dia
tdv_5_8_max = tdv_5_8.groupby(tdv_5_8.index.date).max()
# calcula el incremento del máximo entre cada día
tdv_5_8_max_diff = tdv_5_8_max.diff(periods=1).dropna()
# obtiene el signo del incremento del máximo entre cada día
tdv_5_8_max_diff_sign = tdv_5_8_max_diff.apply(np.sign)
# sustituye los valores negativos por 0
tdv_5_8_max_diff_sign[tdv_5_8_max_diff_sign<0]=0
# crea un dataframe que valga 1 cuando tdv_5_8_max_diff_sign vale 0 y 0 cuando vale 1
tdv_5_8_max_diff_sign_inv=tdv_5_8_max_diff_sign.apply(lambda x: 1-x)

# crea dos dataframe con el tamaño de tdv_5_8_max_diff_sign y valores 0
pk0=pd.DataFrame(np.zeros(tdv_5_8_max_diff_sign.shape),index=tdv_5_8_max_diff_sign.index,columns=tdv_5_8_max_diff_sign.columns)
pk1=pd.DataFrame(np.zeros(tdv_5_8_max_diff_sign.shape),index=tdv_5_8_max_diff_sign.index,columns=tdv_5_8_max_diff_sign.columns)
# por cada día en tdv_5_8_max_diff_sign	
for i in tdv_5_8_max_diff_sign.index:
    # si es la primera fila
    if i==tdv_5_8_max_diff_sign.index[0]:
        # añade a pk0 el valor de tdv_5_8_max_diff_sign_inv
        pk0.loc[i]=tdv_5_8_max_diff_sign_inv.loc[i]
        # añade a pk1 el valor de tdv_5_8_max_diff_sign
        pk1.loc[i]=tdv_5_8_max_diff_sign.loc[i]
    # si no es la primera fila
    else:
        #calcula el indice anterior restándole un día
        i_ant=i-pd.Timedelta(days=1)
        #añade a pk0 el valor de la fila anterior de pk0 mas el valor de la fila de tdv_5_8_max_diff_sign_inv, multiplicado por el valor de la fila de tdv_5_8_max_diff_sign_inv
        pk0.loc[i]=(pk0.loc[i_ant]+tdv_5_8_max_diff_sign_inv.loc[i])*tdv_5_8_max_diff_sign_inv.loc[i]
        #añade a pk1 el valor de la fila anterior de pk1 mas el valor de la fila de tdv_5_8_max_diff_sign, multiplicado por el valor de la fila de tdv_5_8_max_diff_sign
        pk1.loc[i]=(pk1.loc[i_ant]+tdv_5_8_max_diff_sign.loc[i])*tdv_5_8_max_diff_sign.loc[i]
#suma pk0 y pk1
pk=pk1-pk0



#crea una copia de tdv_5_8_max_diff_sign
bk=tdv_5_8_max_diff_sign.copy()
#crea otra copia de tdv_5_8_max_diff_sign para usarla como auxiliar
bk_aux=tdv_5_8_max_diff_sign.copy()
#elimina valores nan
bk=bk.dropna()
bk_aux=bk_aux.dropna()

#repite n-1 veces
for i in range(1,n,1):
    #desplaza pk_aux un día hacia adelante
    bk_aux.index = bk_aux.index + pd.Timedelta(days=1)
    #duplica el valor de pk
    bk=bk*2
    #añade el valor de pk_aux a pk
    bk=bk+bk_aux

#elimina los valores nan
bk=bk.dropna()

# crea un dataframe con diff tdv_5_8_max_diff_sign que representa los cambios de tendencia
ctend=pd.DataFrame(tdv_5_8_max_diff_sign.diff(periods=1).dropna())

# iguala a 1 los valores no nulos
ctend[ctend!=0]=1

# obtiene los valores del máximo en la franja horaria cuando hay cambio de tendencia
max_ctend=tdv_5_8_max[ctend!=0]
# rellena los valores nulos con el valor anterior
max_ctend=max_ctend.fillna(method='ffill')
#cuando no hay valor anterior, rellena con 0
max_ctend=max_ctend.fillna(0)
#añade un día a la fecha
max_ctend.index = max_ctend.index + pd.Timedelta(days=1)
#calcula la diferencia entre el máximo actual y el máximo en el último cambio de tendencia
max_ctend_diff=tdv_5_8_max-max_ctend
#elimina nan
max_ctend_diff=max_ctend_diff.dropna()

#aplica a valdatapd un desfase de desfase_estres dias
valdatapd.index = valdatapd.index + pd.Timedelta(days=desfase_estres)

# convierte los índices de tdv_5_8_max, pk, bk, max_ctend_diff y valdatapd a datetime
tdv_5_8_max.index = pd.to_datetime(tdv_5_8_max.index)
pk.index = pd.to_datetime(pk.index)
bk.index = pd.to_datetime(bk.index)
max_ctend_diff.index = pd.to_datetime(max_ctend_diff.index)
valdatapd.index = pd.to_datetime(valdatapd.index)

# recorta los dataframes tdv_5_8_max, pk, bk, max_ctend_diff y valdatapd para que tengan el mismo tamaño e índices
common_index = tdv_5_8_max.index.intersection(pk.index).intersection(bk.index).intersection(max_ctend_diff.index).intersection(valdatapd.index)
tdv_5_8_max = tdv_5_8_max.loc[common_index]
pk = pk.loc[common_index]
bk = bk.loc[common_index]
max_ctend_diff = max_ctend_diff.loc[common_index]
valdatapd = valdatapd.loc[common_index]

# stackea todos los dataframes
tdv_max_stack=tdv_5_8_max.stack()
pk_stack=pk.stack()
bk_stack=bk.stack()
ctend_stack=max_ctend_diff.stack()
data_stack=valdatapd.stack()

# crea un índice de colores para representar los puntos según el valor de valdatapd
colors=['blue','green','red']

# crea una figura
plt.figure(figsize=(20,10))
# crea un scatter plot con pk en x, tdv_5_8_max en y y colores según valdatapd
plt.scatter(pk_stack,tdv_max_stack,c=data_stack.apply(lambda x: colors[int(x-1)]),alpha=alfa)

# añade etiquetas a los ejes
plt.xlabel('Nº de días desde el último cambio de tendencia')
plt.ylabel('Máximo actual')

# añade una leyenda en la que se indique que los colores azul, verde y rojo se corresponden a los valores 1, 2 y 3 de estrés
plt.legend(handles=[mpatches.Patch(color='blue', label='1'),mpatches.Patch(color='green', label='2'),mpatches.Patch(color='red', label='3')])

# guarda la figura
plt.savefig(save_folder+'tdv_max_vs_pk.png')

# cierra la figura
plt.close()

# crea una figura
plt.figure(figsize=(20,10))

# crea un scatter plot con bk en x, tdv_5_8_max en y y colores según valdatapd
plt.scatter(bk_stack,tdv_max_stack,c=data_stack.apply(lambda x: colors[int(x-1)]),alpha=alfa)

#añade etiquetas a los ejes
plt.xlabel('Valor decimal del binario de tendencias')
plt.ylabel('Máximo actual')

# añade una leyenda en la que se indique que los colores azul, verde y rojo se corresponden a los valores 1, 2 y 3 de estrés
plt.legend(handles=[mpatches.Patch(color='blue', label='1'),mpatches.Patch(color='green', label='2'),mpatches.Patch(color='red', label='3')])
# guarda la figura
plt.savefig(save_folder+'tdv_max_vs_bk.png')

# cierra la figura
plt.close()

# crea una figura
plt.figure(figsize=(20,10))

# crea un scatter plot con pk en x, ctend en y y colores según valdatapd
plt.scatter(pk_stack,ctend_stack,c=data_stack.apply(lambda x: colors[int(x-1)]),alpha=alfa)

# añade etiquetas a los ejes
plt.xlabel('Nº de días desde el último cambio de tendencia')
plt.ylabel('Diferencia entre el máximo actual y el máximo en el último cambio de tendencia')

# añade una leyenda en la que se indique que los colores azul, verde y rojo se corresponden a los valores 1, 2 y 3 de estrés
plt.legend(handles=[mpatches.Patch(color='blue', label='1'),mpatches.Patch(color='green', label='2'),mpatches.Patch(color='red', label='3')])
# guarda la figura
plt.savefig(save_folder+'ctend_vs_pk.png')

# cierra la figura
plt.close()

# crea una figura
plt.figure(figsize=(20,10))

# crea un scatter plot con bk en x, ctend en y y colores según valdatapd
plt.scatter(bk_stack,ctend_stack,c=data_stack.apply(lambda x: colors[int(x-1)]),alpha=alfa)

# añade etiquetas a los ejes
plt.xlabel('Valor decimal del binario de tendencias')
plt.ylabel('Diferencia entre el máximo actual y el máximo en el último cambio de tendencia')

# añade una leyenda en la que se indique que los colores azul, verde y rojo se corresponden a los valores 1, 2 y 3 de estrés
plt.legend(handles=[mpatches.Patch(color='blue', label='1'),mpatches.Patch(color='green', label='2'),mpatches.Patch(color='red', label='3')])
# guarda la figura
plt.savefig(save_folder+'ctend_vs_bk.png')

# cierra la figura
plt.close()

# obtiene el valor máximo de tdv_max_stack
tdv_max_max=tdv_max_stack.max()
# obtiene el valor máximo de ctend_stack
ctend_max=ctend_stack.max()
# obtiene el valor mínimo de tdv_max_stack
tdv_max_min=tdv_max_stack.min()
# obtiene el valor mínimo de ctend_stack
ctend_min=ctend_stack.min()
# obtiene el valor mínimo de pk_stack
pk_min=pk_stack.min()

# si no existe una subcarpeta de nombre pk en la carpeta de resultados, la crea
if not os.path.exists(save_folder+'pk/'):
    os.makedirs(save_folder+'pk/')

# por cada valor único de pk
for pk_value in pk_stack.unique():
    #obtiene los índices de los valores de pk que son iguales a pk_value
    pk_index=pk_stack[pk_stack==pk_value].index
    # crea una figura
    plt.figure(figsize=(20,10))
    # crea un scatter plot con tdv_5_8_max en x, ctend en y y colores según valdatapd
    plt.scatter(tdv_max_stack[pk_index],ctend_stack[pk_index],c=data_stack[pk_index].apply(lambda x: colors[int(x-1)]),alpha=alfa)
    # añade etiquetas a los ejes
    plt.xlabel('Máximo actual')
    plt.ylabel('Diferencia entre el máximo actual y el máximo en el último cambio de tendencia')
    # limita los ejes al valor máximo y mínimo de tdv_max_stack y ctend_stack añadiendo un 10% de margen
    plt.xlim(tdv_max_min-(tdv_max_max-tdv_max_min)*0.1,tdv_max_max+(tdv_max_max-tdv_max_min)*0.1)
    plt.ylim(ctend_min-(ctend_max-ctend_min)*0.1,ctend_max+(ctend_max-ctend_min)*0.1)
    # añade un título a la figura
    plt.title('Nº de días desde el último cambio de tendencia: '+str(pk_value))
    # añade una leyenda en la que se indique que los colores azul, verde y rojo se corresponden a los valores 1, 2 y 3 de estrés
    plt.legend(handles=[mpatches.Patch(color='blue', label='1'),mpatches.Patch(color='green', label='2'),mpatches.Patch(color='red', label='3')])
    # guarda la figura
    plt.savefig(save_folder+'pk/'+str(pk_value-pk_min)+'- pk '+str(pk_value)+'.png')
    # cierra la figura
    plt.close()

# si no existe una subcarpeta de nombre bk en la carpeta de resultados, la crea
if not os.path.exists(save_folder+'bk/'):
    os.makedirs(save_folder+'bk/')

# por cada valor único de bk
for bk_value in bk_stack.unique():
    #obtiene los índices de los valores de bk que son iguales a bk_value
    bk_index=bk_stack[bk_stack==bk_value].index
    # crea una figura
    plt.figure(figsize=(20,10))
    # crea un scatter plot con tdv_5_8_max en x, ctend en y y colores según valdatapd
    plt.scatter(tdv_max_stack[bk_index],ctend_stack[bk_index],c=data_stack[bk_index].apply(lambda x: colors[int(x-1)]),alpha=alfa)
    # añade etiquetas a los ejes
    plt.xlabel('Máximo actual')
    plt.ylabel('Diferencia entre el máximo actual y el máximo en el último cambio de tendencia')
    # limita los ejes al valor máximo y mínimo de tdv_max_stack y ctend_stack añadiendo un 10% de margen
    plt.xlim(tdv_max_min-(tdv_max_max-tdv_max_min)*0.1,tdv_max_max+(tdv_max_max-tdv_max_min)*0.1)
    plt.ylim(ctend_min-(ctend_max-ctend_min)*0.1,ctend_max+(ctend_max-ctend_min)*0.1)
    # añade un título a la figura
    plt.title('Valor decimal del binario de tendencias: '+str(bk_value))
    # añade una leyenda en la que se indique que los colores azul, verde y rojo se corresponden a los valores 1, 2 y 3 de estrés
    plt.legend(handles=[mpatches.Patch(color='blue', label='1'),mpatches.Patch(color='green', label='2'),mpatches.Patch(color='red', label='3')])
    # guarda la figura
    plt.savefig(save_folder+'bk/'+str(bk_value)+'.png')
    # cierra la figura
    plt.close()

#crea una figura
plt.figure(figsize=(20,10))
#crea un scatter plot con tdv_5_8_max en x, ctend en y y colores según valdatapd
plt.scatter(tdv_max_stack,ctend_stack,c=data_stack.apply(lambda x: colors[int(x-1)]),alpha=alfa)
# añade etiquetas a los ejes
plt.xlabel('Máximo actual')
plt.ylabel('Diferencia entre el máximo actual y el máximo en el último cambio de tendencia')
# añade una leyenda en la que se indique que los colores azul, verde y rojo se corresponden a los valores 1, 2 y 3 de estrés
plt.legend(handles=[mpatches.Patch(color='blue', label='1'),mpatches.Patch(color='green', label='2'),mpatches.Patch(color='red', label='3')])
# guarda la figura
plt.savefig(save_folder+'ctend_vs_tdv_max.png')
# cierra la figura
plt.close()