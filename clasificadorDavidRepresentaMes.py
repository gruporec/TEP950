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

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import isadoralib as isl
import sklearn.discriminant_analysis as skda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp

# usa agg
matplotlib.use('Agg')

year_train="2014"
desfase_estres_train=-1
years_val=["2015","2016","2019"]
desfases_estres_val=[1,1,0]
n = 3
save_folder = "ignore\\resultadosTDV\\batch\\Caracteristicas TDV\\Clasificador 1 dia ant\\"

# Ejecuta cargaRaw.py si no existe rawDiarios.csv o rawMinutales.csv
if not os.path.isfile("rawDiarios"+year_train+".csv") or not os.path.isfile("rawMinutales"+year_train+".csv"):
    os.system("python3 cargaRaw.py")

#datos de entrenamiento

# Carga de datos
tdv_train,ltp_train,meteo,trdatapd=isl.cargaDatosTDV(year_train,"")

#elimina los valores nan
tdv_train=tdv_train.dropna()
ltp_train=ltp_train.dropna()
meteo=meteo.dropna()
trdatapd=trdatapd.dropna()

# obtiene los valores de tdv entre las 5 y las 8 de cada dia
tdv_5_8 = tdv_train.between_time(time(5,0),time(8,0))
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
#crea otra copia más para hacer el cálculo como si n=1
bk1=tdv_5_8_max_diff_sign.copy()

#elimina valores nan
bk=bk.dropna()
bk_aux=bk_aux.dropna()
bk1=bk1.dropna()

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
trdatapd.index = trdatapd.index + pd.Timedelta(days=desfase_estres_train)

# convierte los índices de tdv_5_8_max, pk, bk, max_ctend_diff y valdatapd a datetime
tdv_5_8_max.index = pd.to_datetime(tdv_5_8_max.index)
pk.index = pd.to_datetime(pk.index)
bk.index = pd.to_datetime(bk.index)
bk1.index = pd.to_datetime(bk1.index)
max_ctend_diff.index = pd.to_datetime(max_ctend_diff.index)
trdatapd.index = pd.to_datetime(trdatapd.index)
#guarda una versión sin recortar de trdatapd
trdatapd_full=trdatapd.copy()

# recorta los dataframes tdv_5_8_max, pk, bk, bk1, max_ctend_diff y valdatapd para que tengan el mismo tamaño e índices
common_index = tdv_5_8_max.index.intersection(pk.index).intersection(bk.index).intersection(max_ctend_diff.index).intersection(trdatapd.index).intersection(bk1.index)
tdv_5_8_max = tdv_5_8_max.loc[common_index]
pk = pk.loc[common_index]
bk = bk.loc[common_index]
bk1 = bk1.loc[common_index]
max_ctend_diff = max_ctend_diff.loc[common_index]
trdatapd = trdatapd.loc[common_index]

# stackea todos los dataframes
tdv_max_stack=tdv_5_8_max.stack()
pk_stack=pk.stack()
bk_stack=bk.stack()
bk1_stack=bk1.stack()
ctend_stack=max_ctend_diff.stack()
data_stack_tr=trdatapd.stack()
tdv_train_stack=tdv_train.stack()
datafull_stack_tr=trdatapd_full.stack()

# crea un dataframe con los valores de tdv_max_stack, pk_stack, bk_stack y ctend_stack como columnas
#data_tr=pd.DataFrame({'tdv_max':tdv_max_stack.copy(),'pk':pk_stack.copy(),'bk':bk_stack.copy(),'ctend':ctend_stack.copy()})
# crea un dataframe con los valores de pk_stack, bk_stack, ctend_stack y bk1 como columnas
data_tr=pd.DataFrame({'pk':pk_stack.copy(),'bk':bk_stack.copy(),'ctend':ctend_stack.copy(),'bk1':bk1_stack.copy()})
# ordena las filas según el orden de data_stack_tr
data_tr=data_tr.loc[data_stack_tr.index]

# crea un clasificador QDA
clf=skda.QuadraticDiscriminantAnalysis()

# entrena el clasificador con los valores de data y los valores de data_stack
clf.fit(data_tr,data_stack_tr)

# Realiza la representación de los datos de entrenamiento
# por cada elemento único del segundo nivel del índice de data_stack_tr
for sens in data_stack_tr.index.get_level_values(1).unique():
    # por cada mes en el primer nivel del índice de data_stack_tr
    for month in data_stack_tr.index.get_level_values(0).month.unique():
        # obtiene los valores de data_stack_tr para el mes y el sensor
        data_stack_tr_month_sens=data_stack_tr.loc[data_stack_tr.index.get_level_values(0).month==month]
        data_stack_tr_month_sens=data_stack_tr_month_sens.loc[data_stack_tr_month_sens.index.get_level_values(1)==sens]
        # elimina el segundo nivel del índice
        data_stack_tr_month_sens.index=data_stack_tr_month_sens.index.droplevel(1)

        # obtiene los valores de datafull_stack_tr para el mes y el sensor
        datafull_stack_tr_month_sens=datafull_stack_tr.loc[datafull_stack_tr.index.get_level_values(0).month==month]
        datafull_stack_tr_month_sens=datafull_stack_tr_month_sens.loc[datafull_stack_tr_month_sens.index.get_level_values(1)==sens]
        # elimina el segundo nivel del índice
        datafull_stack_tr_month_sens.index=datafull_stack_tr_month_sens.index.droplevel(1)

        # obtiene los valores de tdv_train para el mes y el sensor
        tdv_train_sens=tdv_train_stack.loc[tdv_train_stack.index.get_level_values(1)==sens]
        tdv_train_month_sens=tdv_train_sens.loc[tdv_train_sens.index.get_level_values(0).month==month]
        # obtiene los valores de tdv_train para el mes anterior y el sensor
        tdv_train_month1_sens=tdv_train_sens.loc[tdv_train_sens.index.get_level_values(0).month==month-1]
        # elimina el segundo nivel del índice
        tdv_train_month_sens.index=tdv_train_month_sens.index.droplevel(1)
        tdv_train_month1_sens.index=tdv_train_month1_sens.index.droplevel(1)

        # añade el último día del mes anterior al dataframe teniendo en cuenta que son 1440 muestras por día
        tdv_train_month_sens=tdv_train_month_sens.append(tdv_train_month1_sens.iloc[-1440:])
        # ordena el dataframe por fecha
        tdv_train_month_sens=tdv_train_month_sens.sort_index()

        # obtiene los valores de data_tr para el mes y el sensor
        data_tr_month_sens=data_tr.loc[data_tr.index.get_level_values(0).month==month]
        data_tr_month_sens=data_tr_month_sens.loc[data_tr_month_sens.index.get_level_values(1)==sens]
        # elimina el segundo nivel del índice
        data_tr_month_sens.index=data_tr_month_sens.index.droplevel(1)

        # aplica el clasificador a los valores de data_stack_tr
        pred_tr_month_sens=clf.predict(data_tr_month_sens)
        # convierte los valores de pred_tr_month_sens a un dataframe con los índices de data_stack_tr_month_sens
        pred_tr_month_sens=pd.DataFrame(pred_tr_month_sens,index=data_stack_tr_month_sens.index)
        
        # crea una figura con la proporcion adecuada para un A4 horizontal
        fig=plt.figure(figsize=(11.69,8.27))
        # grafica los valores de data_stack_tr_month_sens
        plt.plot(tdv_train_month_sens)

        # obtiene los índices de la columna j del mes month en los que el día es par y la hora las 00:00:00
        days=tdv_train_month_sens.loc[(tdv_train_month_sens.index.day%2==0) & (tdv_train_month_sens.index.time==time(0,0,0))].index
        # por cada día par
        for day in days:
            # crea un área de color gris
            plt.axvspan(day,day+pd.Timedelta(days=1),color='gray',alpha=0.1)
            
        # obtiene los índices de los datos de test de la columna j del mes month en los que la hora es las 00:00:00
        days=tdv_train_month_sens.loc[(tdv_train_month_sens.index.month==month) & (tdv_train_month_sens.index.time==time(0,0,0))].index

        # por cada día
        for day in days:
            #añade un área de color verde entre las 5:00 y las 8:00
            plt.axvspan(day+pd.Timedelta(hours=5),day+pd.Timedelta(hours=8),color='green',alpha=0.1)

        
        #elimina la hora de days
        days=days.date

        # por cada día
        for day in days:
            #si existe, obtiene el valor de datafull_stack_tr_month_sens para el día
            if str(day) in datafull_stack_tr_month_sens.index:
                real=int(datafull_stack_tr_month_sens.loc[str(day)])
                xpos=pd.to_datetime(str(day)+" 12:00:00",format="%Y-%m-%d %H:%M:%S")
                #escribe el valor de real en el gráfico con un recuadro de color rojo, verde o azul dependiendo de si es 3, 2 o 1
                if real==1:
                    plt.text(xpos,tdv_train_month_sens.min(),str(real),horizontalalignment='center',verticalalignment='bottom',fontsize=8,color='blue')
                elif real==2:
                    plt.text(xpos,tdv_train_month_sens.min(),str(real),horizontalalignment='center',verticalalignment='bottom',fontsize=8,color='green')
                elif real==3:
                    plt.text(xpos,tdv_train_month_sens.min(),str(real),horizontalalignment='center',verticalalignment='bottom',fontsize=8,color='red')
            #si existe, obtiene el valor de pred_tr_month_sens para el día
            if str(day) in pred_tr_month_sens.index:
                pred=int(pred_tr_month_sens.loc[str(day)])
                xpos=pd.to_datetime(str(day)+" 12:00:00",format="%Y-%m-%d %H:%M:%S")
                #escribe el valor de pred en el gráfico con una bbox de color rojo, verde o azul dependiendo de si es 3, 2 o 1
                if pred==1:
                    plt.text(xpos,tdv_train_month_sens.min(),str(pred),horizontalalignment='center',verticalalignment='top',fontsize=8,color='blue')
                elif pred==2:
                    plt.text(xpos,tdv_train_month_sens.min(),str(pred),horizontalalignment='center',verticalalignment='top',fontsize=8,color='green')
                elif pred==3:
                    plt.text(xpos,tdv_train_month_sens.min(),str(pred),horizontalalignment='center',verticalalignment='top',fontsize=8,color='red')
        # añade un título al gráfico
        plt.title("Sensor: "+str(sens)+" mes: "+str(month))
        # añade una leyenda al gráfico que indique dos X en negro, junto a la de arriba el texto "valor predicho" y junto a la de abajo, "valor real"
        legend_elements = [matplotlib.lines.Line2D([0], [0], marker='x', linestyle='None', label='Valor real', color='k', markersize=8),matplotlib.lines.Line2D([0], [0], marker='x', linestyle='None', label='Valor predicho', color='k', markersize=8)]
        #añade la leyenda en la esquina superior izquierda
        plt.legend(handles=legend_elements, loc="upper left")
        # si no existe una carpeta en save_folder/año/sensor/, la crea
        if not os.path.exists(save_folder+str(year_train)+"/"+str(sens)+"/"):
            os.makedirs(save_folder+str(year_train)+"/"+str(sens)+"/")
        # guarda el gráfico en save_folder/año/sensor/mes.png
        plt.savefig(save_folder+str(year_train)+"/"+str(sens)+"/"+f'{month:02}'+".png")
        # cierra la figura
        plt.close(fig)


# datos de validación
# crea una lista en blanco para añadir resultados
resultados_val=[]
# crea una lista en blanco para guardar el número de datos de cada año
num_datos_val=[]

# por cada año de la lista
for i in range(0,len(years_val),1):
    # año de validación
    year_val=years_val[i]
    # desfase
    desfase_estres_val=desfases_estres_val[i]

    # Carga de datos
    tdv_val,ltp_val,meteo,valdatapd=isl.cargaDatosTDV(year_val,"")

    #elimina los valores nan
    tdv_val=tdv_val.dropna()
    ltp_val=ltp_val.dropna()
    meteo=meteo.dropna()
    valdatapd=valdatapd.dropna()

    # obtiene los valores de tdv entre las 5 y las 8 de cada dia
    tdv_5_8 = tdv_val.between_time(time(5,0),time(8,0))
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
    
    #crea otra copia más para hacer el cálculo como si n=1
    bk1=tdv_5_8_max_diff_sign.copy()
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
    bk1=bk1.dropna()

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
    valdatapd.index = valdatapd.index + pd.Timedelta(days=desfase_estres_val)

    # convierte los índices de tdv_5_8_max, pk, bk, max_ctend_diff y valdatapd a datetime
    tdv_5_8_max.index = pd.to_datetime(tdv_5_8_max.index)
    pk.index = pd.to_datetime(pk.index)
    bk.index = pd.to_datetime(bk.index)
    bk1.index = pd.to_datetime(bk1.index)
    max_ctend_diff.index = pd.to_datetime(max_ctend_diff.index)
    valdatapd.index = pd.to_datetime(valdatapd.index)
    #guarda una versión sin recortar de valdatapd
    valdatapd_full=valdatapd.copy()

    # recorta los dataframes tdv_5_8_max, pk, bk, bk1, max_ctend_diff y valdatapd para que tengan el mismo tamaño e índices
    common_index = tdv_5_8_max.index.intersection(pk.index).intersection(bk.index).intersection(max_ctend_diff.index).intersection(valdatapd.index).intersection(bk1.index)
    tdv_5_8_max = tdv_5_8_max.loc[common_index]
    pk = pk.loc[common_index]
    bk = bk.loc[common_index]
    bk1 = bk1.loc[common_index]
    max_ctend_diff = max_ctend_diff.loc[common_index]
    valdatapd = valdatapd.loc[common_index]
    valdatapd_full = valdatapd_full.loc[common_index]

    # stackea todos los dataframes
    tdv_max_stack=tdv_5_8_max.stack()
    pk_stack=pk.stack()
    bk_stack=bk.stack()
    bk1_stack=bk1.stack()
    ctend_stack=max_ctend_diff.stack()
    data_stack_val=valdatapd.stack()
    tdv_val_stack=tdv_val.stack()
    datafull_stack_val=valdatapd_full.stack()

    # crea un dataframe con los valores de tdv_max_stack, pk_stack, bk_stack y ctend_stack como columnas
    #data_val=pd.DataFrame({'tdv_max':tdv_max_stack.copy(),'pk':pk_stack.copy(),'bk':bk_stack.copy(),'ctend':ctend_stack.copy()})
    data_val=pd.DataFrame({'pk':pk_stack.copy(),'bk':bk_stack.copy(),'ctend':ctend_stack.copy(),'bk1':bk1_stack.copy()})


    # Realiza la representación de los datos de validación
    # por cada elemento único del segundo nivel del índice de data_stack_val
    for sens in data_stack_val.index.get_level_values(1).unique():
        # por cada mes en el primer nivel del índice de data_stack_val
        for month in data_stack_val.index.get_level_values(0).month.unique():
            # obtiene los valores de data_stack_val para el mes y el sensor
            data_val_month_sens=data_val.loc[data_val.index.get_level_values(0).month==month]
            data_val_month_sens=data_val_month_sens.loc[data_val_month_sens.index.get_level_values(1)==sens]
            # elimina el segundo nivel del índice
            data_val_month_sens.index=data_val_month_sens.index.droplevel(1)

            # obtiene los valores de datafull_stack_val para el mes y el sensor
            datafull_val_month_sens=datafull_stack_val.loc[datafull_stack_val.index.get_level_values(0).month==month]
            datafull_val_month_sens=datafull_val_month_sens.loc[datafull_val_month_sens.index.get_level_values(1)==sens]
            # elimina el segundo nivel del índice
            datafull_val_month_sens.index=datafull_val_month_sens.index.droplevel(1)

            # obtiene los valores de tdv_val para el mes y el sensor
            tdv_val_sens=tdv_val_stack.loc[tdv_val_stack.index.get_level_values(1)==sens]
            tdv_val_month_sens=tdv_val_sens.loc[tdv_val_sens.index.get_level_values(0).month==month]
            # obtiene los valores de tdv_val para el mes anterior y el sensor
            tdv_val_month1_sens=tdv_val_sens.loc[tdv_val_sens.index.get_level_values(0).month==month-1]
            
            # elimina el segundo nivel del índice
            tdv_val_month_sens.index=tdv_val_month_sens.index.droplevel(1)
            tdv_val_month1_sens.index=tdv_val_month1_sens.index.droplevel(1)

            # añade el último día del mes anterior al dataframe teniendo en cuenta que son 1440 muestras por día
            tdv_val_month_sens=tdv_val_month_sens.append(tdv_val_month1_sens.iloc[-1440:])
            # ordena el dataframe por fecha
            tdv_val_month_sens=tdv_val_month_sens.sort_index()


            # obtiene los valores de data_val para el mes y el sensor
            data_val_month_sens=data_val.loc[data_val.index.get_level_values(0).month==month]
            data_val_month_sens=data_val_month_sens.loc[data_val_month_sens.index.get_level_values(1)==sens]
            # elimina el segundo nivel del índice
            data_val_month_sens.index=data_val_month_sens.index.droplevel(1)

            # aplica el clasificador a los valores de data_stack_val
            pred_val_month_sens=clf.predict(data_val_month_sens)
            # convierte los valores de pred_val_month_sens a un dataframe con los índices de data_stack_val_month_sens
            pred_val_month_sens=pd.DataFrame(pred_val_month_sens,index=data_val_month_sens.index)
            
            # crea una figura con la proporcion adecuada para un A4 horizontal
            fig=plt.figure(figsize=(11.69,8.27))
            # grafica los valores de data_stack_val_month_sens
            plt.plot(tdv_val_month_sens)

            # obtiene los índices de la columna j del mes month en los que el día es par y la hora las 00:00:00
            days=tdv_val_month_sens.loc[(tdv_val_month_sens.index.day%2==0) & (tdv_val_month_sens.index.time==time(0,0,0))].index
            # por cada día par
            for day in days:
                # crea un área de color gris
                plt.axvspan(day,day+pd.Timedelta(days=1),color='gray',alpha=0.1)
                
            # obtiene los índices de los datos de test de la columna j del mes month en los que la hora es las 00:00:00
            days=tdv_val_month_sens.loc[(tdv_val_month_sens.index.month==month) & (tdv_val_month_sens.index.time==time(0,0,0))].index

            # por cada día
            for day in days:
                #añade un área de color verde entre las 5:00 y las 8:00
                plt.axvspan(day+pd.Timedelta(hours=5),day+pd.Timedelta(hours=8),color='green',alpha=0.1)

            
            #elimina la hora de days
            days=days.date

            # por cada día
            for day in days:
                #si existe, obtiene el valor de datafull_val_month_sens para el día
                if str(day) in datafull_val_month_sens.index:
                    real=int(datafull_val_month_sens.loc[str(day)])
                    xpos=pd.to_datetime(str(day)+" 12:00:00",format="%Y-%m-%d %H:%M:%S")
                    #escribe el valor de real en el gráfico con un recuadro de color rojo, verde o azul dependiendo de si es 3, 2 o 1
                    if real==1:
                        plt.text(xpos,tdv_val_month_sens.min(),str(real),horizontalalignment='center',verticalalignment='bottom',fontsize=8,color='blue')
                    elif real==2:
                        plt.text(xpos,tdv_val_month_sens.min(),str(real),horizontalalignment='center',verticalalignment='bottom',fontsize=8,color='green')
                    elif real==3:
                        plt.text(xpos,tdv_val_month_sens.min(),str(real),horizontalalignment='center',verticalalignment='bottom',fontsize=8,color='red')
                #si existe, obtiene el valor de pred_tr_month_sens para el día
                if str(day) in pred_val_month_sens.index:
                    pred=int(pred_val_month_sens.loc[str(day)])
                    xpos=pd.to_datetime(str(day)+" 12:00:00",format="%Y-%m-%d %H:%M:%S")
                    #escribe el valor de pred en el gráfico con una bbox de color rojo, verde o azul dependiendo de si es 3, 2 o 1
                    if pred==1:
                        plt.text(xpos,tdv_val_month_sens.min(),str(pred),horizontalalignment='center',verticalalignment='top',fontsize=8,color='blue')
                    elif pred==2:
                        plt.text(xpos,tdv_val_month_sens.min(),str(pred),horizontalalignment='center',verticalalignment='top',fontsize=8,color='green')
                    elif pred==3:
                        plt.text(xpos,tdv_val_month_sens.min(),str(pred),horizontalalignment='center',verticalalignment='top',fontsize=8,color='red')
            # añade un título al gráfico
            plt.title("Sensor: "+str(sens)+" mes: "+str(month))
            # añade una leyenda al gráfico que indique dos X en negro, junto a la de arriba el texto "valor predicho" y junto a la de abajo, "valor real"
            legend_elements = [matplotlib.lines.Line2D([0], [0], marker='x', linestyle='None', label='Valor real', color='k', markersize=8),matplotlib.lines.Line2D([0], [0], marker='x', linestyle='None', label='Valor predicho', color='k', markersize=8)]
            #añade la leyenda en la esquina superior izquierda
            plt.legend(handles=legend_elements, loc="upper left")

            # si no existe una carpeta en save_folder/año/sensor/, la crea
            if not os.path.exists(save_folder+str(year_val)+"/"+str(sens)+"/"):
                os.makedirs(save_folder+str(year_val)+"/"+str(sens)+"/")
            # guarda el gráfico en save_folder/año/sensor/mes.png
            plt.savefig(save_folder+str(year_val)+"/"+str(sens)+"/"+f'{month:02}'+".png")
            # cierra la figura
            plt.close(fig)