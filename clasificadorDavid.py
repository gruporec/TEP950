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

year_train="2014"
desfase_estres_train=-1
years_val=["2015","2016"]
desfases_estres_val=[1,1]
n_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]

# Ejecuta cargaRaw.py si no existe rawDiarios.csv o rawMinutales.csv
if not os.path.isfile("rawDiarios"+year_train+".csv") or not os.path.isfile("rawMinutales"+year_train+".csv"):
    os.system("python3 cargaRaw.py")

#crea un dataframe vacio para guardar los resultados con una columna para el año de entrenamiento y una para cada año de validación, y una más para la media de validación
resultados=pd.DataFrame()

for n in n_list:
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

    # aplica el clasificador a los valores de data de entrenamiento
    pred_tr=clf.predict(data_tr)

    # calcula el acierto balanceado
    acierto_tr=skmetrics.balanced_accuracy_score(data_stack_tr,pred_tr)

    # añade el acierto al dataframe de resultados en la columna year_train y la fila n
    resultados.loc[n,year_train]=acierto_tr

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

        # recorta los dataframes tdv_5_8_max, pk, bk, bk1, max_ctend_diff y valdatapd para que tengan el mismo tamaño e índices
        common_index = tdv_5_8_max.index.intersection(pk.index).intersection(bk.index).intersection(max_ctend_diff.index).intersection(valdatapd.index).intersection(bk1.index)
        tdv_5_8_max = tdv_5_8_max.loc[common_index]
        pk = pk.loc[common_index]
        bk = bk.loc[common_index]
        bk1 = bk1.loc[common_index]
        max_ctend_diff = max_ctend_diff.loc[common_index]
        valdatapd = valdatapd.loc[common_index]

        # stackea todos los dataframes
        tdv_max_stack=tdv_5_8_max.stack()
        pk_stack=pk.stack()
        bk_stack=bk.stack()
        bk1_stack=bk1.stack()
        ctend_stack=max_ctend_diff.stack()
        data_stack_val=valdatapd.stack()

        # crea un dataframe con los valores de tdv_max_stack, pk_stack, bk_stack y ctend_stack como columnas
        #data_val=pd.DataFrame({'tdv_max':tdv_max_stack.copy(),'pk':pk_stack.copy(),'bk':bk_stack.copy(),'ctend':ctend_stack.copy()})
        data_val=pd.DataFrame({'pk':pk_stack.copy(),'bk':bk_stack.copy(),'ctend':ctend_stack.copy(),'bk1':bk1_stack.copy()})

        # aplica el clasificador a los valores de data de validación
        pred_val=clf.predict(data_val)

        acc=skmetrics.balanced_accuracy_score(data_stack_val,pred_val)

        print('n: '+str(n))
        print('Año: '+str(year_val))
        print('Matriz de confusión:')
        print(skmetrics.confusion_matrix(data_stack_val,pred_val))
        print('Acierto balanceado:')
        print(acc)
        print('')

        # añade el acierto a la lista de resultados
        resultados_val.append(acc)

        # añade el acierto al dataframe de resultados
        resultados.loc[n,year_val]=acc

        # añade el número de datos de este año a la lista
        num_datos_val.append(len(data_stack_val))

    # calcula el acierto medio ponderado por el número de datos
    acc_medio=np.average(resultados_val,weights=num_datos_val)

    # añade el acierto medio al dataframe de resultados
    resultados.loc[n,'media']=acc_medio

    print('Acierto medio con n='+str(n)+':')
    print(acc_medio)
    print('')

print(resultados)

# grafica los resultados
resultados.plot()
plt.show()
