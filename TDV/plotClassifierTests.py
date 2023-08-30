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
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp

# add the path to the lib folder to the system path. This allows to import the library, and makes it so that the script runs from the main folder, at least when running using VSCode
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
# import the isadoralib library. Some IDEs may mark it as a missing file (tested on VSCode), but it will still work thanks to the previous line.
import isadoralib as isl

matplotlib.use("Agg")

years=["2014","2015","2016","2019"]
desfases_estres=[-1,0,0,0]
n=8
alfa=0.25

for iyear in range(len(years)):
    year=years[iyear]
    desfase_estres=desfases_estres[iyear]
    save_folder = './ignore/resultadosTDV/batch/Caracteristicas TDV v3/'+year+'/'

    # if the folder doesn't exist, create it
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # # Run cargaRaw.py if the raw data files don't exist (not working for now)
    # if not os.path.isfile("../rawDiarios"+year+".csv") or not os.path.isfile("../rawMinutales"+year+".csv"):
    #     os.system("python3 cargaRaw.py")

    # Load the data
    tdv,ltp,meteo,valdatapd=isl.cargaDatosTDV(year,"")

    # Delete the rows with nan values
    tdv=tdv.dropna()
    ltp=ltp.dropna()
    meteo=meteo.dropna()
    valdatapd=valdatapd.dropna()

    # Get the values of tdv between 5 and 8 of each day
    tdv_5_8 = tdv.between_time(time(5,0),time(8,0))
    # Get the maximum value of tdv between 5 and 8 of each day
    tdv_5_8_max = tdv_5_8.groupby(tdv_5_8.index.date).max()
    # Get the difference between the maximum of each day and the maximum of the previous day
    tdv_5_8_max_diff = tdv_5_8_max.diff(periods=1).dropna()
    # Get the sign of the difference between the maximum of each day and the maximum of the previous day
    tdv_5_8_max_diff_sign = tdv_5_8_max_diff.apply(np.sign)
    # Replace the negative values with 0
    tdv_5_8_max_diff_sign[tdv_5_8_max_diff_sign<0]=0
    # Create a dataframe that is 1 when tdv_5_8_max_diff_sign is 0 and 0 when it is 1
    tdv_5_8_max_diff_sign_inv=tdv_5_8_max_diff_sign.apply(lambda x: 1-x)

    # Create two dataframes with the size of tdv_5_8_max_diff_sign and values 0
    pk0=pd.DataFrame(np.zeros(tdv_5_8_max_diff_sign.shape),index=tdv_5_8_max_diff_sign.index,columns=tdv_5_8_max_diff_sign.columns)
    pk1=pd.DataFrame(np.zeros(tdv_5_8_max_diff_sign.shape),index=tdv_5_8_max_diff_sign.index,columns=tdv_5_8_max_diff_sign.columns)

    # For each day in tdv_5_8_max_diff_sign
    for i in tdv_5_8_max_diff_sign.index:
        # If it is the first row
        if i==tdv_5_8_max_diff_sign.index[0]:
            # Add to pk0 the value of tdv_5_8_max_diff_sign_inv
            pk0.loc[i]=tdv_5_8_max_diff_sign_inv.loc[i]
            # Add to pk1 the value of tdv_5_8_max_diff_sign
            pk1.loc[i]=tdv_5_8_max_diff_sign.loc[i]
        # If it is not the first row
        else:
            # Get the previous index by subtracting one day
            i_ant=i-pd.Timedelta(days=1)
            # Add to pk0 the value of the previous row of pk0 plus the value of the row of tdv_5_8_max_diff_sign_inv, multiplied by the value of the row of tdv_5_8_max_diff_sign_inv
            pk0.loc[i]=(pk0.loc[i_ant]+tdv_5_8_max_diff_sign_inv.loc[i])*tdv_5_8_max_diff_sign_inv.loc[i]
            # Add to pk1 the value of the previous row of pk1 plus the value of the row of tdv_5_8_max_diff_sign, multiplied by the value of the row of tdv_5_8_max_diff_sign
            pk1.loc[i]=(pk1.loc[i_ant]+tdv_5_8_max_diff_sign.loc[i])*tdv_5_8_max_diff_sign.loc[i]
    # Substract pk0 from pk1
    pk=pk1-pk0

    # Create a copy of tdv_5_8_max_diff_sign
    bk=tdv_5_8_max_diff_sign.copy()
    # Create another copy of tdv_5_8_max_diff_sign to use as an auxiliary
    bk_aux=tdv_5_8_max_diff_sign.copy()
    # Remove nan values
    bk=bk.dropna()
    bk_aux=bk_aux.dropna()

    # Repeat n-1 times
    for i in range(1,n,1):
        # Move bk_aux one day forward
        bk_aux.index = bk_aux.index + pd.Timedelta(days=1)
        # Duplicate the value of pk
        bk=bk*2
        # Add the value of pk_aux to pk
        bk=bk+bk_aux

    # Remove nan values
    bk=bk.dropna()

    # Create a dataframe with diff tdv_5_8_max_diff_sign that represents the changes of trend
    ctend=pd.DataFrame(tdv_5_8_max_diff_sign.diff(periods=1).dropna())

    # Make non-null values equal to 1
    ctend[ctend!=0]=1

    # Get the values of the maximum in the time slot when there is a change of trend
    max_ctend=tdv_5_8_max[ctend!=0]
    # Fill null values with the previous value
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

    # recorta también las columnas
    common_cols=tdv_5_8_max.columns.intersection(pk.columns).intersection(bk.columns).intersection(max_ctend_diff.columns).intersection(valdatapd.columns)
    tdv_5_8_max = tdv_5_8_max[common_cols]
    pk = pk[common_cols]
    bk = bk[common_cols]
    max_ctend_diff = max_ctend_diff[common_cols]
    valdatapd = valdatapd[common_cols]

    # stackea todos los dataframes
    tdv_max_stack=tdv_5_8_max.stack()
    pk_stack=pk.stack()
    bk_stack=bk.stack()
    ctend_stack=max_ctend_diff.stack()
    data_stack=valdatapd.stack()

    # crea un índice de colores para representar los puntos según el valor de valdatapd
    colors=['blue','green','red']
    color_stack=data_stack.apply(lambda x: colors[int(x-1)])

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