import sys
from time import time
import traceback
from matplotlib.markers import MarkerStyle
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import time
import isadoralib as isl
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp
import multiprocessing as mp
import matplotlib.patches as mpatches

matplotlib.use('Agg')


years_test = ["2014","2015","2016","2019"]

load_folder = 'ignore/resultadosTDV/batch/PCALDA6am/IMG/AnalisisID80/'
save_folder = 'ignore/resultadosTDV/batch/PCALDA6am/IMG/AnalisisID80/mensual/'
if not os.path.exists(load_folder):
    os.makedirs(load_folder)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#crea una carpeta para guardar los resultados si no existe por cada año de test
for year_test in years_test:
    if not os.path.exists(save_folder+year_test):
        os.makedirs(save_folder+year_test)



#crea dos listas vacías para los datos de test
tdv_tests=[]
data_tests=[]
data_results=[]
#por cada año de test 
for year_test in years_test:

    if not os.path.isfile("rawDiarios"+year_test+".csv") or not os.path.isfile("rawMinutales"+year_test+".csv"):
        os.system("python3 cargaRaw.py")
    #carga los datos de test
    tdv_test,ltp_test,meteo_test,data_test=isl.cargaDatosTDV(year_test,"rht")
    #añade los datos de test a las listas
    tdv_tests.append(tdv_test.copy())
    data_tests.append(data_test.copy())
    #Carga el archivo de resultados del clasificador conviertiendo las dos primeras columnas en índice
    results=pd.read_csv(load_folder+year_test+"_data_test_pred_proba.csv",index_col=[0,1])
    #unstackea el segundo índice
    results=results.unstack(level=1)
    #convierte el índice en datetime con formato año-mes-día
    #results.index=pd.to_datetime(results.index,format="%Y-%m-%d")
    #vuelve a stackear el segundo índice
    results=results.stack(level=1)
    #añade los resultados a la lista
    data_results.append(results.copy())

# por cada elemento de tdv_tests
for i in range(len(tdv_tests)):
    # por cada columna de tdv_tests[i]
    for j in range(len(tdv_tests[i].columns)):
        #por cada mes en el índice de tdv_tests[i]
        for month in tdv_tests[i].index.month.unique():
            # crea una figura con el tamaño adecuado para un A4 horizontal
            plt.figure(figsize=(11.69,8.27))
            # extrae el dataframe que contiene los datos de la columna j del mes month como copia
            tdv_test_month=tdv_tests[i].loc[tdv_tests[i].index.month==month].iloc[:,j].copy()
            # añade el último día del mes anterior al dataframe
            tdv_test_month=tdv_test_month.append(tdv_tests[i].loc[tdv_tests[i].index.month==month-1].iloc[:,j].copy().tail(1440))
            # ordena los datos por fecha
            tdv_test_month=tdv_test_month.sort_index()

            # grafica los datos de tdv_tests[i] de la columna j del mes month
            plt.plot(tdv_test_month,label=tdv_tests[i].columns[j])
            # obtiene los índices de la columna j del mes month en los que el día es par y la hora las 00:00:00
            days=tdv_test_month.loc[(tdv_test_month.index.day%2==0) & (tdv_test_month.index.time==time(0,0,0))].index
            # por cada día par
            for day in days:
                # crea un área de color gris
                plt.axvspan(day,day+pd.Timedelta(days=1),color='gray',alpha=0.1)
            # obtiene el nombre de la columna j
            col=tdv_tests[i].columns[j]
            # obtiene los índices de los datos de test de la columna j del mes month en los que la hora es las 00:00:00
            days=data_tests[i].loc[(data_tests[i].index.month==month) & (data_tests[i].index.time==time(0,0,0))].index

            # por cada día
            for day in days:
                #añade un área de color verde entre las 5:00 y las 8:00
                plt.axvspan(day+pd.Timedelta(hours=5),day+pd.Timedelta(hours=8),color='green',alpha=0.1)

            #elimina la hora de days
            days=days.date

            # por cada día
            for day in days:
                #si existe, obtiene el valor de la columna "real" de results correspondiente al día day en el primer índice y la columna j en el segundo índice
                if (str(day),col) in data_results[i].index:
                    real=int(data_results[i].loc[(str(day),col),"real"])
                    xpos=pd.to_datetime(str(day)+" 12:00:00",format="%Y-%m-%d %H:%M:%S")
                    #escribe el valor de real en el gráfico con un recuadro de color rojo, verde o azul dependiendo de si es 3, 2 o 1
                    if real==1:
                        plt.text(xpos,tdv_test_month.min(),str(real),horizontalalignment='center',verticalalignment='bottom',fontsize=8,color='blue')
                    elif real==2:
                        plt.text(xpos,tdv_test_month.min(),str(real),horizontalalignment='center',verticalalignment='bottom',fontsize=8,color='green')
                    elif real==3:
                        plt.text(xpos,tdv_test_month.min(),str(real),horizontalalignment='center',verticalalignment='bottom',fontsize=8,color='red')
                #si existe, obtiene el valor de la columna "pred" de results correspondiente al día day en el primer índice y la columna j en el segundo índice
                if (str(day),col) in data_results[i].index:
                    pred=int(data_results[i].loc[(str(day),col),"pred"])
                    xpos=pd.to_datetime(str(day)+" 12:00:00",format="%Y-%m-%d %H:%M:%S")
                    #escribe el valor de pred en el gráfico con una bbox de color rojo, verde o azul dependiendo de si es 3, 2 o 1
                    if pred==1:
                        plt.text(xpos,tdv_test_month.min(),str(pred),horizontalalignment='center',verticalalignment='top',fontsize=8,color='blue')
                    elif pred==2:
                        plt.text(xpos,tdv_test_month.min(),str(pred),horizontalalignment='center',verticalalignment='top',fontsize=8,color='green')
                    elif pred==3:
                        plt.text(xpos,tdv_test_month.min(),str(pred),horizontalalignment='center',verticalalignment='top',fontsize=8,color='red')
                # añade un título al gráfico
            plt.title("Sensor: "+str(col)+" mes: "+str(month))
            # añade una leyenda al gráfico que indique dos X en negro, junto a la de arriba el texto "valor predicho" y junto a la de abajo, "valor real"
            legend_elements = [matplotlib.lines.Line2D([0], [0], marker='x', linestyle='None', label='Valor real', color='k', markersize=8),matplotlib.lines.Line2D([0], [0], marker='x', linestyle='None', label='Valor predicho', color='k', markersize=8)]
            #añade la leyenda en la esquina superior izquierda
            plt.legend(handles=legend_elements, loc="upper left")


            # crea una carpeta para guardar los gráficos si no existe
            if not os.path.isdir(save_folder+years_test[i]+"/"+str(tdv_tests[i].columns[j])):
                os.makedirs(save_folder+years_test[i]+"/"+str(tdv_tests[i].columns[j]))

            # guarda el gráfico
            plt.savefig(save_folder+years_test[i]+"/"+str(tdv_tests[i].columns[j])+"/"+f'{month:02}'+".png")

            # cierra el gráfico
            plt.close()