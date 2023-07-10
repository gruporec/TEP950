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


#years_test = ["2014","2015","2016","2019"]
#years_test = ["2015"]
years_test = ["2014","2015","2016","2019"]
              
save_folder = 'ignore/resultadosTDV/batch/PCALDA6am/IMG/OriginalRiegoDatosNuevosColor/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#crea una carpeta para guardar los resultados si no existe por cada año de test
for year_test in years_test:
    if not os.path.exists(save_folder+year_test):
        os.makedirs(save_folder+year_test)



#crea dos listas vacías para los datos de test
tdv_tests=[]
data_tests=[]
#por cada año de test 
for year_test in years_test:

    if not os.path.isfile("rawDiarios"+year_test+".csv") or not os.path.isfile("rawMinutales"+year_test+".csv"):
        os.system("python3 cargaRaw.py")
    #carga los datos de test
    tdv_test,ltp_test,meteo_test,data_test=isl.cargaDatosTDV(year_test,"rht")
    #añade los datos de test a las listas
    tdv_tests.append(tdv_test.copy())
    data_tests.append(data_test.copy())


# por cada elemento de tdv_tests
for i in range(len(tdv_tests)):
    year_test=years_test[i]
    # por cada columna de tdv_tests[i]
    for j in range(len(tdv_tests[i].columns)):
        # obtiene el valor máximo de la columna j de tdv_tests[i]
        max_an=tdv_tests[i].iloc[:,j].max()
        # obtiene el valor mínimo de la columna j de tdv_tests[i]
        min_an=tdv_tests[i].iloc[:,j].min()

        #por cada mes en el índice de tdv_tests[i]
        for month in tdv_tests[i].index.month.unique():
            # crea una figura con el tamaño adecuado para un A4 horizontal
            plt.figure(figsize=(11.69,8.27))
            # extrae el dataframe que contiene los datos de la columna j del mes month como copia
            tdv_test_month=tdv_tests[i].loc[tdv_tests[i].index.month==month].iloc[:,j].copy()
            # añade el último día del mes anterior al dataframe
            #tdv_test_month=tdv_test_month.append(tdv_tests[i].loc[tdv_tests[i].index.month==month-1].iloc[:,j].copy().tail(1440))
            # ordena los datos por fecha
            tdv_test_month=tdv_test_month.sort_index()

            # obtiene el nombre de la columna j
            col=tdv_tests[i].columns[j]

            # obtiene los días de los índices de tdv_tests[i] de la columna j del mes month
            days=tdv_tests[i].loc[tdv_tests[i].index.month==month].index.day.unique()

            # obtiene las fechas de los índices de tdv_tests[i] de la columna j del mes month añadiendo el año y el mes
            days_date=[str(year_test)+"-"+str(month)+"-"+str(day) for day in days]
            # por cada día
            for iday in range(len(days)):
                day=days[iday]
                day_date=days_date[iday]
                #si existe, obtiene el valor de la columna "real" correspondiente al día day en el primer índice y la columna j en el segundo índice
                if (str(day_date),col) in data_tests[i].stack().index:
                    real=str(int(data_tests[i].stack().loc[(str(day_date),col)]))
                else:
                    real=""

                # ordena los datos de tdv_test_month del día day por hora
                tdv_test_month_day=tdv_test_month.loc[tdv_test_month.index.day==day].sort_index().dropna()

                # grafica los datos de tdv_tests[i] de la columna j del mes month del día day con el color correspondiente a real
                if real=="1":
                    plt.plot(tdv_test_month_day,label=tdv_tests[i].columns[j],color='blue')
                elif real=="2":
                    plt.plot(tdv_test_month_day,label=tdv_tests[i].columns[j],color='green')
                elif real=="3":
                    plt.plot(tdv_test_month_day,label=tdv_tests[i].columns[j],color='red')
                else:
                    plt.plot(tdv_test_month_day,label=tdv_tests[i].columns[j],color='black')
                #plt.plot(tdv_test_month,label=tdv_tests[i].columns[j])

            #limita el eje vertical al máximo anual por arriba y al mínimo anual más un 5% por abajo
            plt.ylim(min_an-(max_an-min_an)*0.05,max_an)
            bottom, top = plt.ylim()
            #amplia el eje vertical hacia abajo para que quepa texto
            # bottom, top = plt.ylim()
            # plt.ylim(bottom-(top-bottom)*0.1,top)

            # obtiene los índices de la columna j del mes month en los que el día es par y la hora las 00:00:00
            days=tdv_test_month.loc[(tdv_test_month.index.day%2==0) & (tdv_test_month.index.time==time(0,0,0))].index
            # por cada día par
            for day in days:
                # crea un área de color gris
                plt.axvspan(day,day+pd.Timedelta(days=1),color='gray',alpha=0.1)


            # obtiene los índices de los datos de test de la columna j del mes month
            days=data_tests[i].loc[(data_tests[i].index.month==month)].index

            # por cada día
            for day in days:
                #añade un área de color verde entre las 5:00 y las 8:00
                plt.axvspan(day+pd.Timedelta(hours=5),day+pd.Timedelta(hours=8),color='green',alpha=0.1)

            # obtiene los índices de data_tests
            days=data_tests[i].loc[data_tests[i].index.month==month].index
            days_date=days.date

            # por cada día
            for iday in range(len(days)):
                day=days[iday]
                day_date=days_date[iday]

                #si existe, obtiene el valor de la columna "real" correspondiente al día day en el primer índice y la columna j en el segundo índice
                if (str(day_date),col) in data_tests[i].stack().index:
                    real=str(int(data_tests[i].stack().loc[(str(day_date),col)]))
                else:
                    real=""
                
                xpos=pd.to_datetime(str(day_date)+" 12:00:00",format="%Y-%m-%d %H:%M:%S")
                #si hay datos en tdv_test_month, asigna el mínimo a ypos
                if day in tdv_test_month.index:
                    ypos=min_an-(max_an-min_an)*0.03
                #si no, asigna 0 a ypos
                else:
                    ypos=0


                day_text=real
                #escribe el valor de real en el gráfico con un recuadro de color rojo, verde o azul dependiendo de si es 3, 2 o 1
                if real=="1":
                    plt.text(xpos,ypos,day_text,horizontalalignment='center',verticalalignment='top',fontsize=8,color='blue')
                elif real=="2":
                    plt.text(xpos,ypos,day_text,horizontalalignment='center',verticalalignment='top',fontsize=8,color='green')
                elif real=="3":
                    plt.text(xpos,ypos,day_text,horizontalalignment='center',verticalalignment='top',fontsize=8,color='red')
                else:
                    plt.text(xpos,ypos,day_text,horizontalalignment='center',verticalalignment='top',fontsize=8,color='black')
                # añade un título al gráfico
            plt.title("Sensor: "+str(col)+" mes: "+str(month))
            # añade una leyenda al gráfico que indique dos X en negro, junto a la de arriba el texto "valor predicho" y junto a la de abajo, "valor real"
            #legend_elements = [matplotlib.lines.Line2D([0], [0], marker='x', linestyle='None', label='Valor real', color='k', markersize=8),matplotlib.lines.Line2D([0], [0], marker='x', linestyle='None', label='Valor predicho', color='k', markersize=8)]

            # crea una carpeta para guardar los gráficos si no existe
            if not os.path.isdir(save_folder+years_test[i]+"/"+str(tdv_tests[i].columns[j])):
                os.makedirs(save_folder+years_test[i]+"/"+str(tdv_tests[i].columns[j]))

            # guarda el gráfico
            plt.savefig(save_folder+years_test[i]+"/"+str(tdv_tests[i].columns[j])+"/"+f'{month:02}'+".png")

            # cierra el gráfico
            plt.close()