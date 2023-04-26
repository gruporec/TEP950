import sys
from time import time
from matplotlib.markers import MarkerStyle
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import time
import isadoralib as isl

# ejecuta pyplot en modo agg
plt.switch_backend('agg')

years = ["2014","2015","2016","2019"]
#nDiass = [4,7,14,21]
nDiass = [4,7]
for nDias in nDiass:
    folder = "ignore/resultadosTDV/comp_max/"+str(nDias)
    for year in years:
        # Ejecuta cargaRaw.py si no existe rawDiarios.csv o rawMinutales.csv
        if not os.path.isfile("rawDiarios"+year+".csv") or not os.path.isfile("rawMinutales"+year+".csv"):
            os.system("python3 cargaRaw.py")

        # Crea tres carpetas si no existen
        if not os.path.isdir(folder+"/"+year):
            os.makedirs(folder+"/"+year)
        if not os.path.isdir(folder+"/"+year+"/est1"):
            os.makedirs(folder+"/"+year+"/est1")
        if not os.path.isdir(folder+"/"+year+"/est2"):
            os.makedirs(folder+"/"+year+"/est2")
        if not os.path.isdir(folder+"/"+year+"/est3"):
            os.makedirs(folder+"/"+year+"/est3")

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
        #obtiene los máximos de tdv_prev
        tdv_prev = tdv_prev.max(level=0)
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

        #calcula la media de cada fila de tdv_prev
        tdv_prev_mean = tdv_prev.mean(axis=1)
        #calcula la desviación estándar de cada fila de tdv_prev
        tdv_prev_std = tdv_prev.std(axis=1)

        print('6')
        #resta a cada fila su media
        tdv_prev = tdv_prev.sub(tdv_prev_mean,axis=0)
        #divide cada fila entre su desviación estándar
        tdv_prev = tdv_prev.div(tdv_prev_std,axis=0)

        #unstackea el segundo nivel del índice de tdv_prev
        tdv_prev = tdv_prev.unstack(1)
        #stackea el primer nivel del índice de tdv_prev
        tdv_prev = tdv_prev.stack(0)

        #extrae la primera columna de tdv_prev que contiene la palabra "Control"
        tdv_prev_ref = tdv_prev[tdv_prev.columns[tdv_prev.columns.str.contains('Control')]]
        #limita ref a 1 columna
        tdv_prev_ref = tdv_prev_ref.iloc[:,0]

        #unstackea el segundo nivel del índice de tdv_prev
        tdv_prev = tdv_prev.unstack(1)
        #stackea el primer nivel del índice de tdv_prev
        tdv_prev = tdv_prev.stack(0)

        #unstackea el segundo nivel del índice de ref
        tdv_prev_ref = tdv_prev_ref.unstack(1)

        print('7')


        #extrae los elementos únicos del segundo índice de tdv_prev
        tdv_prev_index = tdv_prev.index.get_level_values(1).unique()

        #por cada elemento del índice de tdv_prev
        for i in tdv_prev_index:
            #extrae las filas de tdv_prev que tengan índice i
            tdv_prev_i = tdv_prev[tdv_prev.index.get_level_values(1)==i]
            #elimina el segundo nivel del índice de tdv_prev_i
            tdv_prev_i.index = tdv_prev_i.index.droplevel(1)
            
            # por cada elemento del índice de tdv_prev_i
            for j in tdv_prev_i.index:
                #extrae la fila de tdv_prev_i que tenga índice j
                tdv_prev_ij = tdv_prev_i.loc[j]
                #extrae la fila de tdv_prev_ref que tenga índice j
                tdv_prev_ref_j = tdv_prev_ref.loc[j]

                #crea una figura con 3 subplots con tamaño adecuado para un A4 vertical
                fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(8.27,11.69))
                
                #pone el título de la figura
                fig.suptitle('Fecha: '+str(j)+'\nÁrbol a clasificar: '+str(i)+'\nNivel de estrés: '+str(int(data.loc[j,i])))
                #grafica tdv_prev_ij en el primer subplot
                ax1.plot(tdv_prev_ij)
                #pone el título del primer subplot
                ax1.set_title('Árbol a clasificar')

                #grafica tdv_prev_ref_j en el segundo subplot
                ax2.plot(tdv_prev_ref_j)
                #pone el título del segundo subplot
                ax2.set_title('Árbol de referencia')

                #calcula la diferencia entre tdv_prev_ij y tdv_prev_ref_j
                tdv_dif=tdv_prev_ij-tdv_prev_ref_j
                #elimina valores nulos
                tdv_dif=tdv_dif.dropna()
                #resta a tdv_dif su valor inicial
                tdv_dif=tdv_dif-tdv_dif.iloc[0]
                #grafica la diferencia entre tdv_prev_ij y tdv_prev_ref_j en el tercer subplot
                ax3.plot(tdv_dif)
                #pone el título del tercer subplot
                ax3.set_title('Diferencia')

                #ajusta el espacio al principio para que no se corte el título
                plt.subplots_adjust(top=0.85)
                #ajusta el espacio entre subplots
                plt.subplots_adjust(hspace=0.5)
                
                #guarda la figura en la carpeta creada anteriormente
                plt.savefig(folder+'/'+year+'/est'+str(int(data.loc[j,i]))+'/'+str(i)+str(j)+'.png')
                #cierra la figura
                plt.close()