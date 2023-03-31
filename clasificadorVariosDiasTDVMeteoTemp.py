import sys
from time import time
import traceback
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


year_train = "2014"
years_test = ["2015","2016","2019"]

res_file='ignore/resultadosTDV/PCA_LDA_results_temp.csv'


temperature = 100
pca_temp_scale = 1
days_temp_scale = 0.21
sampling_temp_scale = 10

temp_step_scale = 3
batch_size = 5
epochs = 100

# starting values for search
best_PCA_components = 40
best_TDV_days = 15
best_meteo_days = 8
best_TDV_sampling = 1440
best_meteo_sampling = 1440

# initialize best accuracy
best_accuracy = 0

#abre el fichero de resultados para añadir los nuevos resultados
if os.path.isfile(res_file):
    file=open(res_file,"a")
else:
    file=open(res_file,"w")
    file.write("year_test,PCA_components,TDV_days,meteo_days,TDV_sampling,meteo_sampling,train_score,test_score\n")

# Ejecuta cargaRaw.py si no existe rawDiarios.csv o rawMinutales.csv
if not os.path.isfile("rawDiarios"+year_train+".csv") or not os.path.isfile("rawMinutales"+year_train+".csv"):
    os.system("python3 cargaRaw.py")

for epoch in range(epochs):
    search_PCA_components = best_PCA_components
    search_TDV_days = best_TDV_days
    search_meteo_days = best_meteo_days
    search_TDV_sampling = best_TDV_sampling
    search_meteo_sampling = best_meteo_sampling

    for batch_el in range(batch_size):
        avg_accuracy = 0
        PCA_components=search_PCA_components+np.random.randint(-temperature*pca_temp_scale,temperature*pca_temp_scale)
        TDV_days=search_TDV_days+np.random.randint(-temperature*days_temp_scale,temperature*days_temp_scale)
        meteo_days=search_meteo_days+np.random.randint(-temperature*days_temp_scale,temperature*days_temp_scale)
        TDV_sampling=search_TDV_sampling+np.random.randint(-temperature*sampling_temp_scale,temperature*sampling_temp_scale)
        meteo_sampling=search_meteo_sampling+np.random.randint(-temperature*sampling_temp_scale,temperature*sampling_temp_scale)
        #clamp values
        if PCA_components < 1:
            PCA_components = 1
        if TDV_days < 1:
            TDV_days = 1
        if meteo_days < 1:
            meteo_days = 1
        if TDV_sampling < 1:
            TDV_sampling = 1
        if meteo_sampling < 1:
            meteo_sampling = 1

        # aproxima los valores de sampling a divisores enteros de 1440
        
        if TDV_sampling > 1440:
            TDV_sampling = 1440
        if meteo_sampling > 1440:
            meteo_sampling = 1440
        if TDV_sampling < 1440:
            TDV_sampling = 1440 // (1440 // TDV_sampling)
        if meteo_sampling < 1440:
            meteo_sampling = 1440 // (1440 // meteo_sampling)

        try:
            for year_test in years_test:
                if not os.path.isfile("rawDiarios"+year_test+".csv") or not os.path.isfile("rawMinutales"+year_test+".csv"):
                    os.system("python3 cargaRaw.py")
     
                # Carga de datos
                tdv_train,ltp_train,meteo_train,data_train=isl.cargaDatosTDV(year_train,"rht")
                tdv_test,ltp_test,meteo_test,data_test=isl.cargaDatosTDV(year_test,"rht")

                #elimina los datos nan de tdv y meteo
                tdv_train = tdv_train.dropna()
                tdv_test = tdv_test.dropna()
                meteo_train = meteo_train.dropna()
                meteo_test = meteo_test.dropna()

                #convierte el índice de tdv y meteo a fecha
                tdv_train.index = pd.to_datetime(tdv_train.index)
                tdv_test.index = pd.to_datetime(tdv_test.index)
                meteo_train.index = pd.to_datetime(meteo_train.index)
                meteo_test.index = pd.to_datetime(meteo_test.index)


                #remuestrea tdv y meteo al sampling en minutos correspondiente
                tdv_train = tdv_train.resample(str(TDV_sampling)+'T').mean()
                tdv_test = tdv_test.resample(str(TDV_sampling)+'T').mean()
                meteo_train = meteo_train.resample(str(meteo_sampling)+'T').mean()
                meteo_test = meteo_test.resample(str(meteo_sampling)+'T').mean()

                #crea columnas de fecha y hora en tdv y meteo
                tdv_train['Fecha'] = tdv_train.index.date
                tdv_train['Hora'] = tdv_train.index.time
                meteo_train['Fecha'] = meteo_train.index.date
                meteo_train['Hora'] = meteo_train.index.time
                tdv_test['Fecha'] = tdv_test.index.date
                tdv_test['Hora'] = tdv_test.index.time
                meteo_test['Fecha'] = meteo_test.index.date
                meteo_test['Hora'] = meteo_test.index.time

                #convierte el índice de data a fecha
                data_train.index = pd.to_datetime(data_train.index)
                data_train.index = data_train.index.date
                data_test.index = pd.to_datetime(data_test.index)
                data_test.index = data_test.index.date

                #cambia el índice de tdv y meteo por la columna de fecha
                tdv_train = tdv_train.set_index('Fecha')
                tdv_test = tdv_test.set_index('Fecha')
                meteo_train = meteo_train.set_index('Fecha')
                meteo_test = meteo_test.set_index('Fecha')

                #convierte la columna de hora a un valor numérico de minutos teniendo en cuenta que es un objeto datetime.time
                tdv_train['Hora'] = tdv_train['Hora'].apply(lambda x: x.hour*60 + x.minute)
                tdv_test['Hora'] = tdv_test['Hora'].apply(lambda x: x.hour*60 + x.minute)
                meteo_train['Hora'] = meteo_train['Hora'].apply(lambda x: x.hour*60 + x.minute)
                meteo_test['Hora'] = meteo_test['Hora'].apply(lambda x: x.hour*60 + x.minute)

                #copia tdv y meteo en un dataframe nuevo al que añadir los datos de dias anteriores
                tdv_prev_train = tdv_train.copy()
                meteo_prev_train = meteo_train.copy()
                tdv_prev_test = tdv_test.copy()
                meteo_prev_test = meteo_test.copy()       
                #por cada valor entre 1 y days, crea un dataframe temporal con los datos de tdv, restando 24*i horas al valor de la columna hora y añadiendo un dia a cada valor del índice
                for i in range(1,TDV_days,1):
                    #copia tdv en un dataframe temporal
                    tdv_temp_train = tdv_train.copy()
                    tdv_temp_test = tdv_test.copy()
                    #resta 24*i horas al valor de la columna hora
                    tdv_temp_train['Hora'] = tdv_temp_train['Hora'] - 24*60*i
                    tdv_temp_test['Hora'] = tdv_temp_test['Hora'] - 24*60*i
                    #añade i días al índice
                    tdv_temp_train.index = tdv_temp_train.index + pd.Timedelta(days=i)
                    tdv_temp_test.index = tdv_temp_test.index + pd.Timedelta(days=i)
                    #añade el dataframe temporal al dataframe de dias anteriores
                    tdv_prev_train = tdv_prev_train.append(tdv_temp_train)
                    tdv_prev_test = tdv_prev_test.append(tdv_temp_test)
                #repite el proceso para meteo
                for i in range(1,meteo_days,1):
                    meteo_temp_train = meteo_train.copy()
                    meteo_temp_test = meteo_test.copy()
                    meteo_temp_train['Hora'] = meteo_temp_train['Hora'] - 24*60*i
                    meteo_temp_test['Hora'] = meteo_temp_test['Hora'] - 24*60*i
                    meteo_temp_train.index = meteo_temp_train.index + pd.Timedelta(days=i)
                    meteo_temp_test.index = meteo_temp_test.index + pd.Timedelta(days=i)
                    meteo_prev_train = meteo_prev_train.append(meteo_temp_train)
                    meteo_prev_test = meteo_prev_test.append(meteo_temp_test)

                #añade la columna hora al índice en un segundo nivel
                tdv_prev_train = tdv_prev_train.set_index('Hora',append=True)
                tdv_prev_test = tdv_prev_test.set_index('Hora',append=True)
                meteo_prev_train = meteo_prev_train.set_index('Hora',append=True)
                meteo_prev_test = meteo_prev_test.set_index('Hora',append=True)

                #print('1')
                #stackea las columnas de prev
                tdv_prev_train = tdv_prev_train.stack()
                tdv_prev_test = tdv_prev_test.stack()
                meteo_prev_train = meteo_prev_train.stack()
                meteo_prev_test = meteo_prev_test.stack()

                #unstackea la columna de hora
                tdv_prev_train = tdv_prev_train.unstack('Hora')
                tdv_prev_test = tdv_prev_test.unstack('Hora')
                meteo_prev_train = meteo_prev_train.unstack('Hora')
                meteo_prev_test = meteo_prev_test.unstack('Hora')

                #print('2')

                #unstackea el segundo nivel del índice de tdv y meteo
                tdv_prev_train = tdv_prev_train.unstack(1)
                tdv_prev_test = tdv_prev_test.unstack(1)
                meteo_prev_train = meteo_prev_train.unstack(1)
                meteo_prev_test = meteo_prev_test.unstack(1)

                #elimina los valores de tdv que no estén en meteo
                tdv_prev_train = tdv_prev_train[tdv_prev_train.index.isin(meteo_prev_train.index)]
                tdv_prev_test = tdv_prev_test[tdv_prev_test.index.isin(meteo_prev_test.index)]

                #elimina los valores de meteo que no estén en tdv
                meteo_prev_train = meteo_prev_train[meteo_prev_train.index.isin(tdv_prev_train.index)]
                meteo_prev_test = meteo_prev_test[meteo_prev_test.index.isin(tdv_prev_test.index)]

                #vuelve a stackear las columnas de tdv y meteo
                tdv_prev_train = tdv_prev_train.stack()
                tdv_prev_test = tdv_prev_test.stack()
                meteo_prev_train = meteo_prev_train.stack()
                meteo_prev_test = meteo_prev_test.stack()

                #stackea las columnas de data
                data_train = data_train.stack()
                data_test = data_test.stack()

                #print('3')

                #elimina todos los valores nan de tdv_prev y meteo_prev
                tdv_prev_train = tdv_prev_train.dropna(how='any')
                tdv_prev_test = tdv_prev_test.dropna(how='any')
                meteo_prev_train = meteo_prev_train.dropna(how='any')
                meteo_prev_test = meteo_prev_test.dropna(how='any')

                #elimina los valores de data que no estén en tdv_prev o meteo_prev
                data_train = data_train[data_train.index.isin(tdv_prev_train.index)]
                #data_train = data_train[data_train.index.isin(meteo_prev_train.index)]
                data_test = data_test[data_test.index.isin(tdv_prev_test.index)]
                #data_test = data_test[data_test.index.isin(meteo_prev_test.index)]

                #print('4')
                #elimina los valores de tdv_prev que no estén en data o meteo_prev
                tdv_prev_train = tdv_prev_train[tdv_prev_train.index.isin(data_train.index)]
                #tdv_prev_train = tdv_prev_train[tdv_prev_train.index.isin(meteo_prev_train.index)]
                tdv_prev_test = tdv_prev_test[tdv_prev_test.index.isin(data_test.index)]
                #tdv_prev_test = tdv_prev_test[tdv_prev_test.index.isin(meteo_prev_test.index)]

                #elimina los valores de meteo_prev que no estén en data o tdv_prev
                # meteo_prev_train = meteo_prev_train[meteo_prev_train.index.isin(data_train.index)]
                # meteo_prev_train = meteo_prev_train[meteo_prev_train.index.isin(tdv_prev_train.index)]
                # meteo_prev_test = meteo_prev_test[meteo_prev_test.index.isin(data_test.index)]
                # meteo_prev_test = meteo_prev_test[meteo_prev_test.index.isin(tdv_prev_test.index)]

                #añade a tdv un nivel de columnas y lo rellena con "TDV"
                tdv_prev_train.columns = pd.MultiIndex.from_product([tdv_prev_train.columns, ['TDV']])
                tdv_prev_test.columns = pd.MultiIndex.from_product([tdv_prev_test.columns, ['TDV']])

                #unstackea el segundo nivel del índice de meteo
                meteo_prev_train = meteo_prev_train.unstack(1)
                meteo_prev_test = meteo_prev_test.unstack(1)

                #unstackea el segundo nivel del índice de tdv
                tdv_prev_train = tdv_prev_train.unstack(1)
                tdv_prev_test = tdv_prev_test.unstack(1)

                #elimina los valores de tdv que no estén en meteo
                tdv_prev_train = tdv_prev_train[tdv_prev_train.index.isin(meteo_prev_train.index)]
                tdv_prev_test = tdv_prev_test[tdv_prev_test.index.isin(meteo_prev_test.index)]

                #elimina los valores de meteo que no estén en tdv
                meteo_prev_train = meteo_prev_train[meteo_prev_train.index.isin(tdv_prev_train.index)]
                meteo_prev_test = meteo_prev_test[meteo_prev_test.index.isin(tdv_prev_test.index)]

                #vuelve a stackear las columnas de tdv
                tdv_prev_train = tdv_prev_train.stack()
                tdv_prev_test = tdv_prev_test.stack()

                #print('5')

                #calcula la media de cada fila de tdv_prev
                tdv_prev_train_mean = tdv_prev_train.mean(axis=1)
                tdv_prev_test_mean = tdv_prev_test.mean(axis=1)

                #calcula la desviación estándar de cada fila de tdv_prev
                tdv_prev_train_std = tdv_prev_train.std(axis=1)
                tdv_prev_test_std = tdv_prev_test.std(axis=1)

                #print('6')

                #resta a cada fila su media
                tdv_prev_train = tdv_prev_train.sub(tdv_prev_train_mean,axis=0)
                tdv_prev_test = tdv_prev_test.sub(tdv_prev_test_mean,axis=0)

                #divide cada fila entre su desviación estándar
                tdv_prev_train = tdv_prev_train.div(tdv_prev_train_std,axis=0)
                tdv_prev_test = tdv_prev_test.div(tdv_prev_test_std,axis=0)


                # crea un nuevo dataframe copiando tdv_prev
                tdv_meteo_train = tdv_prev_train.copy()
                tdv_meteo_test = tdv_prev_test.copy()

                #crea un nuevo dataframe para almacenar los datos de meteo con la forma adecuada
                meteo_temp2_train = pd.DataFrame()
                meteo_temp2_test = pd.DataFrame()

                # por cada valor único del segundo nivel del índice de tdv_meteo, crea una copia de meteo_prev, añade un segundo índice con ese valor y lo añade a tdv_meteo
                for i in tdv_meteo_train.index.get_level_values(1).unique():
                    meteo_temp_train = meteo_prev_train.copy()
                    meteo_temp_train.index = pd.MultiIndex.from_product([meteo_temp_train.index, [i]]) 

                    #añade los datos de meteo_temp a meteo_temp2
                    meteo_temp2_train = meteo_temp2_train.append(meteo_temp_train)
                for i in tdv_meteo_test.index.get_level_values(1).unique():
                    meteo_temp_test = meteo_prev_test.copy()
                    meteo_temp_test.index = pd.MultiIndex.from_product([meteo_temp_test.index, [i]]) 

                    #añade los datos de meteo_temp a meteo_temp2
                    meteo_temp2_test = meteo_temp2_test.append(meteo_temp_test)

                #añade los datos de meteo_temp2 a tdv_meteo horizontalemente
                tdv_meteo_train = pd.concat([tdv_meteo_train, meteo_temp2_train], axis=1)
                tdv_meteo_test = pd.concat([tdv_meteo_test, meteo_temp2_test], axis=1)

                #elimina los valores nan de tdv_meteo
                tdv_meteo_train = tdv_meteo_train.dropna(how='any')
                tdv_meteo_test = tdv_meteo_test.dropna(how='any')

                #elimina los valores de tdv_meteo que no estén en data
                tdv_meteo_train = tdv_meteo_train[tdv_meteo_train.index.isin(data_train.index)]
                tdv_meteo_test = tdv_meteo_test[tdv_meteo_test.index.isin(data_test.index)]

                #elimina los valores de data que no estén en tdv_meteo
                data_train = data_train[data_train.index.isin(tdv_meteo_train.index)]
                data_test = data_test[data_test.index.isin(tdv_meteo_test.index)]
                #crea un modelo PCA de sklearn
                #si el número de componentes es mayor que el número de columnas, se salta el bucle
                if PCA_components > tdv_meteo_train.shape[1]:
                    continue
                pca = skdecomp.PCA(n_components=PCA_components)

                #entrena el modelo con los datos de tdv_prev
                pca.fit(tdv_meteo_train)

                #transforma los datos de tdv_prev
                tdv_meteo_train_f = pca.transform(tdv_meteo_train)
                tdv_meteo_test_f = pca.transform(tdv_meteo_test)

                #crea un clasificador LDA de sklearn
                lda = sklda.LinearDiscriminantAnalysis(solver='svd')

                #entrena el clasificador con los datos de tdv_meteo
                lda.fit(tdv_meteo_train_f,data_train)

                #aplica el clasificador a los datos de tdv_meteo
                data_train_pred = lda.predict(tdv_meteo_train_f)
                data_test_pred = lda.predict(tdv_meteo_test_f)

                #calcula la precisión del clasificador
                accuracy_train = skmetrics.balanced_accuracy_score(data_train,data_train_pred)
                accuracy_test = skmetrics.balanced_accuracy_score(data_test,data_test_pred)

                #calcula la matriz de confusión
                confusion_matrix_train = skmetrics.confusion_matrix(data_train,data_train_pred)
                confusion_matrix_test = skmetrics.confusion_matrix(data_test,data_test_pred)

                avg_accuracy=avg_accuracy+accuracy_test

                #añade los resultados al archivo csv
                file.write(str(year_test) + ',' + str(PCA_components) + ',' + str(TDV_days) + ',' + str(meteo_days) + ',' + str(TDV_sampling) + ',' + str(meteo_sampling) + ',' + str(accuracy_train) + ',' + str(accuracy_test) + '\n')
                file.flush()
            #si hay una excepción por teclado, sale de los bucles
        except KeyboardInterrupt:
            file.close()
            traceback.print_exc()
            print('PCA_components: ' + str(PCA_components), 'TDV_days: ' + str(TDV_days), 'meteo_days: ' + str(meteo_days), 'TDV_sampling: ' + str(TDV_sampling), 'meteo_sampling: ' + str(meteo_sampling))
            sys.exit()
        #si hay cualquier otra excepción, la imprime en consola y sigue
        except:
            traceback.print_exc()
            print('PCA_components: ' + str(PCA_components), 'TDV_days: ' + str(TDV_days), 'meteo_days: ' + str(meteo_days), 'TDV_sampling: ' + str(TDV_sampling), 'meteo_sampling: ' + str(meteo_sampling))
            continue
        avg_accuracy=avg_accuracy/len(years_test)
        if avg_accuracy>best_accuracy:
            best_accuracy=avg_accuracy
            best_PCA_components=PCA_components
            best_TDV_days=TDV_days
            best_meteo_days=meteo_days
            best_TDV_sampling=TDV_sampling
            best_meteo_sampling=meteo_sampling
            print('best accuracy: ' + str(best_accuracy))
    temperature=temperature-(temperature*temp_step_scale/epochs)
    print(epoch/epochs*100,'%% done')
file.close()