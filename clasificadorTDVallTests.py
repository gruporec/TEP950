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
import multiprocessing as mp

# Define una función que por cada valor entre 1 y days, devuelve un dataframe con los valores de days días antes
def get_prev_days(df, days):
    #crea un dataframe vacío
    temp_df = df.copy()
    for i in range(1,days,1):
        #copia df en un dataframe temporal
        df_temp = df.copy()
        #añade i a la columna Carac
        df_temp['Carac'] = df_temp['Carac'] + str(i)
        #añade i días al índice
        df_temp.index = df_temp.index + pd.Timedelta(days=i)
        #añade el dataframe temporal al dataframe de dias anteriores
        temp_df = temp_df.append(df_temp)
    return temp_df

# Define una función que normaliza los valores de un dataframe
def normalize_df(df):
    #elimina las filas con valores NaN
    df = df.dropna(how='any')
    #calcula la media y la desviación típica de cada fila
    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1)
    #normaliza cada fila
    df = df.sub(df_mean, axis=0)
    df = df.div(df_std, axis=0)
    return df

#define una función para procesar cada elemento de un batch
def process_el(args):
    (PCA_components,TDV_days,years_test,year_train)=args
    avg_accuracy = 0
    #meteo_days=search_meteo_days+np.random.randint(-temperature*days_temp_scale,temperature*days_temp_scale)
    #TDV_sampling=search_TDV_sampling+np.random.randint(-temperature*sampling_temp_scale,temperature*sampling_temp_scale)
    #meteo_sampling=search_meteo_sampling+np.random.randint(-temperature*sampling_temp_scale,temperature*sampling_temp_scale)

    #si los parámetros ya se han probado, se salta esta iteración
    #if [PCA_components,TDV_days,meteo_days,TDV_sampling,meteo_sampling] in tested_params:
    # if [PCA_components,TDV_days,meteo_days,meteo_sampling] in tested_params:
    # if not ([PCA_components,TDV_days] in tested_params):
    #     #tested_params.append([PCA_components,TDV_days,meteo_days,TDV_sampling,meteo_sampling])
    #     # tested_params.append([PCA_components,TDV_days,meteo_days,meteo_sampling])
    #     tested_params.append([PCA_components,TDV_days])

    #clamp values
    if PCA_components < 1:
        PCA_components = 1
    if TDV_days < 1:
        TDV_days = 1
    # if meteo_days < 1:
    #     meteo_days = 1
    # if TDV_sampling < 1:
    #     TDV_sampling = 1
    # if meteo_sampling < 1:
    #     meteo_sampling = 1

    # aproxima los valores de sampling a divisores enteros de 1440
    
    # if TDV_sampling > 1440:
    #     TDV_sampling = 1440
    # if meteo_sampling > 1440:
    #     meteo_sampling = 1440
    # if TDV_sampling < 1440:
    #     TDV_sampling = 1440 // (1440 // TDV_sampling)
    # if meteo_sampling < 1440:
    #     meteo_sampling = 1440 // (1440 // meteo_sampling)

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
            # meteo_train = meteo_train.dropna()
            # meteo_test = meteo_test.dropna()

            #convierte el índice de tdv y meteo a fecha
            tdv_train.index = pd.to_datetime(tdv_train.index)
            tdv_test.index = pd.to_datetime(tdv_test.index)
            # meteo_train.index = pd.to_datetime(meteo_train.index)
            # meteo_test.index = pd.to_datetime(meteo_test.index)


            #remuestrea tdv y meteo al sampling en minutos correspondiente
            # tdv_train = tdv_train.resample(str(TDV_sampling)+'T').mean()
            # tdv_test = tdv_test.resample(str(TDV_sampling)+'T').mean()
            # meteo_train = meteo_train.resample(str(meteo_sampling)+'T').mean()
            # meteo_test = meteo_test.resample(str(meteo_sampling)+'T').mean()

            #obtiene el máximo diario de tdv
            tdv_train_max = tdv_train.groupby(tdv_train.index.date).max()
            tdv_test_max = tdv_test.groupby(tdv_test.index.date).max()

            #obtiene el minimo diario de tdv
            tdv_train_min = tdv_train.groupby(tdv_train.index.date).min()
            tdv_test_min = tdv_test.groupby(tdv_test.index.date).min()

            #obtiene la diferencia entre el máximo y el mínimo del dia anterior
            tdv_train_min_shift = tdv_train_min.shift(1)
            tdv_test_min_shift = tdv_test_min.shift(1)
            tdv_train_diff = tdv_train_max - tdv_train_min_shift
            tdv_test_diff = tdv_test_max - tdv_test_min_shift

            #obtiene el incremento del máximo respecto al día anterior
            tdv_train_max_shift = tdv_train_max.shift(1)
            tdv_test_max_shift = tdv_test_max.shift(1)
            tdv_train_inc = tdv_train_max - tdv_train_max_shift
            tdv_test_inc = tdv_test_max - tdv_test_max_shift

            #convierte el índice de data a fecha
            data_train.index = pd.to_datetime(data_train.index)
            data_train.index = data_train.index.date
            data_test.index = pd.to_datetime(data_test.index)
            data_test.index = data_test.index.date

            #crea un dataframe con data desfasado en 1 día
            data_train_shift = data_train.shift(1)
            data_test_shift = data_test.shift(1)

            # extrae de max la primera columna que contenga "Control"
            max_control_train = tdv_train_max.filter(regex='Control').iloc[:,0]
            max_control_test = tdv_test_max.filter(regex='Control').iloc[:,0]
            # convierte max_control a dataframe con las mismas columnas que tdv_train_max duplicando el valor de max_control en todas las filas
            max_control_train = pd.DataFrame(np.transpose(np.tile(max_control_train.values, (len(tdv_train_max.columns), 1))), index=tdv_train_max.index, columns=tdv_train_max.columns)
            max_control_test = pd.DataFrame(np.transpose(np.tile(max_control_test.values, (len(tdv_test_max.columns), 1))), index=tdv_test_max.index, columns=tdv_test_max.columns)
            # calcula la diferencia entre max y max_control
            max_control_train = tdv_train_max - max_control_train
            max_control_test = tdv_test_max - max_control_test
            
            # # extrae de min la primera columna que contenga "Control"
            # min_control_train = tdv_train_min.filter(regex='Control').iloc[:,0]
            # min_control_test = tdv_test_min.filter(regex='Control').iloc[:,0]
            # # convierte min_control a dataframe con las mismas columnas que tdv_train_min duplicando el valor de min_control en todas las filas
            # min_control_train = pd.DataFrame(np.transpose(np.tile(min_control_train.values, (len(tdv_train_min.columns), 1))), index=tdv_train_min.index, columns=tdv_train_min.columns)
            # min_control_test = pd.DataFrame(np.transpose(np.tile(min_control_test.values, (len(tdv_test_min.columns), 1))), index=tdv_test_min.index, columns=tdv_test_min.columns)

            # # extrae de diff la primera columna que contenga "Control"
            # diff_control_train = tdv_train_diff.filter(regex='Control').iloc[:,0]
            # diff_control_test = tdv_test_diff.filter(regex='Control').iloc[:,0]
            # # convierte diff_control a dataframe con las mismas columnas que tdv_train_diff duplicando el valor de diff_control en todas las filas
            # diff_control_train = pd.DataFrame(np.transpose(np.tile(diff_control_train.values, (len(tdv_train_diff.columns), 1))), index=tdv_train_diff.index, columns=tdv_train_diff.columns)
            # diff_control_test = pd.DataFrame(np.transpose(np.tile(diff_control_test.values, (len(tdv_test_diff.columns), 1))), index=tdv_test_diff.index, columns=tdv_test_diff.columns)

            # # extrae de inc la primera columna que contenga "Control"
            # inc_control_train = tdv_train_inc.filter(regex='Control').iloc[:,0]
            # inc_control_test = tdv_test_inc.filter(regex='Control').iloc[:,0]
            # # convierte inc_control a dataframe con las mismas columnas que tdv_train_inc duplicando el valor de inc_control en todas las filas
            # inc_control_train = pd.DataFrame(np.transpose(np.tile(inc_control_train.values, (len(tdv_train_inc.columns), 1))), index=tdv_train_inc.index, columns=tdv_train_inc.columns)
            # inc_control_test = pd.DataFrame(np.transpose(np.tile(inc_control_test.values, (len(tdv_test_inc.columns), 1))), index=tdv_test_inc.index, columns=tdv_test_inc.columns)

            #añade la columna Carac de valor max
            tdv_train_max['Carac'] = 'max'
            tdv_test_max['Carac'] = 'max'
            #añade la columna Carac de valor min
            tdv_train_min['Carac'] = 'min'
            tdv_test_min['Carac'] = 'min'
            #añade la columna Carac de valor diff
            tdv_train_diff['Carac'] = 'diff'
            tdv_test_diff['Carac'] = 'diff'
            #añade la columna Carac de valor slevel
            data_train_shift['Carac'] = 'slevel'
            data_test_shift['Carac'] = 'slevel'
            #añade la columna Carac de valor inc
            tdv_train_inc['Carac'] = 'inc'
            tdv_test_inc['Carac'] = 'inc'
            #añade la columna Carac de valor diff_control
            max_control_train['Carac'] = 'diff_control'
            max_control_test['Carac'] = 'diff_control'

            #obtiene los datos de dias anteriores de max usando la función get_prev_days
            tdv_prev_train_max = get_prev_days(tdv_train_max,TDV_days)
            tdv_prev_test_max = get_prev_days(tdv_test_max,TDV_days)
            #añade la columna carac al indice en un nivel superior
            tdv_prev_train_max = tdv_prev_train_max.set_index('Carac',append=True)
            tdv_prev_test_max = tdv_prev_test_max.set_index('Carac',append=True)
            #unstack para que los valores de carac se conviertan en columnas
            tdv_prev_train_max = tdv_prev_train_max.unstack(level='Carac')
            tdv_prev_test_max = tdv_prev_test_max.unstack(level='Carac')
            #normaliza los datos
            tdv_prev_train_max = normalize_df(tdv_prev_train_max)
            tdv_prev_test_max = normalize_df(tdv_prev_test_max)

            #obtiene los datos de dias anteriores de min usando la función get_prev_days
            tdv_prev_train_min = get_prev_days(tdv_train_min,TDV_days)
            tdv_prev_test_min = get_prev_days(tdv_test_min,TDV_days)
            #añade la columna carac al indice en un nivel superior
            tdv_prev_train_min = tdv_prev_train_min.set_index('Carac',append=True)
            tdv_prev_test_min = tdv_prev_test_min.set_index('Carac',append=True)
            #unstack para que los valores de carac se conviertan en columnas
            tdv_prev_train_min = tdv_prev_train_min.unstack(level='Carac')
            tdv_prev_test_min = tdv_prev_test_min.unstack(level='Carac')
            #normaliza los datos
            tdv_prev_train_min = normalize_df(tdv_prev_train_min)
            tdv_prev_test_min = normalize_df(tdv_prev_test_min)

            #obtiene los datos de dias anteriores de diff usando la función get_prev_days
            tdv_prev_train_diff = get_prev_days(tdv_train_diff,TDV_days)
            tdv_prev_test_diff = get_prev_days(tdv_test_diff,TDV_days)
            #añade la columna carac al indice en un nivel superior
            tdv_prev_train_diff = tdv_prev_train_diff.set_index('Carac',append=True)
            tdv_prev_test_diff = tdv_prev_test_diff.set_index('Carac',append=True)
            #unstack para que los valores de carac se conviertan en columnas
            tdv_prev_train_diff = tdv_prev_train_diff.unstack(level='Carac')
            tdv_prev_test_diff = tdv_prev_test_diff.unstack(level='Carac')
            #normaliza los datos
            tdv_prev_train_diff = normalize_df(tdv_prev_train_diff)
            tdv_prev_test_diff = normalize_df(tdv_prev_test_diff)

            #obtiene los datos de dias anteriores de inc usando la función get_prev_days
            tdv_prev_train_inc = get_prev_days(tdv_train_inc,TDV_days)
            tdv_prev_test_inc = get_prev_days(tdv_test_inc,TDV_days)
            #añade la columna carac al indice en un nivel superior
            tdv_prev_train_inc = tdv_prev_train_inc.set_index('Carac',append=True)
            tdv_prev_test_inc = tdv_prev_test_inc.set_index('Carac',append=True)
            #unstack para que los valores de carac se conviertan en columnas
            tdv_prev_train_inc = tdv_prev_train_inc.unstack(level='Carac')
            tdv_prev_test_inc = tdv_prev_test_inc.unstack(level='Carac')
            #normaliza los datos
            tdv_prev_train_inc = normalize_df(tdv_prev_train_inc)
            tdv_prev_test_inc = normalize_df(tdv_prev_test_inc)

            # crea tdv_prev con los datos normalizados
            tdv_prev_train=pd.concat([tdv_prev_train_max,tdv_prev_train_min,tdv_prev_train_diff,tdv_prev_train_inc],axis=1)
            tdv_prev_test=pd.concat([tdv_prev_test_max,tdv_prev_test_min,tdv_prev_test_diff,tdv_prev_test_inc],axis=1)

            #stackea el primer nivel de columnas
            tdv_prev_train = tdv_prev_train.stack(0)
            tdv_prev_test = tdv_prev_test.stack(0)

            #stackea las columnas de data
            data_train = data_train.stack()
            data_test = data_test.stack()

            #print('3')

            #elimina todos los valores nan de tdv_prev y meteo_prev
            tdv_prev_train = tdv_prev_train.dropna(how='any')
            tdv_prev_test = tdv_prev_test.dropna(how='any')
            # meteo_prev_train = meteo_prev_train.dropna(how='any')
            # meteo_prev_test = meteo_prev_test.dropna(how='any')

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


            #añade a tdv un nivel de columnas y lo rellena con "TDV"
            tdv_prev_train.columns = pd.MultiIndex.from_product([tdv_prev_train.columns, ['TDV']])
            tdv_prev_test.columns = pd.MultiIndex.from_product([tdv_prev_test.columns, ['TDV']])

            #unstackea el segundo nivel del índice de tdv
            tdv_prev_train = tdv_prev_train.unstack(1)
            tdv_prev_test = tdv_prev_test.unstack(1)

            #vuelve a stackear las columnas de tdv
            tdv_prev_train = tdv_prev_train.stack()
            tdv_prev_test = tdv_prev_test.stack()

            # crea un nuevo dataframe copiando tdv_prev
            tdv_meteo_train = tdv_prev_train.copy()
            tdv_meteo_test = tdv_prev_test.copy()

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
            #si el número de componentes es mayor que el número de columnas, reduce el número de componentes
            if PCA_components > tdv_meteo_train.shape[1]:
                PCA_components = tdv_meteo_train.shape[1]
            pca = skdecomp.PCA(n_components=PCA_components)

            #entrena el modelo con los datos de tdv_prev
            pca.fit(tdv_meteo_train)

            #transforma los datos de tdv_prev
            # tdv_meteo_train_f = tdv_meteo_train
            # tdv_meteo_test_f = tdv_meteo_test

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

            # #añade los resultados al archivo csv
            # file.write(str(year_test) + ',' + str(PCA_components) + ',' + str(TDV_days) + ',' + str(meteo_days) + ',' + str(meteo_sampling) + ',' + str(accuracy_train) + ',' + str(accuracy_test) + '\n')
            # file.flush()
            # file.write(str(year_test) + ',' + str(PCA_components) + ',' + str(TDV_days) + ',' + str(accuracy_train) + ',' + str(accuracy_test) + '\n')
            # file.flush()
            
        avg_accuracy=avg_accuracy/len(years_test)
        return (avg_accuracy,PCA_components,TDV_days,years_test,year_train)
            
        #si hay una excepción por teclado, sale de los bucles
    except KeyboardInterrupt:
        file.close()
        traceback.print_exc()
        # print('PCA_components: ' + str(PCA_components), 'TDV_days: ' + str(TDV_days), 'meteo_days: ' + str(meteo_days), 'meteo_sampling: ' + str(meteo_sampling))
        print('PCA_components: ' + str(PCA_components),'TDV_days: ' + str(TDV_days))
        sys.exit()
        return 0
    #si hay cualquier otra excepción, la imprime en consola y sigue
    except:
        traceback.print_exc()
        # print('PCA_components: ' + str(PCA_components), 'TDV_days: ' + str(TDV_days), 'meteo_days: ' + str(meteo_days), 'meteo_sampling: ' + str(meteo_sampling))
        print('PCA_components: ' + str(PCA_components), 'TDV_days: ' + str(TDV_days))
        return 0

if __name__=='__main__':
    year_train = "2014"
    years_test = ["2015","2016","2019"]

    res_file_target='ignore/resultadosTDV/batch/test'
    # comprueba si el fichero de resultados existe. Si existe, modifica el nombre para no sobreescribirlo utilizando un contador hasta encontrar un nombre que no exista
    i=0
    res_file=res_file_target+'.csv'
    while os.path.isfile(res_file):
        i+=1
        res_file=res_file_target+'('+str(i)+').csv'


    temperature = 200
    pca_temp_scale = 1
    days_temp_scale = 0.21
    sampling_temp_scale = 10

    temp_step_scale = 2.5
    batch_size = 10 # 20 es demasiado para 400 epochs (2% en más de 12 horas). 200 de temperatura puede estar bien; revisar. No parece que los resultados mejoren mucho.
    epochs = 100

    # valores iniciales de los parámetros
    best_PCA_components = 40
    best_TDV_days = 15
    #best_meteo_days = 8
    #best_TDV_sampling = 1440
    #best_meteo_sampling = 1440

    # inicializa la mejor precisión a 0
    best_accuracy = 0

    # lista de parámetros que ya se han probado
    tested_params = []

    #abre el fichero de resultados para añadir los nuevos resultados
    if os.path.isfile(res_file):
        file=open(res_file,"a")
    else:
        file=open(res_file,"w")
        #file.write("year_test,PCA_components,TDV_days,meteo_days,TDV_sampling,meteo_sampling,train_score,test_score\n")
        # file.write("year_test,PCA_components,TDV_days,meteo_days,meteo_sampling,train_score,test_score\n")
        file.write("PCA_components,TDV_days,test_score\n")

    # Ejecuta cargaRaw.py si no existe rawDiarios.csv o rawMinutales.csv
    if not os.path.isfile("rawDiarios"+year_train+".csv") or not os.path.isfile("rawMinutales"+year_train+".csv"):
        os.system("python3 cargaRaw.py")



    for epoch in range(epochs):
        search_PCA_components = best_PCA_components
        search_TDV_days = best_TDV_days
        #search_meteo_days = best_meteo_days
        #search_TDV_sampling = best_TDV_sampling
        #search_meteo_sampling = best_meteo_sampling

        #lanza un proceso para cada elemento del batch

        #crea una lista con los inputs de cada proceso
        inputs = []
        for batch_el in range(batch_size):
            PCA_components=search_PCA_components+np.random.randint(-temperature*pca_temp_scale,temperature*pca_temp_scale)
            TDV_days=search_TDV_days+np.random.randint(-temperature*days_temp_scale,temperature*days_temp_scale)
            inputs.append((PCA_components,TDV_days,years_test,year_train))
        
        #crea una pool de procesos
        pool = mp.Pool(processes=batch_size)

        #lanza los procesos
        results = pool.map(process_el,inputs)

        #cierra la pool
        pool.close()

        #comprueba si alguno de los resultados es mejor que el actual y obtiene los parámetros de entrada asociados
        for i in range(len(results)):
            if results[i][0]==None:
                results[i][0]=0
            if results[i][0]>best_accuracy:
                #(avg_accuracy,PCA_components,TDV_days,years_test,year_train)
                best_accuracy=results[i][0]
                best_PCA_components=results[i][1]
                best_TDV_days=results[i][2]
                #best_meteo_days=inputs[i][2]
                #best_TDV_sampling=inputs[i][3]
                #best_meteo_sampling=inputs[i][4]
                print('best accuracy: ',best_accuracy)
        temperature=temperature-(temperature*temp_step_scale/epochs)
        print(epoch/epochs*100,'% done, temperature: ',temperature)
        #guarda los parámetros de entrada y el resultado en el fichero de resultados
        for i in range(len(results)):
            file.write(str(results[i][1]) + ',' + str(results[i][2]) + ',' + str(results[i][0]) + '\n')
            file.flush()
    file.close()