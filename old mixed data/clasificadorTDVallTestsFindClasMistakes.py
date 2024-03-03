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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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

# Define una función que por cada valor entre 1 y days, devuelve un dataframe con los valores de days días antes para los datos raw
def get_prev_days_raw(df, days):
    #crea una copia de df separando el índice en fecha y hora
    df_fh = df.copy()
    df_fh['Fecha'] = df_fh.index.date
    df_fh['Hora'] = df_fh.index.time
    #añade la columna Carac con valor 0
    df_fh['Carac'] = 0

    #convierte el dataframe para que tenga doble índice con fecha, hora y carac
    df_fh.index = pd.MultiIndex.from_arrays([df_fh['Fecha'],df_fh['Hora'],df_fh['Carac']])
    #duplica el índice de fecha y hora en columnas
    df_fh['Fecha'] = df_fh.index.get_level_values(0)
    df_fh['Hora'] = df_fh.index.get_level_values(1)
    df_fh['Carac'] = df_fh.index.get_level_values(2)

    #crea un dataframe vacío
    temp_df = df_fh.copy()

    for i in range(1,days,1):
        #copia df en un dataframe temporal
        df_temp = df_fh.copy()
        #añade i dias a la fecha
        df_temp['Fecha'] = df_temp['Fecha'] + pd.Timedelta(days=i)
        #añade i a la columna Carac
        df_temp['Carac'] = df_temp['Carac'] + i
        #resetea el índice
        df_temp = df_temp.reset_index(drop=True)
        #actualiza el índice con fecha, hora y carac
        df_temp.index = pd.MultiIndex.from_arrays([df_temp['Fecha'],df_temp['Hora'],df_temp['Carac']])
        #añade el dataframe temporal al dataframe de dias anteriores
        temp_df = temp_df.append(df_temp)
    #elimina las columnas de fecha, hora y carac
    temp_df = temp_df.drop(columns=['Fecha','Hora','Carac'])
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
    (PCA_components,TDV_days,years_test,year_train,chars,tdv_train_orig,data_train_orig,tdv_tests_orig,data_tests_orig)=args


    #desempaqueta los booleanos de chars
    max_max_sens,max_min_sens,stress,max_max_cont,max_min_cont,max_min_ratio,norm,_=chars

    #inicializa los valores de accuracy
    avg_accuracy = 0

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

    try:
        #por cada elemento en years_test
        for yt in range(len(years_test)):
            #recarga los datos de entrenamiento y test
            tdv_train = tdv_train_orig.copy()
            data_train = data_train_orig.copy()

            year_test = years_test[yt]
            tdv_test = tdv_tests_orig[yt].copy()
            data_test = data_tests_orig[yt].copy()

            # #pasa los índices a datetime
            # tdv_train.index = pd.to_datetime(tdv_train.index)
            # tdv_test.index = pd.to_datetime(tdv_test.index)
            # data_train.index = pd.to_datetime(data_train.index)
            # data_test.index = pd.to_datetime(data_test.index)

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

            if norm:
                # añade los datos de los días anteriores a cada fecha de tdv
                tdv_train = get_prev_days_raw(tdv_train,TDV_days)
                tdv_test = get_prev_days_raw(tdv_test,TDV_days)
                # unstack de la fecha
                tdv_train = tdv_train.unstack(level=0)
                tdv_test = tdv_test.unstack(level=0)
                #transpone los dataframes
                tdv_train = tdv_train.transpose()
                tdv_test = tdv_test.transpose()

                #normaliza los valores de tdv
                tdv_train = normalize_df(tdv_train)
                tdv_test = normalize_df(tdv_test)

                #stack carac
                tdv_train = tdv_train.stack(level=1)
                tdv_test = tdv_test.stack(level=1)

                #obtiene el máximo diario de tdv calculando el máximo de cada fila
                tdv_train_max = tdv_train.max(axis=1)
                tdv_test_max = tdv_test.max(axis=1)

                #unstack de tdv
                tdv_train_max = tdv_train_max.unstack(level=0)
                tdv_test_max = tdv_test_max.unstack(level=0)

                #unstack de carac
                tdv_train_max = tdv_train_max.unstack(level=1)
                tdv_test_max = tdv_test_max.unstack(level=1)
                #columnas: TDV, Carac
                #índices: fecha
                #si max_max_sens es True
                if max_max_sens or max_max_cont:
                    #obtiene el incremento del máximo respecto al día anterior
                    tdv_train_max_shift = tdv_train_max.shift(1)
                    tdv_test_max_shift = tdv_test_max.shift(1)
                    tdv_prev_train_inc = tdv_train_max.sub(tdv_train_max_shift)
                    tdv_prev_test_inc = tdv_test_max.sub(tdv_test_max_shift)

                    #añade a carac el prefijo 'inc' para indicar que es el incremento
                    tdv_prev_train_inc.columns = pd.MultiIndex.from_tuples([(a,f'inc_{b}') for a,b in tdv_prev_train_inc.columns])
                    tdv_prev_test_inc.columns = pd.MultiIndex.from_tuples([(a,f'inc_{b}') for a,b in tdv_prev_test_inc.columns])
                #si max_min_sens es True
                if max_min_sens or max_min_cont or max_min_ratio:
                    #obtiene el mínimo diario de tdv calculando el mínimo de cada fila
                    tdv_train_min = tdv_train.min(axis=1)
                    tdv_test_min = tdv_test.min(axis=1)
                    #unstack de tdv
                    tdv_train_min = tdv_train_min.unstack(level=0)
                    tdv_test_min = tdv_test_min.unstack(level=0)
                    #unstack de carac
                    tdv_train_min = tdv_train_min.unstack(level=1)
                    tdv_test_min = tdv_test_min.unstack(level=1)

                    #obtiene la diferencia entre el máximo y el mínimo del día anterior
                    tdv_train_min_shift = tdv_train_min.shift(1)
                    tdv_test_min_shift = tdv_test_min.shift(1)
                    tdv_prev_train_diff = tdv_train_max.sub(tdv_train_min_shift)
                    tdv_prev_test_diff = tdv_test_max.sub(tdv_test_min_shift)

                    #añade a carac el prefijo 'diff' para indicar que es la diferencia
                    tdv_prev_train_diff.columns = pd.MultiIndex.from_tuples([(a,f'diff_{b}') for a,b in tdv_prev_train_diff.columns])
                    tdv_prev_test_diff.columns = pd.MultiIndex.from_tuples([(a,f'diff_{b}') for a,b in tdv_prev_test_diff.columns])
                
                #convierte el índice de data a fecha
                data_train.index = pd.to_datetime(data_train.index)
                data_train.index = data_train.index.date
                data_test.index = pd.to_datetime(data_test.index)
                data_test.index = data_test.index.date

                if stress:
                    #crea un dataframe con data desfasado en 1 día
                    data_train_shift = data_train.shift(1)
                    data_test_shift = data_test.shift(1)
                    #añade la columna Carac de valor stress
                    data_train_shift['Carac'] = 'stress'
                    data_test_shift['Carac'] = 'stress'
                    #obtiene los datos de dias anteriores de stress usando la función get_prev_days
                    tdv_prev_train_stress = get_prev_days(data_train_shift,TDV_days)
                    tdv_prev_test_stress = get_prev_days(data_test_shift,TDV_days)
                    #añade la columna carac al indice en un nivel superior
                    tdv_prev_train_stress = tdv_prev_train_stress.set_index('Carac',append=True)
                    tdv_prev_test_stress = tdv_prev_test_stress.set_index('Carac',append=True)
                    #unstack para que los valores de carac se conviertan en columnas
                    tdv_prev_train_stress = tdv_prev_train_stress.unstack(level='Carac')
                    tdv_prev_test_stress = tdv_prev_test_stress.unstack(level='Carac')

                #si max_max_cont es True
                if max_max_cont:
                    #unstack carac de inc
                    tdv_prev_train_inc = tdv_prev_train_inc.stack(level=1)
                    tdv_prev_test_inc = tdv_prev_test_inc.stack(level=1)
                    #extrae de inc la primera columna que contenga "Control"
                    tdv_train_inc_control = tdv_prev_train_inc.filter(regex='Control').iloc[:,0]
                    tdv_test_inc_control = tdv_prev_test_inc.filter(regex='Control').iloc[:,0]
                    #convierte inc_control a dataframe con las mismas columnas que inc duplicando el valor de inc_control en todas las filas
                    tdv_train_inc_control = pd.DataFrame(np.transpose(np.tile(tdv_train_inc_control.values, (len(tdv_prev_train_inc.columns), 1))), index=tdv_prev_train_inc.index, columns=tdv_prev_train_inc.columns)
                    tdv_test_inc_control = pd.DataFrame(np.transpose(np.tile(tdv_test_inc_control.values, (len(tdv_prev_test_inc.columns), 1))), index=tdv_prev_test_inc.index, columns=tdv_prev_test_inc.columns)
                    #vuelve a stackear carac
                    tdv_prev_train_inc_control = tdv_train_inc_control.unstack(level=1)
                    tdv_prev_test_inc_control = tdv_test_inc_control.unstack(level=1)
                    tdv_prev_train_inc = tdv_prev_train_inc.unstack(level=1)
                    tdv_prev_test_inc = tdv_prev_test_inc.unstack(level=1)
                    #añade a carac el prefijo 'max_max_cont' para indicar que es la diferencia entre incrementos con control
                    tdv_prev_train_inc_control.columns = pd.MultiIndex.from_tuples([(a,f'max_max_cont_{b}') for a,b in tdv_prev_train_inc_control.columns])
                    tdv_prev_test_inc_control.columns = pd.MultiIndex.from_tuples([(a,f'max_max_cont_{b}') for a,b in tdv_prev_test_inc_control.columns])


                #si max_min_cont o max_min_ratio es True
                if max_min_cont or max_min_ratio:
                    #unstack carac de diff
                    tdv_prev_train_diff = tdv_prev_train_diff.stack(level=1)
                    tdv_prev_test_diff = tdv_prev_test_diff.stack(level=1)
                    #extrae de diff la primera columna que contenga "Control"
                    tdv_train_diff_control = tdv_prev_train_diff.filter(regex='Control').iloc[:,0]
                    tdv_test_diff_control = tdv_prev_test_diff.filter(regex='Control').iloc[:,0]
                    #convierte diff_control a dataframe con las mismas columnas que diff duplicando el valor de diff_control en todas las filas
                    tdv_train_diff_control = pd.DataFrame(np.transpose(np.tile(tdv_train_diff_control.values, (len(tdv_prev_train_diff.columns), 1))), index=tdv_prev_train_diff.index, columns=tdv_prev_train_diff.columns)
                    tdv_test_diff_control = pd.DataFrame(np.transpose(np.tile(tdv_test_diff_control.values, (len(tdv_prev_test_diff.columns), 1))), index=tdv_prev_test_diff.index, columns=tdv_prev_test_diff.columns)
                    #vuelve a stackear carac
                    tdv_prev_train_diff_control = tdv_train_diff_control.unstack(level=1)
                    tdv_prev_test_diff_control = tdv_test_diff_control.unstack(level=1)
                    tdv_prev_train_diff = tdv_prev_train_diff.unstack(level=1)
                    tdv_prev_test_diff = tdv_prev_test_diff.unstack(level=1)
                    
                    #si max_min_ratio es True
                    if max_min_ratio:
                        #divide diff entre diff_control
                        tdv_prev_train_ratio = tdv_prev_train_diff / tdv_prev_train_diff_control
                        tdv_prev_test_ratio = tdv_prev_test_diff / tdv_prev_test_diff_control
                        #añade a carac el prefijo 'max_min_ratio' para indicar que es la diferencia entre ratios con control
                        tdv_prev_train_ratio.columns = pd.MultiIndex.from_tuples([(a,f'max_min_ratio_{b}') for a,b in tdv_prev_train_ratio.columns])
                        tdv_prev_test_ratio.columns = pd.MultiIndex.from_tuples([(a,f'max_min_ratio_{b}') for a,b in tdv_prev_test_ratio.columns])
                    #si max_min_cont es True
                    if max_min_cont:
                        #añade a carac el prefijo 'max_min_cont' para indicar que es la diferencia entre diferencias con control
                        tdv_prev_train_diff_control.columns = pd.MultiIndex.from_tuples([(a,f'max_min_cont_{b}') for a,b in tdv_prev_train_diff_control.columns])
                        tdv_prev_test_diff_control.columns = pd.MultiIndex.from_tuples([(a,f'max_min_cont_{b}') for a,b in tdv_prev_test_diff_control.columns])

                #sacar:
                    #columnas:
                        #TDV
                        #Carac
                    #índices:
                        #fecha
            else:
                #obtiene el máximo diario de tdv
                tdv_train_max = tdv_train.groupby(tdv_train.index.date).max()
                tdv_test_max = tdv_test.groupby(tdv_test.index.date).max()
                
                #si max_max_sens es True
                if max_max_sens or max_max_cont:
                    #obtiene el incremento del máximo respecto al día anterior
                    tdv_train_max_shift = tdv_train_max.shift(1)
                    tdv_test_max_shift = tdv_test_max.shift(1)
                    tdv_train_inc = tdv_train_max - tdv_train_max_shift
                    tdv_test_inc = tdv_test_max - tdv_test_max_shift
                #si max_min_sens es True
                if max_min_sens or max_min_cont or max_min_ratio:
                    #obtiene el mínimo diario de tdv
                    tdv_train_min = tdv_train.groupby(tdv_train.index.date).min()
                    tdv_test_min = tdv_test.groupby(tdv_test.index.date).min()

                    #obtiene la diferencia entre el máximo y el mínimo del dia anterior
                    tdv_train_min_shift = tdv_train_min.shift(1)
                    tdv_test_min_shift = tdv_test_min.shift(1)
                    tdv_train_diff = tdv_train_max - tdv_train_min_shift
                    tdv_test_diff = tdv_test_max - tdv_test_min_shift


                #convierte el índice de data a fecha
                data_train.index = pd.to_datetime(data_train.index)
                data_train.index = data_train.index.date
                data_test.index = pd.to_datetime(data_test.index)
                data_test.index = data_test.index.date

                if stress:
                    #crea un dataframe con data desfasado en 1 día
                    data_train_shift = data_train.shift(1)
                    data_test_shift = data_test.shift(1)
                
                #si max_max_cont es True
                if max_max_cont:
                    #extrae de inc la primera columna que contenga "Control"
                    tdv_train_inc_control = tdv_train_inc.filter(regex='Control').iloc[:,0]
                    tdv_test_inc_control = tdv_test_inc.filter(regex='Control').iloc[:,0]
                    #convierte inc_control a dataframe con las mismas columnas que inc duplicando el valor de inc_control en todas las filas
                    tdv_train_inc_control = pd.DataFrame(np.transpose(np.tile(tdv_train_inc_control.values, (len(tdv_train_inc.columns), 1))), index=tdv_train_inc.index, columns=tdv_train_inc.columns)
                    tdv_test_inc_control = pd.DataFrame(np.transpose(np.tile(tdv_test_inc_control.values, (len(tdv_test_inc.columns), 1))), index=tdv_test_inc.index, columns=tdv_test_inc.columns)
                #si max_min_cont o max_min_ratio es True
                if max_min_cont or max_min_ratio:
                    #extrae de diff la primera columna que contenga "Control"
                    tdv_train_diff_control = tdv_train_diff.filter(regex='Control').iloc[:,0]
                    tdv_test_diff_control = tdv_test_diff.filter(regex='Control').iloc[:,0]
                    #convierte diff_control a dataframe con las mismas columnas que diff duplicando el valor de diff_control en todas las filas
                    tdv_train_diff_control = pd.DataFrame(np.transpose(np.tile(tdv_train_diff_control.values, (len(tdv_train_diff.columns), 1))), index=tdv_train_diff.index, columns=tdv_train_diff.columns)
                    tdv_test_diff_control = pd.DataFrame(np.transpose(np.tile(tdv_test_diff_control.values, (len(tdv_test_diff.columns), 1))), index=tdv_test_diff.index, columns=tdv_test_diff.columns)
                    #si max_min_ratio es True
                    if max_min_ratio:
                        #divide diff entre diff_control
                        tdv_train_ratio = tdv_train_diff / tdv_train_diff_control
                        tdv_test_ratio = tdv_test_diff / tdv_test_diff_control

                #si max_max_sens es True
                if max_max_sens:
                    #añade la columna Carac de valor max_max
                    tdv_train_inc['Carac'] = 'max_max'
                    tdv_test_inc['Carac'] = 'max_max'
                    #obtiene los datos de dias anteriores de inc usando la función get_prev_days
                    tdv_prev_train_inc = get_prev_days(tdv_train_inc,TDV_days)
                    tdv_prev_test_inc = get_prev_days(tdv_test_inc,TDV_days)
                    #añade la columna carac al indice en un nivel superior
                    tdv_prev_train_inc = tdv_prev_train_inc.set_index('Carac',append=True)
                    tdv_prev_test_inc = tdv_prev_test_inc.set_index('Carac',append=True)
                    #unstack para que los valores de carac se conviertan en columnas
                    tdv_prev_train_inc = tdv_prev_train_inc.unstack(level='Carac')
                    tdv_prev_test_inc = tdv_prev_test_inc.unstack(level='Carac')
                
                #si max_min_sens es True
                if max_min_sens:
                    #añade la columna Carac de valor max_min
                    tdv_train_diff['Carac'] = 'max_min'
                    tdv_test_diff['Carac'] = 'max_min'
                    #obtiene los datos de dias anteriores de diff usando la función get_prev_days
                    tdv_prev_train_diff = get_prev_days(tdv_train_diff,TDV_days)
                    tdv_prev_test_diff = get_prev_days(tdv_test_diff,TDV_days)
                    #añade la columna carac al indice en un nivel superior
                    tdv_prev_train_diff = tdv_prev_train_diff.set_index('Carac',append=True)
                    tdv_prev_test_diff = tdv_prev_test_diff.set_index('Carac',append=True)
                    #unstack para que los valores de carac se conviertan en columnas
                    tdv_prev_train_diff = tdv_prev_train_diff.unstack(level='Carac')
                    tdv_prev_test_diff = tdv_prev_test_diff.unstack(level='Carac')

                #si max_max_cont es True
                if max_max_cont:
                    #añade la columna Carac de valor max_max_cont
                    tdv_train_inc_control['Carac'] = 'max_max_cont'
                    tdv_test_inc_control['Carac'] = 'max_max_cont'
                    #obtiene los datos de dias anteriores de inc_control usando la función get_prev_days
                    tdv_prev_train_inc_control = get_prev_days(tdv_train_inc_control,TDV_days)
                    tdv_prev_test_inc_control = get_prev_days(tdv_test_inc_control,TDV_days)
                    #añade la columna carac al indice en un nivel superior
                    tdv_prev_train_inc_control = tdv_prev_train_inc_control.set_index('Carac',append=True)
                    tdv_prev_test_inc_control = tdv_prev_test_inc_control.set_index('Carac',append=True)
                    #unstack para que los valores de carac se conviertan en columnas
                    tdv_prev_train_inc_control = tdv_prev_train_inc_control.unstack(level='Carac')
                    tdv_prev_test_inc_control = tdv_prev_test_inc_control.unstack(level='Carac')

                #si max_min_cont es True
                if max_min_cont:
                    #añade la columna Carac de valor max_min_control
                    tdv_train_diff_control['Carac'] = 'max_min_control'
                    tdv_test_diff_control['Carac'] = 'max_min_control'
                    #obtiene los datos de dias anteriores de diff_control usando la función get_prev_days
                    tdv_prev_train_diff_control = get_prev_days(tdv_train_diff_control,TDV_days)
                    tdv_prev_test_diff_control = get_prev_days(tdv_test_diff_control,TDV_days)
                    #añade la columna carac al indice en un nivel superior
                    tdv_prev_train_diff_control = tdv_prev_train_diff_control.set_index('Carac',append=True)
                    tdv_prev_test_diff_control = tdv_prev_test_diff_control.set_index('Carac',append=True)
                    #unstack para que los valores de carac se conviertan en columnas
                    tdv_prev_train_diff_control = tdv_prev_train_diff_control.unstack(level='Carac')
                    tdv_prev_test_diff_control = tdv_prev_test_diff_control.unstack(level='Carac')

                #si max_min_ratio es True
                if max_min_ratio:
                    #añade la columna Carac de valor max_min_ratio
                    tdv_train_ratio['Carac'] = 'max_min_ratio'
                    tdv_test_ratio['Carac'] = 'max_min_ratio'
                    #obtiene los datos de dias anteriores de ratio usando la función get_prev_days
                    tdv_prev_train_ratio = get_prev_days(tdv_train_ratio,TDV_days)
                    tdv_prev_test_ratio = get_prev_days(tdv_test_ratio,TDV_days)
                    #añade la columna carac al indice en un nivel superior
                    tdv_prev_train_ratio = tdv_prev_train_ratio.set_index('Carac',append=True)
                    tdv_prev_test_ratio = tdv_prev_test_ratio.set_index('Carac',append=True)
                    #unstack para que los valores de carac se conviertan en columnas
                    tdv_prev_train_ratio = tdv_prev_train_ratio.unstack(level='Carac')
                    tdv_prev_test_ratio = tdv_prev_test_ratio.unstack(level='Carac')

                #si stress es True
                if stress:
                    #añade la columna Carac de valor stress
                    data_train_shift['Carac'] = 'stress'
                    data_test_shift['Carac'] = 'stress'
                    #obtiene los datos de dias anteriores de stress usando la función get_prev_days
                    tdv_prev_train_stress = get_prev_days(data_train_shift,TDV_days)
                    tdv_prev_test_stress = get_prev_days(data_test_shift,TDV_days)
                    #añade la columna carac al indice en un nivel superior
                    tdv_prev_train_stress = tdv_prev_train_stress.set_index('Carac',append=True)
                    tdv_prev_test_stress = tdv_prev_test_stress.set_index('Carac',append=True)
                    #unstack para que los valores de carac se conviertan en columnas
                    tdv_prev_train_stress = tdv_prev_train_stress.unstack(level='Carac')
                    tdv_prev_test_stress = tdv_prev_test_stress.unstack(level='Carac')


            # crea tdv_prev con los datos normalizados y sin normalizar, añadiendo los que correspondan según chars
            # crea los dataframes de train y test vacíos
            tdv_prev_train = pd.DataFrame()
            tdv_prev_test = pd.DataFrame()
            
            # añade las características que correspondan según los valores de chars
            # si max_max_sens es True añade prev_inc
            if max_max_sens:
                tdv_prev_train = pd.concat([tdv_prev_train,tdv_prev_train_inc],axis=1)
                tdv_prev_test = pd.concat([tdv_prev_test,tdv_prev_test_inc],axis=1)
            # si max_min_sens es True añade prev_diff
            if max_min_sens:
                tdv_prev_train = pd.concat([tdv_prev_train,tdv_prev_train_diff],axis=1)
                tdv_prev_test = pd.concat([tdv_prev_test,tdv_prev_test_diff],axis=1)
            # si stress es True añade prev_stress
            if stress:
                tdv_prev_train = pd.concat([tdv_prev_train,tdv_prev_train_stress],axis=1)
                tdv_prev_test = pd.concat([tdv_prev_test,tdv_prev_test_stress],axis=1)
            # si max_max_control es True añade prev_inc_control
            if max_max_cont:
                tdv_prev_train = pd.concat([tdv_prev_train,tdv_prev_train_inc_control],axis=1)
                tdv_prev_test = pd.concat([tdv_prev_test,tdv_prev_test_inc_control],axis=1)
            # si max_min_control es True añade prev_diff_control
            if max_min_cont:
                tdv_prev_train = pd.concat([tdv_prev_train,tdv_prev_train_diff_control],axis=1)
                tdv_prev_test = pd.concat([tdv_prev_test,tdv_prev_test_diff_control],axis=1)
            # si max_min_ratio es True añade prev_ratio
            if max_min_ratio:
                tdv_prev_train = pd.concat([tdv_prev_train,tdv_prev_train_ratio],axis=1)
                tdv_prev_test = pd.concat([tdv_prev_test,tdv_prev_test_ratio],axis=1)

            #stackea el primer nivel de columnas
            tdv_prev_train = tdv_prev_train.stack(0)
            tdv_prev_test = tdv_prev_test.stack(0)

            #stackea las columnas de data
            data_train = data_train.stack()
            data_test = data_test.stack()

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

            #elimina los valores de tdv_prev que no estén en data o meteo_prev
            tdv_prev_train = tdv_prev_train[tdv_prev_train.index.isin(data_train.index)]
            #tdv_prev_train = tdv_prev_train[tdv_prev_train.index.isin(meteo_prev_train.index)]
            tdv_prev_test = tdv_prev_test[tdv_prev_test.index.isin(data_test.index)]
            #tdv_prev_test = tdv_prev_test[tdv_prev_test.index.isin(meteo_prev_test.index)]

            
            # si como resultado sólo se obtiene una serie, la convierte en dataframe
            if isinstance(tdv_prev_train,pd.Series):
                tdv_prev_train = pd.DataFrame(tdv_prev_train)
                tdv_prev_test = pd.DataFrame(tdv_prev_test)

            #añade a tdv un nivel de columnas y lo rellena con "TDV"
            tdv_prev_train.columns = pd.MultiIndex.from_product([tdv_prev_train.columns, ['TDV']])
            tdv_prev_test.columns = pd.MultiIndex.from_product([tdv_prev_test.columns, ['TDV']])

            
            # si como resultado sólo se obtiene una serie, la convierte en dataframe
            if isinstance(tdv_prev_train,pd.Series):
                tdv_prev_train = pd.DataFrame(tdv_prev_train)
                tdv_prev_test = pd.DataFrame(tdv_prev_test)
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
    years_test = ["2014"]

    meta_file='ignore/resultadosTDV/batch/PCALDA_test/meta2.csv'
    # comprueba si el fichero meta existe. Si no existe, lo crea, añadiendo la cabecera
    if not os.path.isfile(meta_file):
        m_file=open(meta_file,'w')
        m_file.write('ID, best acc\n')
        m_file.close()

    # carga el archivo de tests en un dataframe
    tests = pd.read_csv('ignore/resultadosTDV/batch/PCALDA_test/programmedTests3.csv')
    # asigna la primera columna como índice
    tests.set_index('ID',inplace=True)
    # convierte el contenido de tests en booleano
    tests = tests.astype(bool)
    # por cada fila del dataframe
    for index, row in tests.iterrows():
        if row['done']==False:
            # convierte row en una tupla
            row = tuple(row)

            res_file_target='ignore/resultadosTDV/batch/PCALDA_test/'+str(index)
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
            epochs = 50
            max_days = 21

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
    
            # Carga de datos
            tdv_train,ltp_train,meteo_train,data_train=isl.cargaDatosTDV(year_train,"rht")

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



            for epoch in range(epochs):
                search_PCA_components = best_PCA_components
                search_TDV_days = best_TDV_days
                #search_meteo_days = best_meteo_days
                #search_TDV_sampling = best_TDV_sampling
                #search_meteo_sampling = best_meteo_sampling

                #lanza un proceso para cada elemento del batch

                #crea una lista con los inputs de cada proceso
                inputs = []

                #cuenta el número de elementos de row que son true
                num_params = sum(row)
                
                for batch_el in range(batch_size):
                    TDV_days=search_TDV_days+np.random.randint(max(-temperature*days_temp_scale,-search_TDV_days+1),min(temperature*days_temp_scale,max_days-search_TDV_days))
                    #print(search_PCA_components-temperature*pca_temp_scale-1,search_PCA_components-search_PCA_components,search_PCA_components+temperature*pca_temp_scale,1+num_params*TDV_days)
                    if int(max(-temperature*pca_temp_scale-1,-search_PCA_components))<int(min(temperature*pca_temp_scale,num_params*TDV_days-search_PCA_components)):
                        PCA_components=search_PCA_components+np.random.randint(max(-temperature*pca_temp_scale-1,-search_PCA_components),min(temperature*pca_temp_scale,num_params*TDV_days-search_PCA_components))
                    else:
                        PCA_components=num_params*TDV_days
                    inputs.append((PCA_components,TDV_days,years_test,year_train,row,tdv_train.copy(), data_train.copy(), tdv_tests.copy(), data_tests.copy()))

                #por cada elemento de la lista de inputs
                tested_already=[]
                for i in range(len(inputs)):
                    #separa los dos primeros elementos de la tupla
                    PCA_components,TDV_days = inputs[i][:2]
                    #si los parámetros no se han probado ya
                    if (PCA_components,TDV_days) not in tested_params:
                        #añade los parámetros a la lista de parámetros probados
                        tested_params.append((PCA_components,TDV_days))
                    else:
                        #almacena el índice del elemento que ya se ha probado
                        tested_already.append(i)
                #elimina los elementos que ya se han probado
                for i in range(len(tested_already)):
                    inputs.pop(tested_already[i]-i)

                
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
                #print(epoch/epochs*100,'% done, temperature: ',temperature)
                #guarda los parámetros de entrada y el resultado en el fichero de resultados
                for i in range(len(results)):
                    file.write(str(results[i][1]) + ',' + str(results[i][2]) + ',' + str(results[i][0]) + '\n')
                    file.flush()
            #cierra el fichero de resultados
            file.close()
            #añade el resultado al fichero meta
            m_file=open(meta_file,'a')
            m_file.write(str(index)+','+str(best_accuracy)+'\n')
            m_file.close()
            #marca el test como hecho
            tests.at[index,'done'] = True
            tests.to_csv('ignore/resultadosTDV/batch/PCALDA_test/programmedTests3.csv',index=True)
            print('test ',index,' done; best accuracy: ',best_accuracy)
