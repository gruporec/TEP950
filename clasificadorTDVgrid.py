import sys
from time import time
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

year_train="2014"
year_test="2019"

PCA_components=10
TDV_dev_days=14
TDV_dev_inc_days=14
TDV_inc_days=14
meteo_days=14

PCA_components_grid=range(10,16,1)
TDV_dev_days_grid=range(1,7,1)
TDV_dev_inc_days_grid=range(1,7,1)
TDV_inc_days_grid=range(1,12,3)
meteo_days_grid=range(11,22,3)

best_PCA_components=0
best_TDV_dev_days=0
best_TDV_dev_inc_days=0
best_TDV_inc_days=0
best_meteo_days=0
best_accuracy=0
# Ejecuta cargaRaw.py si no existe rawDiarios.csv o rawMinutales.csv
if not os.path.isfile("rawDiarios"+year_train+".csv") or not os.path.isfile("rawMinutales"+year_train+".csv"):
    os.system("python3 cargaRaw.py")
if not os.path.isfile("rawDiarios"+year_test+".csv") or not os.path.isfile("rawMinutales"+year_test+".csv"):
    os.system("python3 cargaRaw.py")

# Carga de datos
tdv_train,ltp_train,meteo_train,data_train=isl.cargaDatosTDV(year_train,"rht")
tdv_test,ltp_test,meteo_test,data_test=isl.cargaDatosTDV(year_test,"rht")

datacols_train = list(data_train.columns)
dataind_train = data_train.index
datacols_test = list(data_test.columns)
dataind_test = data_test.index

data_train=data_train.groupby(data_train.index.date).max()
data_test=data_test.groupby(data_test.index.date).max()

tdv_train=tdv_train[datacols_train]
tdv_test=tdv_test[datacols_test]

tdvavg24_train = tdv_train.rolling(window=24*60,center=True).mean()
tdvavg24_test = tdv_test.rolling(window=24*60,center=True).mean()
# calcula el valor medio de tdv para cada dia
tdv_medio_train = tdv_train.groupby(tdv_train.index.date).mean()
tdv_medio_test = tdv_test.groupby(tdv_test.index.date).mean()
# calcula el valor de la desviación estándar de tdv para cada dia
tdv_std_train = tdv_train.groupby(tdv_train.index.date).std()
tdv_std_test = tdv_test.groupby(tdv_test.index.date).std()

tdv_max_train = tdv_train.groupby(tdv_train.index.date).max()
tdv_max_test = tdv_test.groupby(tdv_test.index.date).max()

tdv_min_train = tdv_train.groupby(tdv_train.index.date).min()
tdv_min_test = tdv_test.groupby(tdv_test.index.date).min()

tdv_amp_train = tdv_max_train - tdv_min_train
tdv_amp_test = tdv_max_test - tdv_min_test

tdv_cummax_train = tdv_medio_train.cummax()
tdv_cummax_test = tdv_medio_test.cummax()

tdv_de_train = tdv_medio_train - tdv_cummax_train
tdv_de_test = tdv_medio_test - tdv_cummax_test

meteo_mean_train = meteo_train.groupby(meteo_train.index.date).mean()
meteo_mean_test = meteo_test.groupby(meteo_test.index.date).mean()

tdv_dataset_train = [data_train, tdv_amp_train.loc[dataind_train],tdv_de_train.loc[dataind_train]]
tdv_dataset_test = [data_test, tdv_amp_test.loc[dataind_test],tdv_de_test.loc[dataind_test]]

for PCA_components in PCA_components_grid:
    for TDV_dev_days in TDV_dev_days_grid:
        for TDV_dev_inc_days in TDV_dev_inc_days_grid:
            for TDV_inc_days in TDV_inc_days_grid:
                for meteo_days in meteo_days_grid:
                    #comprueba si va a haber suficientes característcas para el PCA
                    if PCA_components > TDV_dev_days+TDV_dev_inc_days+TDV_inc_days+meteo_days*3:
                        continue
                    #añade las columnas de la desviación de tdv de los dias anteriores
                    for i in range(1,TDV_dev_days,1):
                        tdv_de_prev_train = tdv_de_train.copy()
                        tdv_de_prev_test = tdv_de_test.copy()

                        tdv_de_prev_train.index = pd.to_datetime(tdv_de_prev_train.index)
                        tdv_de_prev_test.index = pd.to_datetime(tdv_de_prev_test.index)

                        tdv_de_prev_train.index = tdv_de_prev_train.index + pd.DateOffset(days=i)
                        tdv_de_prev_test.index = tdv_de_prev_test.index + pd.DateOffset(days=i)

                        tdv_de_prev_train = tdv_de_prev_train.groupby(tdv_de_prev_train.index.date).max()
                        tdv_de_prev_test = tdv_de_prev_test.groupby(tdv_de_prev_test.index.date).max()

                        tdv_dataset_train.append(tdv_de_prev_train.loc[tdv_de_prev_train.index.intersection(data_train.index)])
                        tdv_dataset_test.append(tdv_de_prev_test.loc[tdv_de_prev_test.index.intersection(data_test.index)])

                    #añade las columnas del incremento de la desviación de tdv de los dias anteriores
                    for i in range(1,TDV_dev_inc_days,1):
                        tdv_de_prev_train = tdv_de_train.copy()
                        tdv_de_prev_test = tdv_de_test.copy()

                        tdv_de_prev_train.index = pd.to_datetime(tdv_de_prev_train.index)
                        tdv_de_prev_test.index = pd.to_datetime(tdv_de_prev_test.index)

                        tdv_de_prev_train.index = tdv_de_prev_train.index + pd.DateOffset(days=i)
                        tdv_de_prev_test.index = tdv_de_prev_test.index + pd.DateOffset(days=i)

                        tdv_de_prev_train = tdv_de_train - tdv_de_prev_train
                        tdv_de_prev_test = tdv_de_test - tdv_de_prev_test

                        tdv_dataset_train.append(tdv_de_prev_train.loc[tdv_de_prev_train.index.intersection(data_train.index)])
                        tdv_dataset_test.append(tdv_de_prev_test.loc[tdv_de_prev_test.index.intersection(data_test.index)])

                        #añade las columnas del incremento de tdv de los dias anteriores
                    for i in range(1,TDV_inc_days,1):
                        tdv_diff_train = tdv_medio_train.diff(i)
                        tdv_diff_test = tdv_medio_test.diff(i)

                        tdv_dataset_train.append(tdv_diff_train.loc[tdv_diff_train.index.intersection(data_train.index)])
                        tdv_dataset_test.append(tdv_diff_test.loc[tdv_diff_test.index.intersection(data_test.index)])

                    tdv_data_train=pd.concat(tdv_dataset_train,axis=1, keys=['stress level', 'Amp', 'Dev']+['Dev ' + str(s) + 'd' for s in [*range(1,TDV_dev_days,1)]]+['Dev Inc ' + str(s) + 'd' for s in [*range(1,TDV_dev_inc_days,1)]]+['Inc ' + str(s) + 'd' for s in [*range(1,TDV_inc_days,1)]])
                    tdv_data_test=pd.concat(tdv_dataset_test,axis=1, keys=['stress level', 'Amp', 'Dev']+['Dev ' + str(s) + 'd' for s in [*range(1,TDV_dev_days,1)]]+['Dev Inc ' + str(s) + 'd' for s in [*range(1,TDV_dev_inc_days,1)]]+['Inc ' + str(s) + 'd' for s in [*range(1,TDV_inc_days,1)]])

                    #meteo_sensor=pd.concat([meteo_mean]*len(valdatapd.columns), axis=1, keys=valdatapd.columns)
                    meteo_sensor_train=pd.concat([meteo_mean_train]*len(data_train.columns), axis=1, keys=data_train.columns)
                    meteo_sensor_test=pd.concat([meteo_mean_test]*len(data_test.columns), axis=1, keys=data_test.columns)

                    #invierte el orden de los indices de columnas de meteo_sensor
                    meteo_sensor_train.columns = meteo_sensor_train.columns.swaplevel(0,1)
                    meteo_sensor_test.columns = meteo_sensor_test.columns.swaplevel(0,1)

                    #añade las columnas de meteo a tdv_data donde coincidan en el segundo índice
                    tdv_data_train = tdv_data_train.merge(meteo_sensor_train, how='left', left_index=True, right_index=True)
                    tdv_data_test = tdv_data_test.merge(meteo_sensor_test, how='left', left_index=True, right_index=True)

                    #añade las columnas de meteo de dias anteriores
                    for i in range(1,meteo_days,1):
                        meteo_prev_train = meteo_mean_train.copy()
                        meteo_prev_test = meteo_mean_test.copy()

                        meteo_prev_train.index = pd.to_datetime(meteo_prev_train.index)
                        meteo_prev_test.index = pd.to_datetime(meteo_prev_test.index)

                        meteo_prev_train.index = meteo_prev_train.index + pd.Timedelta(days=i)
                        meteo_prev_test.index = meteo_prev_test.index + pd.Timedelta(days=i)

                        meteo_prev_train = meteo_prev_train.groupby(meteo_prev_train.index.date).max()
                        meteo_prev_test = meteo_prev_test.groupby(meteo_prev_test.index.date).max()

                        meteo_prev_train = pd.concat([meteo_prev_train]*len(data_train.columns), axis=1, keys=data_train.columns)
                        meteo_prev_test = pd.concat([meteo_prev_test]*len(data_test.columns), axis=1, keys=data_test.columns)

                        meteo_prev_train.columns = meteo_prev_train.columns.swaplevel(0,1)
                        meteo_prev_test.columns = meteo_prev_test.columns.swaplevel(0,1)

                        tdv_data_train = tdv_data_train.merge(meteo_prev_train, how='left', left_index=True, right_index=True, suffixes=('', ' ' + str(i) + 'd'))
                        tdv_data_test = tdv_data_test.merge(meteo_prev_test, how='left', left_index=True, right_index=True, suffixes=('', ' ' + str(i) + 'd'))

                    tdv_data_train = tdv_data_train.stack(1)
                    tdv_data_test = tdv_data_test.stack(1)

                    tdv_data_train = tdv_data_train.dropna(how='any')
                    tdv_data_test = tdv_data_test.dropna(how='any')

                    x_train = tdv_data_train.drop(['stress level'], axis=1)
                    y_train = tdv_data_train['stress level']

                    x_test = tdv_data_test.drop(['stress level'], axis=1)
                    y_test = tdv_data_test['stress level']

                    pca = skdecomp.PCA(n_components=PCA_components)
                    pca.fit(x_train)

                    x_train = pca.transform(x_train)
                    x_test = pca.transform(x_test)

                    clf = sklda.LinearDiscriminantAnalysis(solver='svd')
                    clf.fit(x_train, y_train)

                    x_new_train = clf.transform(x_train)
                    x_new_test = clf.transform(x_test)

                    ypred_train = clf.predict(x_train)
                    ypred_test = clf.predict(x_test)

                    bcm = skmetrics.confusion_matrix(y_test, ypred_test,normalize='true')
                    accuracy = skmetrics.balanced_accuracy_score(y_test, ypred_test)

                    # print('Pca components: ' + str(PCA_components) + ' TDV dev days: ' + str(TDV_dev_days) + ' TDV dev inc days: ' + str(TDV_dev_inc_days) + ' TDV inc days: ' + str(TDV_inc_days) + ' Meteo days: ' + str(meteo_days))
                    # print('Accuracy: ' + str(accuracy))
                    # print(bcm)
                    # print('')
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_bcm = bcm
                        best_pca = PCA_components
                        best_tdv_dev_days = TDV_dev_days
                        best_tdv_dev_inc_days = TDV_dev_inc_days
                        best_tdv_inc_days = TDV_inc_days
                        best_meteo_days = meteo_days
print('')
print('Best results:')
print('Pca components: ' + str(best_pca) + ' TDV dev days: ' + str(best_tdv_dev_days) + ' TDV dev inc days: ' + str(best_tdv_dev_inc_days) + ' TDV inc days: ' + str(best_tdv_inc_days) + ' Meteo days: ' + str(best_meteo_days))
print('Accuracy: ' + str(best_accuracy))
print(best_bcm)