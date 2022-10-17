import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Ejecuta cargaRaw.py si no existe rawDiarios.csv o rawMinutales.csv
if not os.path.isfile("rawDiarios.csv") or not os.path.isfile("rawMinutales.csv"):
    os.system("python3 cargaRaw.py")

# Carga de datos
dfT = pd.read_csv("rawMinutales.csv",na_values='.')
dfT.loc[:,"Fecha"]=pd.to_datetime(dfT.loc[:,"Fecha"])# Fecha como datetime
dfT=dfT.drop_duplicates(subset="Fecha")
dfT.dropna(subset = ["Fecha"], inplace=True)
dfT=dfT.set_index("Fecha")

dfd = pd.read_csv("rawDiarios.csv",na_values='.')
dfd.loc[:,"Fecha"]=pd.to_datetime(dfd.loc[:,"Fecha"])# Fecha como datetime
dfd=dfd.drop_duplicates(subset="Fecha")
dfd.dropna(subset = ["Fecha"], inplace=True)
dfd=dfd.set_index("Fecha")

# dropea las columnas de dfd que empiezan por estado
dfd=dfd.drop(dfd.columns[dfd.columns.str.startswith('Estado')], axis=1)

# separa dfT en tdv y ltp en función del principio del nombre de cada columna y guarda el resto en meteo
tdv = dfT.loc[:,dfT.columns.str.startswith('TDV')]
ltp = dfT.loc[:,dfT.columns.str.startswith('LTP')]
meteo = dfT.drop(dfT.columns[dfT.columns.str.startswith('TDV')], axis=1)
meteo = meteo.drop(meteo.columns[meteo.columns.str.startswith('LTP')], axis=1)

# calcula la pendiente de ltp
ltp_diff = ltp.diff()

# calcula los estadisticos ltp sin normalizar
# calcula una moving average de los valores diarios de ltp para normalizar
ltp_ma = ltp.resample('D').mean().rolling(window=10).mean()
# calcula el máximo diario
ltpMax = ltp.resample('D').max()
# calcula el mínimo diario
ltpMin = ltp.resample('D').min()
# calcula la media como una medida del area bajo la curva
ltpArea = ltp.resample('D').mean()

# obtiene todos los cambios de signo de R_Neta_Avg en el dataframe meteo
signos = np.sign(meteo.loc[:,meteo.columns.str.startswith('R_Neta_Avg')]).diff()
# obtiene los cambios de signo que sean igual a -2 (de positivo a negativo)
signos = signos==-2
# elimina los valores falsos (que no sean cambios de signo)
signos = signos.replace(False,np.nan).dropna()

# obtiene los valores de la pendiente de ltp en los cambios de signo
ltp_diff_R_Neta_Avg = ltp_diff.loc[signos.index]
# remuestrea al último valor diario
ltp_diff_R_Neta_Avg = ltp_diff_R_Neta_Avg.resample('D').last()

# obtiene la pendiente mínima diaria
ltp_diff_min = ltp_diff.resample('D').min()

# obtiene la pendiente media entre el cambio de signo y una hora después
# obtiene el valor de ltp en el cambio de signo de la radiación
ltp_R_Neta_Avg = ltp.loc[signos.index]
ltp_R_Neta_Avg = ltp_R_Neta_Avg.resample('D').last()
# obtiene el valor de ltp una hora después
ltp_R_Neta_Avg_1h = ltp.loc[signos.index+pd.Timedelta(hours=1)]
ltp_R_Neta_Avg_1h = ltp_R_Neta_Avg_1h.resample('D').last()
# obtiene la pendiente media en ese intervalo
ltp_diff_1h = ltp_R_Neta_Avg_1h - ltp_R_Neta_Avg

# normalización
# obtiene el primer valor del día
ltp0 = ltp.resample('D').first()

# obtiene la media diaria
ltp_m = ltp.dropna().resample('D').mean()
# remuestrea a minutal con el valor del día
ltp_m = ltp_m.resample('T').ffill()

# obtiene la varianza diaria
ltp_v = ltp.dropna().resample('D').var()
# remuestrea a minutal con el valor del día
ltp_v = ltp_v.resample('T').ffill()

# normaliza los estadísticos en torno al primer valor del día
ltpMaxn0 = ltpMax-ltp0
ltpMinn0 = ltpMin-ltp0
ltpArean0 = ltpArea-ltp0

#normaliza los estadísticos en torno a la media móvil
ltpMaxnma = ltpMax-ltp_ma
ltpMinnma = ltpMin-ltp_ma
ltpAreanma = ltpArea-ltp_ma

# normaliza los estadísticos con media nula y varianza unidad (diaria)
ltpMaxnmnvu = (ltpMax-ltp_m)/ltp_v
ltpMinnmnvu = (ltpMin-ltp_m)/ltp_v
#ltpAreanmnvu = (ltpArea-ltp_m)/ltp_v # No tiene sentido esta normalización para el area porque tiene relación proporcional con la media diaria; media diaria nula implica area nula

# normaliza el área restando el valor mínimo y dividiendo por el valor máximo menos el mínimo
ltpArean0_1 = (ltpArea-ltpMin)/(ltpMax-ltpMin)

# normaliza ltp para calcular pendientes
ltpn = (ltp-ltp_m)/ltp_v
ltpn_diff = ltpn.diff()

# obtiene la pendiente mínima diaria normalizada
ltpn_diff_min = ltpn_diff.resample('D').min()

# obtiene los valores de la pendiente de ltp normalizada en los cambios de signo
ltpn_diff_R_Neta_Avg = ltpn_diff.loc[signos.index]
# remuestrea al último valor diario
ltpn_diff_R_Neta_Avg = ltpn_diff_R_Neta_Avg.resample('D').last()

# obtiene la pendiente media normalizada entre el cambio de signo y una hora después
ltpn_R_Neta_Avg = ltpn.loc[signos.index]
ltpn_R_Neta_Avg = ltpn_R_Neta_Avg.resample('D').last()
ltpn_R_Neta_Avg_1h = ltpn.loc[signos.index+pd.Timedelta(hours=1)]
ltpn_R_Neta_Avg_1h = ltpn_R_Neta_Avg_1h.resample('D').last()
ltpn_diff_1h = ltpn_R_Neta_Avg_1h - ltpn_R_Neta_Avg

#combina los dataframes de los estadísticos renombrando las columnas
ltpMax.columns = ltpMax.columns.str.replace('LTP','LTP Max') # ltp máximo sin normalizar
ltpMin.columns = ltpMin.columns.str.replace('LTP','LTP Min') # ltp mínimo sin normalizar
ltpArea.columns = ltpArea.columns.str.replace('LTP','LTP Area') # area ltp sin normalizar
ltpMaxn0.columns = ltpMaxn0.columns.str.replace('LTP','LTP Max_n0') # ltp máximo normalizado al primer valor del día
ltpMinn0.columns = ltpMinn0.columns.str.replace('LTP','LTP Min_n0') # ltp mínimo normalizado al primer valor del día
ltpArean0.columns = ltpArean0.columns.str.replace('LTP','LTP Area_n0') # area ltp normalizada al primer valor del día
ltpMaxnma.columns = ltpMaxnma.columns.str.replace('LTP','LTP Max_ma') # ltp máximo normalizado a la media móvil
ltpMinnma.columns = ltpMinnma.columns.str.replace('LTP','LTP Min_ma') # ltp mínimo normalizado a la media móvil
ltpAreanma.columns = ltpAreanma.columns.str.replace('LTP','LTP Area_ma') # area ltp normalizada a la media móvil
ltpMaxnmnvu.columns = ltpMaxnmnvu.columns.str.replace('LTP','LTP Max_mnvu') # ltp máximo normalizado a media nula y varianza unidad
ltpMinnmnvu.columns = ltpMinnmnvu.columns.str.replace('LTP','LTP Min_mnvu') # ltp mínimo normalizado a media nula y varianza unidad
ltpArean0_1.columns = ltpArean0_1.columns.str.replace('LTP','LTP Area_n0_1') # area ltp normalizada restando el valor mínimo y dividiendo por el valor máximo menos el mínimo
ltp_diff_R_Neta_Avg.columns = ltp_diff_R_Neta_Avg.columns.str.replace('LTP','LTP diff_R_Neta_Avg') # pendiente media en el cambio de signo
ltp_diff_1h.columns = ltp_diff_1h.columns.str.replace('LTP','LTP diff_1h') # pendiente media entre el cambio de signo y una hora después
ltp_diff_min.columns = ltp_diff_min.columns.str.replace('LTP','LTP diff_min') # pendiente mínima diaria
ltpn_diff_R_Neta_Avg.columns = ltpn_diff_R_Neta_Avg.columns.str.replace('LTP','LTP diff_R_Neta_Avg_norm') # pendiente media normalizada en el cambio de signo
ltpn_diff_1h.columns = ltpn_diff_1h.columns.str.replace('LTP','LTP diff_1h_norm') # pendiente media normalizada entre el cambio de signo y una hora después
ltpn_diff_min.columns = ltpn_diff_min.columns.str.replace('LTP','LTP diff_min_norm') # pendiente mínima normalizada diaria

ltp_est = pd.concat([ltpMax,ltpMin,ltpArea,ltpMaxn0,ltpMinn0,ltpArean0,ltpMaxnma,ltpMinnma,ltpAreanma,ltpMaxnmnvu,ltpMinnmnvu,ltpArean0_1,ltp_diff_R_Neta_Avg,ltp_diff_1h,ltp_diff_min,ltpn_diff_R_Neta_Avg,ltpn_diff_1h,ltpn_diff_min],axis=1)
# asegura que ltp_est tiene frecuencia diaria
ltp_est = ltp_est.resample('D').fillna(method='pad')
ltp_est.dropna()

# guarda los estadísticos en estadisticosDiarios.csv
ltp_est.to_csv('estadisticosDiarios.csv',index=True)

# crea un dataframe con los estadísticos y su ecuación
ltp_eq=pd.DataFrame(columns=['LTP Max','LTP Min','LTP Area','LTP Max_n0','LTP Min_n0','LTP Area_n0','LTP Max_ma','LTP Min_ma','LTP Area_ma','LTP Max_mnvu','LTP Min_mnvu','LTP Area_n0_1','LTP diff_R_Neta_Avg','LTP diff_1h','LTP diff_min','LTP diff_R_Neta_Avg_norm','LTP diff_1h_norm','LTP diff_min_norm'])
ltp_eq.loc[0]=['max(LTP)','min(LTP)','mean(LTP)','max(LTP)-LTP(00:00)','min(LTP)-LTP(00:00)','mean(LTP)-LTP(00:00)','max(LTP)-moving_avg(LTP,10D)','min(LTP)-moving_avg(LTP,10D)','mean(LTP)-moving_avg(LTP,10D)','(max(LTP)-mean(LTP))/var(LTP)','(min(LTP)-mean(LTP))/var(LTP)','(mean(LTP)-min(LTP))/(max(LTP)-min(LTP))','LTP(sign_change(R_Neta_Avg)-1min)-LTP(sign_change(R_Neta_Avg))','LTP(sign_change(R_Neta_Avg)-LTP(sign_change(R_Neta_Avg)+1h)','min(LTP(x)-LTP(x-1min))','(LTP(sign_change(R_Neta_Avg)-1min)-LTP(sign_change(R_Neta_Avg)))/var(LTP)','(LTP(sign_change(R_Neta_Avg))-LTP(sign_change(R_Neta_Avg)+1h))/var(LTP)','min((LTP(x)-LTP(x-1min))/var(LTP))']

ltp_eq.to_csv('estDiariosEcuaciones.csv',index=False)
