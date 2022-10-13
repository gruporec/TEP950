from cProfile import label
from codecs import ignore_errors
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import itertools as it
from os.path import exists
import scipy.stats as stats

# listest=['LTP diff_1h','LTP diff_R_Neta_Avg_norm','LTP Area_n0','LTP diff_R_Neta_Avg','LTP Min_n0','LTP Max_n0','LTP diff_1h','LTP diff_R_Neta_Avg_norm','LTP Area_n0','LTP diff_R_Neta_Avg','LTP Min_n0','LTP Max_n0']
# ventanas=[1,10,2,10,1,1,1,1,1,1,1,1]
# f_olvido=[0.8,0.95,0.8,0.8,0.8,0.8,0,0,0,0,0,0]

# listest=['LTP diff_1h','LTP Area_n0','LTP Min_n0','LTP Max_n0']
# ventanas=[1,1,1,1]
# f_olvido=[0,0,0,0]

# numero_elementos_max=12
# listest=['LTP diff_1h','LTP Area_n0']
# ventanas=[1,1]
# f_olvido=[0,0]

# listest=['LTP diff_1h','LTP diff_1h']
# ventanas=[1,5]
# f_olvido=[0,1]

# Carga estimadores
estimadoresraw = pd.read_csv("estadisticosDiarios.csv")
estimadoresraw['Fecha'] = pd.to_datetime(estimadoresraw['Fecha'])
estimadoresraw=estimadoresraw.set_index('Fecha')
estimadoresraw=estimadoresraw.replace(-np.inf, np.nan)
estimadoresraw=estimadoresraw.replace(np.inf, np.nan)

# Carga datos de  validacion
valdatapd=pd.read_csv("validacion.csv")
valdatapd.dropna(inplace=True)
valdatapd['Fecha'] = pd.to_datetime(valdatapd['Fecha'])
valdatapd.set_index('Fecha',inplace=True)

 # Carga de datos de entrenamiento
trdatapd=pd.read_csv("calibracion3.csv")
trdatapd.dropna(inplace=True)
trdatapd['Fecha'] = pd.to_datetime(trdatapd['Fecha'])
trdatapd.set_index('Fecha',inplace=True)

# Separa los índices de las columnas de estimadores en dos índices: sensores y estimadores
estimadoresraw.columns = pd.MultiIndex.from_tuples(estimadoresraw.columns.str.split(' ').tolist())
# Elimina el índice 'LTP'
estimadoresraw.columns = estimadoresraw.columns.droplevel(0)
# Vuelve a añadir el prefijo
estimadoresraw = estimadoresraw.add_prefix('LTP ')

# Crea un dataframe vacío para los datos con factor de olvido
estimadores = pd.DataFrame()

# crea la lista de estimadores con el primer índice de los datos de estimadores
listest=estimadoresraw.xs('LTP Control_1_1', level=1, axis=1).columns
# crea un vector de ventanas del mismo tamaño que la lista de estimadores y con valores 1
ventanas=np.ones(len(listest)).astype(int)
f_olvido=np.zeros(len(listest)).astype(int)
# Aplica el factor de olvido en la ventana correspondiente
for i in range(len(listest)):
    sumdata=estimadoresraw.loc[:,listest[i]].copy().fillna(method='ffill').fillna(0)
    shiftdata=estimadoresraw.loc[:,listest[i]].copy().fillna(method='ffill').fillna(0)
    for j in range(1,ventanas[i]+1):
        shiftdata=shiftdata.shift(1).fillna(0)*f_olvido[i]
        sumdata=shiftdata+sumdata
    strcolumn=listest[i]+","+str(ventanas[i])+","+str(f_olvido[i])
    sumdata.columns = pd.MultiIndex.from_product([sumdata.columns, [strcolumn]])
    estimadores=pd.concat([estimadores,sumdata],axis=1)
    estimadores=estimadores.dropna()

# Selecciona los índices comunes de los datos de entrenamiento y estimadores
trindex = trdatapd.index.intersection(estimadores.index)
trdatapd=trdatapd.loc[trindex]
estimadorestr=estimadores.loc[trindex]

# Selecciona las columnas de entrenamiento en estimadores
estimadorestr=estimadorestr.loc[:,trdatapd.columns]

# Ordena los dataframes de entrenamiento
estimadorestr=estimadorestr.swaplevel(0,1,axis=1)
estimadorestr=estimadorestr.stack()
trdata=trdatapd.stack()

# # Comprueba si existe el archivo de resultados
# if not exists("resultadosNaiveBayes.csv"):
#     # Si no existe, crea un archivo con el encabezado de los resultados
#     f=open("resultadosNaiveBayes.csv","x")
#     f.write("Estimador,sigma1,m1,sigma2,m2,sigma3,m3,,Estado,P,Acierto\n")
# else:
#     # Si existe, abre el archivo para escribir en él
#     f=open("resultadosNaiveBayes.csv","a")

# Crea un dataframe para almacenar los parámetros de la distribución de probabilidad de cada estimador (media y varianza)
pgauss = pd.DataFrame()

# Calcula la media y la varianza de cada estimador a partir de los datos de entrenamiento cuando trdata vale 1, 2 o 3
for i in [1,2,3]:
    # Selecciona los datos de entrenamiento cuando trdata vale i
    trdatai=trdata.loc[trdata==i]
    estimadorestri=estimadorestr.loc[trdata==i]
    # Calcula la media y la varianza de cada estimador
    for j in estimadorestr.columns:
        # Calcula la media y la varianza de cada estimador
        media=estimadorestri[j].mean()
        varianza=estimadorestri[j].var()
        # Usa como íncice de pgauss la columna de estimadores
        pgauss.loc[j,'Media '+str(i)]=media
        pgauss.loc[j,'Varianza '+str(i)]=varianza
# Obtiene la diferencia entre la media 1 y la media 2
difmed1=abs(pgauss['Media 1']-pgauss['Media 2'])
# Obtiene la diferencia entre la media 2 y la media 3
difmed2=abs(pgauss['Media 2']-pgauss['Media 3'])
# Guarda la menor diferencia entre las medias
pgauss['Diferencia medias']=np.minimum(difmed1,difmed2)
# Obtiene la mayor varianza entre la 1, la 2 y la 3
pgauss['Varianza max']=np.maximum(pgauss['Varianza 1'],np.maximum(pgauss['Varianza 2'],pgauss['Varianza 3']))
# Obtiene la relación entre la varianza y la diferencia de medias
pgauss['Relacion varianza']=pgauss['Varianza max']/pgauss['Diferencia medias']
for j in estimadorestr.columns:
    # Calcula la probabilidad base del estimador 
    media=estimadorestr[j].mean()
    varianza=estimadorestr[j].var()
    
    pgauss.loc[j,'Media base']=media
    pgauss.loc[j,'Varianza base']=varianza
print(pgauss)
# Añade un nombre al índice de pgauss
pgauss.index.name='Estimador,ventana,f_olvido'
# Guarda el archivo
pgauss.to_csv("pgauss.csv")