from cProfile import label
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

listest=['LTP diff_min_norm']
ventanas=[1]
f_olvido=[0]

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
for j in estimadorestr.columns:
    # Calcula la probabilidad base del estimador 
    media=estimadorestr[j].mean()
    varianza=estimadorestr[j].var()
    
    pgauss.loc[j,'Media base']=media
    pgauss.loc[j,'Varianza base']=varianza
print(pgauss)
# Añade a un dataframe la probabilidad de cada estado
pEstado=pd.DataFrame()
pEstado['Estado']=[1,2,3]
# Cambia el índice a la columna de estado
pEstado=pEstado.set_index('Estado')
# Calcula la probabilidad de cada estado por el número de veces que aparece en trdata
for i in [1,2,3]:
    pEstado.loc[i,'P']=trdata.loc[trdata==i].count()/trdata.count()
#fija las probabilidades a 1/3
# pEstado.loc[1,'P']=1/3
# pEstado.loc[2,'P']=1/3
# pEstado.loc[3,'P']=1/3
# Aplica naive Bayes a los datos de entrenamiento
# crea un dataframe para almacenar las probabilidades con los mismos índices de fila que trdata
prob=pd.DataFrame()
prob.index=trdata.index
# añade una columna por cada estado con su probabilidad
for i in pEstado.index:
    prob.loc[:,i]=pEstado.loc[i,'P']

# Calcula la probabilidad de cada estimador para cada estado
for j in estimadorestr.columns:
    for i in [1,2,3]:
        # Calcula la probabilidad de cada estimador para cada estado
        pnorm=stats.norm.pdf((estimadorestr.loc[:,j]-pgauss.loc[j,'Media '+str(i)])/pgauss.loc[j,'Varianza '+str(i)])
        pnormbase=stats.norm.pdf((estimadorestr.loc[:,j]-pgauss.loc[j,'Media base'])/pgauss.loc[j,'Varianza base'])
        #multiplica prob por pnorm
        prob.loc[:,i]=prob.loc[:,i]*pnorm/pnormbase
# Selecciona la probabilidad más alta en cada instante
print(prob)
# muestra la suma de las probabilidades de cada instante
print(prob.sum(axis=1))
prob=prob.idxmax(axis=1)
# Reordena los índices de prob
prob=prob.sort_index(axis='index',level=1)
trdata=trdata.sort_index(axis='index',level=1)
estimadorestr=estimadorestr.sort_index(axis='index',level=1)
print(prob)
# grafica prob junto a trdata
plt.figure()
trdata.plot()
prob.plot()
# grafica estimadorestr ltp diff_1h junto a trdata
plt.figure()
ax=trdata.plot()
estimadorestr.plot(ax=ax)
plt.show()