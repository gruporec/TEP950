import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Estimadores a usar
#listest=['LTP diff_1h','LTP diff_R_Neta_Avg_norm','LTP Area_n0','LTP diff_R_Neta_Avg','LTP Min_n0','LTP Max_n0','LTP diff_1h','LTP diff_R_Neta_Avg_norm','LTP Area_n0','LTP diff_R_Neta_Avg','LTP Min_n0','LTP Max_n0']
#ventanas=[1,10,2,10,1,1,1,1,1,1,1,1]
#f_olvido=[0.8,0.95,0.8,0.8,0.8,0.8,0,0,0,0,0,0]

# listest=['LTP Area_n0','LTP Min_n0','LTP Min_n0','LTP diff_R_Neta_Avg']
# ventanas=[2,1,1,1]
# f_olvido=[0.8,0,0.8,0]
# theta=[-0.0052611184442812615,0.007665840044553087,-0.005701819434972201,0.5406726770664151,2.0267128066375357]
# clas=[1.6019999999999999,1.92]

listest=['LTP Area_n0','LTP Area_n0','LTP Max_n0','LTP Min_n0','LTP diff_1h']
ventanas=[1,2,1,1,1,1]
f_olvido=[0,0.8,0.8,0.8,0.8,0]
theta=[0.025663055840714655,-0.01704297016171391,-0.005719137734447238,-0.00937728352820993,-0.005105697049724101,1.5738771973981518]
clas=[1.246,1.31]

# listest=['LTP Max_n0','LTP Min_n0','LTP diff_R_Neta_Avg']
# ventanas=[1,1,10,1]
# f_olvido=[0.8,0.8,0.8]
# theta=[-0.004196376924938115,-0.0036279534537991542,0.356803937815426,2.2241414382800366]

# Carga estimadores
estimadoresraw = pd.read_csv("estadisticosDiarios.csv")
estimadoresraw['Fecha'] = pd.to_datetime(estimadoresraw['Fecha'])
estimadoresraw=estimadoresraw.set_index('Fecha')
estimadoresraw=estimadoresraw.replace(-np.inf, np.nan)
estimadoresraw=estimadoresraw.replace(np.inf, np.nan)

# Carga datos de  validacion
valdatapd=pd.read_csv("validacion2019.csv")
valdatapd.dropna(inplace=True)
valdatapd['Fecha'] = pd.to_datetime(valdatapd['Fecha'])
valdatapd.set_index('Fecha',inplace=True)

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
    estdata=estimadoresraw.loc[:,listest[i]].copy()
    shiftdata=estimadoresraw.loc[:,listest[i]].copy()
    for j in range(1,ventanas[i]+1):
        shiftdata=shiftdata.shift(1)*f_olvido[i]
        sumdata=shiftdata+estdata
    sumdata.columns = pd.MultiIndex.from_product([sumdata.columns, [listest[i]+","+str(ventanas[i])+","+str(f_olvido[i])]])
    estimadores=pd.concat([estimadores,sumdata],axis=1)
    estimadores=estimadores.dropna()


# Selecciona los indices comunes de los datos de validación y estimadores
vindex = valdatapd.index.intersection(estimadores.index)
valdatapd=valdatapd.loc[vindex]
estimadoresv=estimadores.loc[vindex]

# Selecciona las columnas de validación en estimadores
estimadoresv=estimadoresv.loc[:,valdatapd.columns]

# Ordena los dataframes de validación
estimadoresv=estimadoresv.swaplevel(0,1,axis=1)
estimadoresv=estimadoresv.stack()
vdata=valdatapd.stack()

# añade una columna llena de 1 para el término independiente
estimadoresv['Uno']=1

# ordena las filas de estimadoresv y vdata por su segundo índice
estimadoresv=estimadoresv.sort_index(level=1)
vdata=vdata.sort_index(level=1)

# Convierte los dataframes a matrices
varphiv=estimadoresv.values
yv=vdata.values

# Aplica los coeficientes
yvpred=np.dot(varphiv,theta)

# Grafica los datos de validación y los estimados
plt.plot(yv,label='Datos de validación')
plt.plot(yvpred,label='Estimados')
plt.legend()

# crea un vector con los datos de yvpred cuando yv es 1
yvpred1=yvpred[yv==1]
# crea un vector con los datos de yvpred cuando yv es 2
yvpred2=yvpred[yv==2]
# crea un vector con los datos de yvpred cuando yv es 3
yvpred3=yvpred[yv==3]

# ordena los tres vectores de numpy por su valor
yvpred1=np.sort(yvpred1)
yvpred2=np.sort(yvpred2)
yvpred3=np.sort(yvpred3)

# crea un vector desde 1 hasta 100 con el mismo número de elementos que cada uno de los anteriores
x1=np.linspace(1,100,len(yvpred1))
x2=np.linspace(1,100,len(yvpred2))
x3=np.linspace(1,100,len(yvpred3))

# Aplica el selector
# crea un vector de salida del selector con el tamaño de yvpred y valor de 3
yvpredsel=np.full(len(yvpred),3)
# fija el valor del vector de salida del selector a 2 cuando el valor de yvpred es menor que stepv23
yvpredsel[yvpred<clas[1]]=2
# fija el valor del vector de salida del selector a 1 cuando el valor de yvpred es menor que stepv12
yvpredsel[yvpred<clas[0]]=1

# grafica las tres curvas en una figura nueva
plt.figure()
plt.plot(yvpred1,x1,label='y=1')
plt.plot(yvpred2,x2,label='y=2')
plt.plot(yvpred3,x3,label='y=3')
plt.legend()

# grafica las tres curvas y añade lineas verticales en los parámetros del clasificador
plt.figure()
plt.plot(yvpred1,x1,label='y=1')
plt.plot(yvpred2,x2,label='y=2')
plt.plot(yvpred3,x3,label='y=3')
plt.axvline(x=clas[0],color='k',label='clasificador 1')
plt.axvline(x=clas[1],color='k',label='clasificador 2')
plt.legend()

#grafica los datos de validación y los estimados con el selector
plt.figure()
plt.plot(yv,label='Datos de validación')
plt.plot(yvpredsel,label='Estimados')
plt.legend()
plt.show()