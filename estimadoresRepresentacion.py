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

listest=['LTP diff_1h']
ventanas=[1]
f_olvido=[0.8]
theta=[1,0]

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
# plt.plot(yv,label='Estados hídricos')
# plt.plot(yvpred,label='Estimador LTP diff_1h_0.8_1')
# plt.legend()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(yv, color='g')
ax2.plot(yvpred, color='b')

#ax1.set_xlabel('X data')
ax1.set_ylabel('Estado hídrico', color='g')
ax2.set_ylabel('Estimador LTP diff_1h_0.8_1', color='b')

plt.show()