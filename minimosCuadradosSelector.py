import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics

# Lista de estimadores a usar
#listest=['LTP diff_1h','LTP diff_R_Neta_Avg_norm','LTP Area_n0','LTP diff_R_Neta_Avg']
#ventanas=[1,10,2,10]
#f_olvido=[0.8,0.95,0.8,0.8]

listest=['LTP Area_n0',  'LTP Area_n0',  'LTP Max_n0',  'LTP Min_n0',  'LTP diff_1h']
ventanas=[1,  2,  1,  1,  1,  1]
f_olvido=[0, 0.8, 0.8,  0.8,  0.8,  0]

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
    estdata=estimadoresraw.loc[:,listest[i]].copy()
    shiftdata=estimadoresraw.loc[:,listest[i]].copy()
    for j in range(1,ventanas[i]+1):
        shiftdata=shiftdata.shift(1)*f_olvido[i]
        sumdata=shiftdata+estdata
    sumdata.columns = pd.MultiIndex.from_product([sumdata.columns, [listest[i]+str(i)]])
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

# Añade una columna llena de 1 para el término independiente
estimadorestr['Uno']=1

# Convierte los dataframes a matrices
varphitr=estimadorestr.values
ytr=trdata.values

# Cuenta las veces que se repite el valor 3 en ytr
count3=np.count_nonzero(ytr==3)

# Cuenta las veces que se repite el valor 2 en ytr
count2=np.count_nonzero(ytr==2)

# Cuenta las veces que se repite el valor 1 en ytr
count1=np.count_nonzero(ytr==1)

# Crea un vector de pesos sustituyendo cada valor por sum/n, donde n es el número de veces que se repite el valor y sum es el numero total de valores
sum=count1+count2+count3
w=np.zeros(len(ytr))
w[ytr==3]=sum/count3
w[ytr==2]=sum/count2
w[ytr==1]=sum/count1

# Aplica el peso a los datos
varphitrw = varphitr * np.sqrt(w[:,np.newaxis])
ytrw = ytr * np.sqrt(w)

# Calcula los coeficientes
coef=np.linalg.lstsq(varphitrw,ytrw)
theta=coef[0]

# Calcula la y estimada
ytrpred=varphitr.dot(coef[0])

# Calcula el error cuando la y vale 1
# Obtiene los puntos donde ytr=1
index1=np.where(ytr==1)

# Calcula el error cuando la y vale 2
# Obtiene los puntos donde ytr=2
index2=np.where(ytr==2)

# Calcula el error cuando la y vale 3
# Obtiene los puntos donde ytr=3
index3=np.where(ytr==3)

# Extrae los datos de las columnas de ytrpred para las filas donde ytr=1´
ytrpred1=ytrpred[index1]
# Ordena los valores de menor a mayor
ytrpred1=np.sort(ytrpred1)
# Crea un vector del mismo tamaño que ytrpred1 con valores de 0 a 100
ytrpred1v=np.linspace(0,100,len(ytrpred1))

# Extrae los datos de las columnas de ytrpred para las filas donde ytr=2
ytrpred2=ytrpred[index2]
# Ordena los valores de menor a mayor
ytrpred2=np.sort(ytrpred2)
# Crea un vector del mismo tamaño que ytrpred2 con valores de 0 a 100
ytrpred2v=np.linspace(0,100,len(ytrpred2))

# Extrae los datos de las columnas de ytrpred para las filas donde ytr=3
ytrpred3=ytrpred[index3]
# Ordena los valores de menor a mayor
ytrpred3=np.sort(ytrpred3)
# Crea un vector del mismo tamaño que ytrpred3 con valores de 0 a 100
ytrpred3v=np.linspace(0,100,len(ytrpred3))

# Calcula el error
error=np.mean((ytr-ytrpred)**2)
# Calcula el error cuando la y vale 1
error1=np.mean((ytr[index1]-varphitr[index1].dot(coef[0]))**2)
# Calcula el error cuando la y vale 2
error2=np.mean((ytr[index2]-varphitr[index2].dot(coef[0]))**2)
# Calcula el error cuando la y vale 3
error3=np.mean((ytr[index3]-varphitr[index3].dot(coef[0]))**2)

# Muestra los resultados
print("Error de entrenamiento:",error)
print("Para y=1:",error1)
print("Para y=2:",error2)
print("Para y=3:",error3)
print("Coeficientes:",theta)
print("Residuos:",coef[1])
print("Rango de varphi:",coef[2])
print("valores singulares:",coef[3])

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

# Convierte los dataframes a matrices
varphiv=estimadoresv.values
yv=vdata.values

# Aplica los coeficientes
yvpred=np.dot(varphiv,theta)

# Calcula el error cuando la y vale 1
# Obtiene los puntos donde yv=1
index1=np.where(yv==1)
# Calcula la media del estimado cuando la y vale 1
mean1=np.mean(yvpred[index1])

# Calcula el error cuando la y vale 2
# Obtiene los puntos donde yv=2
index2=np.where(yv==2)
# Calcula la media del estimado cuando la y vale 2
mean2=np.mean(yvpred[index2])

# Calcula el error cuando la y vale 3
# Obtiene los puntos donde yv=3
index3=np.where(yv==3)
# Calcula la media del estimado cuando la y vale 3
mean3=np.mean(yvpred[index3])

# Extrae de yvpred los valores que corresponden a la y=1
yvpred1=yvpred[index1]
# Ordena los valores de menor a mayor
yvpred1=np.sort(yvpred1)
# Crea un vector para el eje vertical con el número de elementos del vector yvpred1 y valores de 0 a 100
yvpred1v=np.linspace(0,100,len(yvpred1))

# Extrae de yvpred los valores que corresponden a la y=2
yvpred2=yvpred[index2]
# Ordena los valores de menor a mayor
yvpred2=np.sort(yvpred2)
# Crea un vector para el eje vertical con el número de elementos del vector yvpred2 y valores de 0 a 100
yvpred2v=np.linspace(0,100,len(yvpred2))

# Extrae de yvpred los valores que corresponden a la y=3
yvpred3=yvpred[index3]
# Ordena los valores de menor a mayor
yvpred3=np.sort(yvpred3)
# Crea un vector para el eje vertical con el número de elementos del vector yvpred3 y valores de 0 a 100
yvpred3v=np.linspace(0,100,len(yvpred3))

# Crea un vector de valores de 0 a 4 con 400 muestras para el selector
v=np.linspace(0,4,400)
# Remuestrea los valores predichos cuando y=1 para el selector
yvpred1int=np.interp(v,ytrpred1,ytrpred1v)
# Remuestrea los valores predichos cuando y=2 para el selector
yvpred2int=np.interp(v,ytrpred2,ytrpred2v)
# Remuestrea los valores predichos cuando y=3 para el selector
yvpred3int=np.interp(v,ytrpred3,ytrpred3v)

# Resta el valor estimado de y=2 al valor estimado de y=1 con la interpolación
dist12=yvpred1int-yvpred2int
# Localiza el punto de máxima diferencia
index12=v[np.argmax(np.abs(dist12))]

# Resta el valor estimado de y=3 al valor estimado de y=2 con la interpolación
dist23=yvpred2int-yvpred3int
# Localiza el punto de máxima diferencia
index23=v[np.argmax(np.abs(dist23))]

# # Calcula el punto medio entre la media de 1 y 2
# index12=(mean1+mean2)/2
# # Calcula el punto medio entre la media de 2 y 3
# index23=(mean2+mean3)/2

# Crea yvsel del tamaño de yvpred y con valor 3
yvsel=np.ones(len(yvpred))*3
# Fija a 2 los valores que están por debajo del punto medio entre 2 y 3
yvsel[yvpred<index23]=2
# Fija a 1 los valores que están por debajo del punto medio entre 1 y 2
yvsel[yvpred<index12]=1

# Calcula el error
error=np.mean((yv-yvpred)**2)
# Calcula el error cuando la y vale 1
error1=np.mean((yv[index1]-yvpred[index1])**2)
# Calcula el error cuando la y vale 2
error2=np.mean((yv[index2]-yvpred[index2])**2)
# Calcula el error cuando la y vale 3
error3=np.mean((yv[index3]-yvpred[index3])**2)

# Calcula el error usando el selector
errorsel=np.mean((yv-yvsel)**2)
# Calcula el error cuando la y vale 1 usando el selector
error1sel=np.mean((yv[index1]-yvsel[index1])**2)
# Calcula el error cuando la y vale 2 usando el selector
error2sel=np.mean((yv[index2]-yvsel[index2])**2)
# Calcula el error cuando la y vale 3 usando el selector
error3sel=np.mean((yv[index3]-yvsel[index3])**2)

# Cuenta las veces que se estima 1 cuando la y es 1
count11=np.sum(yvsel[index1]==1)
# Cuenta las veces que se estima 2 cuando la y es 1
count21=np.sum(yvsel[index1]==2)
# Cuenta las veces que se estima 3 cuando la y es 1
count31=np.sum(yvsel[index1]==3)
# Cuenta las veces que se estima 1 cuando la y es 2
count12=np.sum(yvsel[index2]==1)
# Cuenta las veces que se estima 2 cuando la y es 2
count22=np.sum(yvsel[index2]==2)
# Cuenta las veces que se estima 3 cuando la y es 2
count32=np.sum(yvsel[index2]==3)
# Cuenta las veces que se estima 1 cuando la y es 3
count13=np.sum(yvsel[index3]==1)
# Cuenta las veces que se estima 2 cuando la y es 3
count23=np.sum(yvsel[index3]==2)
# Cuenta las veces que se estima 3 cuando la y es 3
count33=np.sum(yvsel[index3]==3)

# Cuenta las veces que la y es 1
count1=np.sum(yv==1)
# Cuenta las veces que la y es 2
count2=np.sum(yv==2)
# Cuenta las veces que la y es 3
count3=np.sum(yv==3)

# Almacena estos valores en un dataframe
counts=pd.DataFrame({'estima 1':[count11,count12,count13],'estima 2':[count21,count22,count23],'estima 3':[count31,count32,count33],'total':[count1,count2,count3]},index=['vale 1','vale 2','vale 3'])

countpercent=pd.DataFrame({'estima 1':[count11/count1*100,count12/count2*100,count13/count3*100],'estima 2':[count21/count1*100,count22/count2*100,count23/count3*100],'estima 3':[count31/count1*100,count32/count2*100,count33/count3*100]},index=['vale 1','vale 2','vale 3'])

# Calcula la correlación
correlacion=np.corrcoef(yv,yvpred)[0,1]
# Calcula la correlación usando el selector
correlacionsel=np.corrcoef(yv,yvsel)[0,1]

# Muestra los resultados
print("Error de validación:",error)
print("Para y=1:",error1)
print("Para y=2:",error2)
print("Para y=3:",error3)
print("Media de yvpred para y=1:",mean1)
print("Media de yvpred para y=2:",mean2)
print("Media de yvpred para y=3:",mean3)
print("Correlación:",correlacion)
print("Error de validación usando el selector:",errorsel)
print("Para y=1 usando el selector:",error1sel)
print("Para y=2 usando el selector:",error2sel)
print("Para y=3 usando el selector:",error3sel)
print("Correlación usando el selector:",correlacionsel)
print(counts)
print(countpercent)

# Convierte la y estimada a una serie con la forma de vdata
yestim=pd.Series(yvpred,index=vdata.index)
# Convierte la y estimada usando el selector a una serie con la forma de vdata
yestimsel=pd.Series(yvsel,index=vdata.index)

# calcula la matriz de confusion
confusion_matrix = skmetrics.confusion_matrix(vdata, yestimsel)

print(confusion_matrix)


# calcula el porcentaje de acierto
accuracy = skmetrics.balanced_accuracy_score(vdata, yestimsel)
print('Porcentaje de acierto: '+str(accuracy*100)+'%')
sys.exit()

# Convierte la serie a dataframe
yestim=yestim.to_frame()
yestimsel=yestimsel.to_frame()

# Mueve el segundo nivel del dataframe a las columnas
yestim=yestim.unstack()
yestimsel=yestimsel.unstack()

# Grafica los valores de ytrpred frente a sus vectores de ordenamiento
plt.figure()
plt.plot(ytrpred1,ytrpred1v,'r',ytrpred2,ytrpred2v,'g',ytrpred3,ytrpred3v,'b')
# Añade líneas verticales en x=index12, x=index23
plt.axvline(x=index12,color='k')
plt.axvline(x=index23,color='k')
plt.title('Sigmoidal de entrenamiento')
plt.ylabel('Número de datos estimados (%)')
plt.xlabel('Valor de los datos')
plt.legend(['y=1','y=2','y=3'])

# Grafica los valores de yvpred frente a sus vectores de ordenamiento
plt.figure()
plt.plot(yvpred1,yvpred1v,'r',yvpred2,yvpred2v,'g',yvpred3,yvpred3v,'b')
# Añade líneas verticales en x=index12, x=index23
plt.axvline(x=index12,color='k')
plt.axvline(x=index23,color='k')
plt.legend(['y=1','y=2','y=3'])
plt.title('Sigmoidal de validación')
plt.ylabel('Número de datos estimados (%)')
plt.xlabel('Valor de los datos')

# Grafica cada y estimada junto a su y real y al selector frente al tiempo en gráficas separadas
for i in range(len(yestim.columns)):
    plt.figure()
    plt.plot(yestim.index,yestim.iloc[:,i],label='y estimada')
    plt.plot(yestim.index,valdatapd.iloc[:,i],label='y real')
    plt.plot(yestim.index,yestimsel.iloc[:,i],label='y selector')
    plt.legend()
    plt.title(yestim.columns[i][1])
    plt.xlabel('Tiempo')
    plt.ylabel('y')
plt.show()
