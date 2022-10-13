import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import itertools as it
from os.path import exists

# Lista de estimadores a usar
#listest=['LTP diff_1h','LTP diff_R_Neta_Avg_norm','LTP Area_n0','LTP diff_R_Neta_Avg']
#ventanas=[1,10,2,10]
#f_olvido=[0.8,0.95,0.8,0.8]

listest=['LTP diff_1h','LTP diff_R_Neta_Avg_norm','LTP Area_n0','LTP diff_R_Neta_Avg','LTP Min_n0','LTP Max_n0','LTP diff_1h','LTP diff_R_Neta_Avg_norm','LTP Area_n0','LTP diff_R_Neta_Avg','LTP Min_n0','LTP Max_n0']
ventanas=[1,10,2,10,1,1,1,1,1,1,1,1]
f_olvido=[0.8,0.95,0.8,0.8,0.8,0.8,0,0,0,0,0,0]
numero_elementos_max=12

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
    estdata=estimadoresraw.loc[:,listest[i]].copy()
    shiftdata=estimadoresraw.loc[:,listest[i]].copy()
    for j in range(1,ventanas[i]+1):
        shiftdata=shiftdata.shift(1)*f_olvido[i]
        sumdata=shiftdata+estdata
    sumdata.columns = pd.MultiIndex.from_product([sumdata.columns, [listest[i]+","+str(ventanas[i])+","+str(f_olvido[i])]])
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

# Convierte el dataframe de entrenamiento a matriz
ytr=trdata.values

# Cuenta las veces que se repite el valor 3 en ytr
count3=np.count_nonzero(ytr==3)

# Cuenta las veces que se repite el valor 2 en ytr
count2=np.count_nonzero(ytr==2)

# Cuenta las veces que se repite el valor 1 en ytr
count1=np.count_nonzero(ytr==1)



# Comprueba si existe el archivo de resultados
if not exists("resultadosMinimosCuadrados.csv"):
    # Si no existe, crea un archivo con el encabezado de los resultados
    f=open("resultadosMinimosCuadrados.csv","x")
    f.write("Numero de elementos,Estadisticos,Ventanas,Factores de olvido,Valores,Pesos,Error cuadrático en aprendizaje,Error cuadrático en aprendizaje (y=1),Error cuadrático en aprendizaje (y=2),Error cuadrático en aprendizaje (y=3),Error ponderado en aprendizaje,Error cuadrático en validación,Error cuadrático en validación (y=1),Error cuadrático en validación (y=2),Error cuadrático en validación (y=3),Error ponderado en validación,Error cuadrático en validación con saturación,Error cuadrático en validación (y=1) con saturación,Error cuadrático en validación (y=2) con saturación,Error cuadrático en validación (y=3) con saturación,Error ponderado en validación con saturación\n")
else:
    # Si existe, abre el archivo para escribir en él
    f=open("resultadosMinimosCuadrados.csv","a")


# Crea un bucle que recorre todas las posibles combinaciones de columnas de estimadorestr
# Recorre el número de elementos de la combinación
for i in range(1,numero_elementos_max+1):
    # Recorre las combinaciones con i elementos
    for j in it.combinations(estimadorestr.columns,i):
        for weight in [1,0]:
            if weight==1:
                # Crea un vector de pesos sustituyendo cada valor por sum/n, donde n es el número de veces que se repite el valor y sum es el numero total de valores
                sum=count1+count2+count3
                w=np.zeros(len(ytr))
                w[ytr==3]=sum/count3
                w[ytr==2]=sum/count2
                w[ytr==1]=sum/count1
                pesos = [sum/count1,sum/count2,sum/count3]
            else:
                w=np.ones(len(ytr))
                pesos = [1,1,1]
            # Crea tres strings para estimadores, ventanas y factores de olvido
            eststr=""
            winstr=""
            fstr=""
            # separa cada elemento de j por comas
            for k in j:
                jexploded=k.split(",")
                eststr=eststr+jexploded[0]+"  "
                winstr=winstr+jexploded[1]+"  "
                fstr=fstr+jexploded[2]+"  "
            # Añade un elemento más con el término independiente
            eststr=eststr+"independiente"
            winstr=winstr+"1"
            fstr=fstr+"0"

            # Añade el número de elementos al archivo
            resultstr=str(i)+","

            # Añade los resultados al archivo
            resultstr=resultstr+eststr+","+winstr+","+fstr+","

            # Recorta el dataframe de estimadores a las columnas seleccionadas
            estimatr=estimadorestr.loc[:,j]

            # Añade una columna llena de 1 para el término independiente
            estimatr['Uno']=1

            #convierte el dataframe de estimadores a matriz
            varphitr=estimatr.values

            # Aplica el peso a los datos
            varphitrw = varphitr * np.sqrt(w[:,np.newaxis])
            ytrw = ytr * np.sqrt(w)

            # Calcula los coeficientes
            coef=np.linalg.lstsq(varphitrw,ytrw,rcond=None)
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

            # Calcula el error
            sqrerror=np.square(ytr-ytrpred)
            error=np.mean(sqrerror)
            # Calcula el error cuando la y vale 1
            error1=np.mean(sqrerror[index1])
            # Calcula el error cuando la y vale 2
            error2=np.mean(sqrerror[index2])
            # Calcula el error cuando la y vale 3
            error3=np.mean(sqrerror[index3])
            # Calcula la media de los tres errores
            errormed=np.mean([error1,error2,error3])

            # # Elimina la dimensión de valor 1 de los índices
            # index1=np.delete(index1,0)
            # index2=np.delete(index2,0)
            # index3=np.delete(index3,0)
            # # Grafica el sqrerror coloreando cada tramo
            # plt.figure()
            # plt.plot(sqrerror)
            # plt.plot(index1,sqrerror[index1],'r')
            # plt.plot(index2,sqrerror[index2],'b')
            # plt.plot(index3,sqrerror[index3],'g')
            # # Añade una línea horizontal en el error medio
            # plt.axhline(y=error,color='k',linestyle='-')
            # # Añade una línea horizontal en el error cuando la y vale 1
            # plt.axhline(y=error1,color='r',linestyle='-')
            # # Añade una línea horizontal en el error cuando la y vale 2
            # plt.axhline(y=error2,color='b',linestyle='-')
            # # Añade una línea horizontal en el error cuando la y vale 3
            # plt.axhline(y=error3,color='g',linestyle='-')
            # plt.title("Error cuadrático en aprendizaje")
            # plt.xlabel("Muestras")
            # plt.ylabel("Error cuadrático")
            # # Guarda la gráfica con un nombre que dependa de los índices del bucle
            # plt.savefig("graficasErrorAprendizaje\errorCuadraticoAprendizaje"+str(i)+str(j)+str(weight)+".png")
            # plt.close()


            # Añade cada elemento de theta
            for k in range(len(theta)):
                resultstr=resultstr+str(theta[k])
                #añade un espacio al final de cada elemento excepto el último
                if k!=len(theta)-1:
                    resultstr=resultstr+"  "
            # Añade cada elemento de pesos
            resultstr=resultstr+","
            for k in range(len(pesos)):
                resultstr=resultstr+str(pesos[k])
                #añade un espacio al final de cada elemento excepto el último
                if k!=len(pesos)-1:
                    resultstr=resultstr+"  "
            
            # Añade el error cuadrático en aprendizaje
            resultstr=resultstr+","+str(error)+","+str(error1)+","+str(error2)+","+str(error3)+","+str(errormed)

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

            # Recorta los datos de validación a las columnas seleccionadas
            estimadoresv=estimadoresv.loc[:,j]

            # añade una columna llena de 1 para el término independiente
            estimadoresv['Uno']=1

            # Convierte los dataframes a matrices
            varphiv=estimadoresv.values
            yv=vdata.values

            # Aplica los coeficientes
            yvpred=np.dot(varphiv,theta)

            # Obtiene los indices donde la y vale 1
            index1=np.where(yv==1)
            # Obtiene los indices donde la y vale 2
            index2=np.where(yv==2)
            # Obtiene los indices donde la y vale 3
            index3=np.where(yv==3)

            # Calcula el error
            sqrerror=np.square(yv-yvpred)
            error=np.mean(sqrerror)
            # Calcula el error cuando la y vale 1
            error1=np.mean(sqrerror[index1])
            # Calcula el error cuando la y vale 2
            error2=np.mean(sqrerror[index2])
            # Calcula el error cuando la y vale 3
            error3=np.mean(sqrerror[index3])
            # Calcula la media entre los tres errores
            errormed=np.mean([error1,error2,error3])

            # # Elimina la dimensión de valor 1 de los índices
            # index1=np.delete(index1,0)
            # index2=np.delete(index2,0)
            # index3=np.delete(index3,0)

            # # Grafica el sqrerror coloreando cada tramo
            # plt.figure()
            # plt.plot(sqrerror)
            # plt.plot(index1,sqrerror[index1],'r')
            # plt.plot(index2,sqrerror[index2],'b')
            # plt.plot(index3,sqrerror[index3],'g')
            # # Añade una línea horizontal en el error medio
            # plt.axhline(y=error,color='k',linestyle='-')
            # # Añade una línea horizontal en el error cuando la y vale 1
            # plt.axhline(y=error1,color='r',linestyle='-')
            # # Añade una línea horizontal en el error cuando la y vale 2
            # plt.axhline(y=error2,color='b',linestyle='-')
            # # Añade una línea horizontal en el error cuando la y vale 3
            # plt.axhline(y=error3,color='g',linestyle='-')
            # plt.title("Error cuadrático en validación")
            # plt.xlabel("Muestras")
            # plt.ylabel("Error cuadrático")
            # # Guarda la gráfica con un nombre que dependa de los índices del bucle
            # plt.savefig("graficasErrorValidacion\errorCuadraticoValidacion"+str(i)+str(j)+str(weight)+".png")
            # plt.close()


            # Añade los resultados al string de resultados
            resultstr=resultstr+","+str(error)+","+str(error1)+","+str(error2)+","+str(error3)+","+str(errormed)

            # Satura yvpred entre 1 y 3
            yvpred[yvpred<1]=1
            yvpred[yvpred>3]=3

            # Calcula el error
            sqrerror=np.square(yv-yvpred)
            error=np.mean(sqrerror)

            # Calcula el error cuando la y vale 1
            error1=np.mean(sqrerror[index1])

            # Calcula el error cuando la y vale 2
            error2=np.mean(sqrerror[index2])

            # Calcula el error cuando la y vale 3
            error3=np.mean(sqrerror[index3])

            # Calcula la media entre los tres errores
            errormed=np.mean([error1,error2,error3])

            # # Grafica el sqrerror coloreando cada tramo
            # plt.figure()
            # plt.plot(sqrerror)
            # plt.plot(index1,sqrerror[index1],'r')
            # plt.plot(index2,sqrerror[index2],'b')
            # plt.plot(index3,sqrerror[index3],'g')
            # # Añade una línea horizontal en el error medio
            # plt.axhline(y=error,color='k',linestyle='-')
            # # Añade una línea horizontal en el error cuando la y vale 1
            # plt.axhline(y=error1,color='r',linestyle='-')
            # # Añade una línea horizontal en el error cuando la y vale 2
            # plt.axhline(y=error2,color='b',linestyle='-')
            # # Añade una línea horizontal en el error cuando la y vale 3
            # plt.axhline(y=error3,color='g',linestyle='-')
            # plt.title("Error cuadrático en validación")
            # plt.xlabel("Muestras")
            # plt.ylabel("Error cuadrático")
            # # Guarda la gráfica con un nombre que dependa de los índices del bucle
            # plt.savefig("graficasErrorValidacionSat\errorCuadraticoValidacionSat"+str(i)+str(j)+str(weight)+".png")
            # plt.close()

            # Añade los resultados al string de resultados
            resultstr=resultstr+","+str(error)+","+str(error1)+","+str(error2)+","+str(error3)+","+str(errormed)

            # Añade el string de resultados al archivo
            f.write(resultstr+"\n")
# Cierra el archivo
f.close()
