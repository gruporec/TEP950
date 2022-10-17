import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import itertools as it
from os.path import exists

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
    sumdata=estimadoresraw.loc[:,listest[i]].copy()
    shiftdata=estimadoresraw.loc[:,listest[i]].copy()
    for j in range(1,ventanas[i]+1):
        shiftdata=shiftdata.shift(1)*f_olvido[i]
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

# Convierte el dataframe de entrenamiento a matriz
ytr=trdata.values

# Cuenta las veces que se repite el valor 3 en ytr
count3=np.count_nonzero(ytr==3)

# Cuenta las veces que se repite el valor 2 en ytr
count2=np.count_nonzero(ytr==2)

# Cuenta las veces que se repite el valor 1 en ytr
count1=np.count_nonzero(ytr==1)

# Crea un string para la lista de estimadores
estimadores_str=''
for i in range(len(listest)):
    estimadores_str=estimadores_str+listest[i]
    if f_olvido[i]==0:
        estimadores_str=estimadores_str+" raw"
    estimadores_str=estimadores_str+","


# Comprueba si existe el archivo de resultados
if not exists("resultadosMinimosCuadradosClasificador2.csv"):
    # Si no existe, crea un archivo con el encabezado de los resultados
    f=open("resultadosMinimosCuadradosClasificador2.csv","x")
    f.write("Numero de elementos,LTP Area_n0 raw,LTP Area_n0,LTP Max_n0 raw,LTP Max_n0,LTP Min_n0 raw,LTP Min_n0,LTP diff_1h raw,LTP diff_1h,LTP diff_R_Neta_Avg raw,LTP diff_R_Neta_Avg,LTP diff_R_Neta_Avg_norm raw,LTP diff_R_Neta_Avg_norm,Estadisticos,Ventanas,Factores de olvido,Valores,Pesos,Error cuadrático en aprendizaje,Error cuadrático en aprendizaje (y=1),Error cuadrático en aprendizaje (y=2),Error cuadrático en aprendizaje (y=3),Error ponderado en aprendizaje,Error cuadrático en validación,Error cuadrático en validación (y=1),Error cuadrático en validación (y=2),Error cuadrático en validación (y=3),Error ponderado en validación,Error cuadrático en validación con saturación,Error cuadrático en validación (y=1) con saturación,Error cuadrático en validación (y=2) con saturación,Error cuadrático en validación (y=3) con saturación,Error ponderado en validación con saturación,Acierto del selector(tanto por uno),Acierto en estrés 1, Acierto en estrés 2, Acierto en estrés 3,Valores del selector\n")
else:
    # Si existe, abre el archivo para escribir en él
    f=open("resultadosMinimosCuadradosClasificador2.csv","a")


# Crea un bucle que recorre todas las posibles combinaciones de columnas de estimadorestr
# Recorre el número de elementos de la combinación
for i in range(1,numero_elementos_max+1):
    # Recorre las combinaciones con i elementos
    for jj in it.combinations(range(len(estimadorestr.columns)),i):

        # crea un array vacío
        j=np.array([])

        # crea un array de ceros con el número de elementos de las columnas de estimadores
        binarray=np.zeros(len(estimadorestr.columns))

        # fija binarray a tipo entero
        binarray=binarray.astype(int)

        # añade a j las columnas de estimadorestr que se encuentran en la tupla jj y pone a 1 el elemento correspondiente en binarray
        for k in range(len(jj)):
            j=np.append(j,estimadorestr.columns[jj[k]])
            binarray[jj[k]]=1
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

            # Añade los elementos de binarray separados por comas al archivo
            resultstr=resultstr+",".join(map(str,binarray))+","
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

            # Entrena el selector
            # satura la y predicha entre 1 y 3
            ytrpred=np.clip(ytrpred,1,3)
            # Separa la y estimada cuando la y vale 1
            ytrpred1=ytrpred[index1]
            # Separa la y estimada cuando la y vale 2
            ytrpred2=ytrpred[index2]
            # Separa la y estimada cuando la y vale 3
            ytrpred3=ytrpred[index3]

            # Ordena los vectores de menor a mayor
            ytrpred1=np.sort(ytrpred1)
            ytrpred2=np.sort(ytrpred2)
            ytrpred3=np.sort(ytrpred3)

            # crea un vector de tamaño igual a ytrpred1 y valor de 0 a 100
            ntrpred1=np.linspace(0,100,len(ytrpred1))
            ntrpred2=np.linspace(0,100,len(ytrpred2))
            ntrpred3=np.linspace(0,100,len(ytrpred3))

            # añade un elemento de valor 1 al inicio de las ytr predichas
            ytrpred1=np.insert(ytrpred1,0,1)
            ytrpred2=np.insert(ytrpred2,0,1)
            ytrpred3=np.insert(ytrpred3,0,1)

            # añade un elemento de valor 3 al final de las ytr predichas
            ytrpred1=np.append(ytrpred1,3)
            ytrpred2=np.append(ytrpred2,3)
            ytrpred3=np.append(ytrpred3,3)

            # añade un elemento de valor 0 al inicio de los ntrpred1
            ntrpred1=np.insert(ntrpred1,0,0)
            ntrpred2=np.insert(ntrpred2,0,0)
            ntrpred3=np.insert(ntrpred3,0,0)

            # añade un elemento de valor 100 al final de los ntrpred1
            ntrpred1=np.append(ntrpred1,100)
            ntrpred2=np.append(ntrpred2,100)
            ntrpred3=np.append(ntrpred3,100)

            # crea un vector de 1 a 3 con un paso de 0.001
            stepv=np.linspace(1,3,int(3/0.001)+1)

            # interpola las y con el vector stepv
            ntrpres1=np.interp(stepv,ytrpred1,ntrpred1)
            ntrpres2=np.interp(stepv,ytrpred2,ntrpred2)
            ntrpres3=np.interp(stepv,ytrpred3,ntrpred3)

            # encuentra el índice del máximo de la distancia entre ntrpres1 y ntrpres2
            index12=np.argmax(ntrpres1-ntrpres2)
            # encuentra el índice del máximo de la distancia entre ntrpres2 y ntrpres3
            index23=np.argmax(ntrpres2-ntrpres3)

            # encuentra el valor de stepv que corresponde al máximo de la distancia entre ntrpres1 y ntrpres2
            stepv12=stepv[index12]
            # encuentra el valor de stepv que corresponde al máximo de la distancia entre ntrpres2 y ntrpres3
            stepv23=stepv[index23]

            # # crea un vector de salida del selector con el tamaño de ytrpred y valor de 3
            # ytrpredsel=np.full(len(ytrpred),3)
            # # fija el valor del vector de salida del selector a 2 cuando el valor de ytrpred es menor que stepv23
            # ytrpredsel[ytrpred<stepv23]=2
            # # fija el valor del vector de salida del selector a 1 cuando el valor de ytrpred es menor que stepv12
            # ytrpredsel[ytrpred<stepv12]=1

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

            # Añade los resultados al string de resultados
            resultstr=resultstr+","+str(error)+","+str(error1)+","+str(error2)+","+str(error3)+","+str(errormed)

            # Aplica un clasificador sobre los datos de validación
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

            # aplica el selector
            # crea un vector de salida del selector con el tamaño de yvpred y valor de 3
            yvpredsel=np.full(len(yvpred),3)
            # fija el valor del vector de salida del selector a 2 cuando el valor de yvpred es menor que stepv23
            yvpredsel[yvpred<stepv23]=2
            # fija el valor del vector de salida del selector a 1 cuando el valor de yvpred es menor que stepv12
            yvpredsel[yvpred<stepv12]=1

            # calcula el porcentaje de acierto cuando la y vale 1
            acierto1=np.mean(yvpredsel[index1]==yv[index1])
            # calcula el porcentaje de acierto cuando la y vale 2
            acierto2=np.mean(yvpredsel[index2]==yv[index2])
            # calcula el porcentaje de acierto cuando la y vale 3
            acierto3=np.mean(yvpredsel[index3]==yv[index3])
            # calcula el porcentaje de acierto medio
            acierto=np.mean([acierto1,acierto2,acierto3])

            # Añade los resultados al string de resultados
            resultstr=resultstr+","+str(acierto)+","+str(acierto1)+","+str(acierto2)+","+str(acierto3)
            #añade los valores de stepv12 y stepv23 al string de resultados
            resultstr=resultstr+","+str(stepv12)+"  "+str(stepv23)
            # Añade el string de resultados al archivo
            f.write(resultstr+"\n")
# Cierra el archivo
f.close()
