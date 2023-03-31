import sys
from matplotlib.markers import MarkerStyle
import matplotlib
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import time
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp
import isadoralib as isl
import time as tm
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})

year_train="2014"
year_datas=["2014","2015","2016","2019"]
sufix="rht"

# maxtdvitems=120
# maxmeteoitems=120
# mintdvitems=3
# minmeteoitems=3
# steptdv=39
# stepmeteo=39

# tdvlist=[*range(mintdvitems,maxtdvitems,steptdv)]
# meteolist=[*range(minmeteoitems,maxmeteoitems,stepmeteo)]
#Haz con tdv de [10:3:50] por ejemplo con PCA de [10:3:50] y meteo [0:1:9] Son 1000 datos más
tdvitems=80
meteoitems=4
comp=13 #LDA 

n_dias_print=5
#matplotlib.use("Agg")

res=pd.DataFrame()
times=pd.DataFrame()
#pd.set_option('display.max_rows', None)

for year_data in year_datas:
    timepreprostart=tm.time()
    print(year_data)
    saveFolder="ignore/figures/PCALDAMETEOresults/"+year_train+"-"+year_data+"-"+sufix+"/"
    # Carga de datos de entrenamiento
    tdvT,ltpT,meteoT,trdatapd=isl.cargaDatosTDV(year_train,sufix)

    # Carga de datos de predicción
    tdvP,ltpP,meteoP,valdatapd=isl.cargaDatosTDV(year_data,sufix)

    # guarda la información raw para plots
    tdvPlot = tdvP.copy()
    meteoPlot = meteoP.copy()

    # añade meteo a tdvT
    tdvT=tdvT.join(meteoT)
    # elimina los valores NaN de tdv
    tdvT = tdvT.dropna(axis=1,how='all')

    # añade meteo a tdvP
    tdvP=tdvP.join(meteoP)

    # elimina los valores NaN de tdv
    tdvP = tdvP.dropna(axis=1,how='all')
    # rellena los valores NaN de tdv con el valor anterior
    tdvP = tdvP.fillna(method='ffill')
    tdvT = tdvT.fillna(method='ffill')
    # rellena los valores NaN de tdv con el valor siguiente
    tdvP = tdvP.fillna(method='bfill')
    tdvT = tdvT.fillna(method='bfill')

    # aplica un filtro de media móvil a tdv
    tdvP = tdvP.rolling(window=240,center=True).mean()
    tdvT = tdvT.rolling(window=240,center=True).mean()

    # calcula el valor medio de tdv para cada dia
    tdv_medioP = tdvP.groupby(tdvP.index.date).mean()
    tdv_medioT = tdvT.groupby(tdvT.index.date).mean()

    # calcula el valor de la desviación estándar de tdv para cada dia
    tdv_stdP = tdvP.groupby(tdvP.index.date).std()
    tdv_stdT = tdvT.groupby(tdvT.index.date).std()

    # cambia el índice a datetime
    tdv_medioP.index = pd.to_datetime(tdv_medioP.index)
    tdv_medioT.index = pd.to_datetime(tdv_medioT.index)
    tdv_stdP.index = pd.to_datetime(tdv_stdP.index)
    tdv_stdT.index = pd.to_datetime(tdv_stdT.index)

    # remuestrea tdv_medio y tdv_std a minutal
    tdv_medioP = tdv_medioP.resample('T').pad()
    tdv_medioT = tdv_medioT.resample('T').pad()
    tdv_stdP = tdv_stdP.resample('T').pad()
    tdv_stdT = tdv_stdT.resample('T').pad()

    # normaliza tdv para cada dia

    tdvP = (tdvP - tdv_medioP) / tdv_stdP
    tdvT = (tdvT - tdv_medioT) / tdv_stdT

    # obtiene todos los cambios de signo de R_Neta_Avg en el dataframe meteo
    signosP = np.sign(meteoP.loc[:,meteoP.columns.str.startswith('R_Neta_Avg')]).diff()
    signosT = np.sign(meteoT.loc[:,meteoT.columns.str.startswith('R_Neta_Avg')]).diff()
    # obtiene los cambios de signo de positivo a negativo
    signos_pnP = signosP<0
    signos_pnT = signosT<0
    # elimina los valores falsos (que no sean cambios de signo)
    signos_pnP = signos_pnP.replace(False,np.nan).dropna()
    signos_pnT = signos_pnT.replace(False,np.nan).dropna()
    # obtiene los cambios de signo de negativo a positivo
    signos_npP = signosP>0
    signos_npT = signosT>0
    # elimina los valores falsos (que no sean cambios de signo)
    signos_npP = signos_npP.replace(False,np.nan).dropna()
    signos_npT = signos_npT.replace(False,np.nan).dropna()

    # duplica el índice de signos np como una columna más en signos_np
    signos_npP['Hora'] = signos_npP.index
    signos_npT['Hora'] = signos_npT.index
    # recorta signos np al primer valor de cada día
    signos_npP = signos_npP.resample('D').first()
    signos_npT = signos_npT.resample('D').first()

    #elimina los dias en los que no haya cambio de signo
    signos_npP=signos_npP.dropna()
    signos_npT=signos_npT.dropna()

    # duplica el índice de signos pn como una columna más en signos_pn
    signos_pnP['Hora'] = signos_pnP.index
    signos_pnT['Hora'] = signos_pnT.index
    # recorta signos pn al último valor de cada día
    signos_pnP = signos_pnP.resample('D').last()
    signos_pnT = signos_pnT.resample('D').last()

    #elimina los días en los que no haya cambio de signo
    signos_pnP = signos_pnP.dropna()
    signos_pnT = signos_pnT.dropna()

    # recoge los valores del índice de tdv donde la hora es 00:00
    tdv_00P = tdvP.index.time == time.min
    tdv_00T = tdvT.index.time == time.min
    # recoge los valores del índice de tdv donde la hora es la mayor de cada día
    tdv_23P = tdvP.index.time == time(23,59)
    tdv_23T = tdvT.index.time == time(23,59)

    # crea una columna en tdv que vale 0 a las 00:00
    tdvP.loc[tdv_00P,'Hora_norm'] = 0
    tdvT.loc[tdv_00T,'Hora_norm'] = 0
    # iguala Hora_norm a 6 en los índices de signos np
    tdvP.loc[signos_npP['Hora'],'Hora_norm'] = 6
    tdvT.loc[signos_npT['Hora'],'Hora_norm'] = 6
    # iguala Hora_norm a 18 en los índices de signos pn
    tdvP.loc[signos_pnP['Hora'],'Hora_norm'] = 18
    tdvT.loc[signos_pnT['Hora'],'Hora_norm'] = 18
    # iguala Hora_norm a 24 en el último valor de cada día
    tdvP.loc[tdv_23P,'Hora_norm'] = 24
    tdvT.loc[tdv_23T,'Hora_norm'] = 24
    # iguala el valor en la última fila de Hora_norm a 24
    tdvP.loc[tdvP.index[-1],'Hora_norm'] = 24
    tdvT.loc[tdvT.index[-1],'Hora_norm'] = 24
    # interpola Hora_norm en tdv
    tdvP.loc[:,'Hora_norm'] = tdvP.loc[:,'Hora_norm'].interpolate()
    tdvT.loc[:,'Hora_norm'] = tdvT.loc[:,'Hora_norm'].interpolate()

    # almacena los valores antes de recortar
    tdvPBase=tdvP

    # recorta tdv a los tramos de 6 a 18 de hora_norm
    tdvP = tdvP.loc[tdvP['Hora_norm']>=6,:]
    tdvT = tdvT.loc[tdvT['Hora_norm']>=6,:]
    tdvP = tdvP.loc[tdvP['Hora_norm']<=18,:]
    tdvT = tdvT.loc[tdvT['Hora_norm']<=18,:]


    # añade la hora normalizada al índice de tdv
    tdvP.index = [tdvP.index.strftime('%Y-%m-%d'),tdvP['Hora_norm']]
    tdvT.index = [tdvT.index.strftime('%Y-%m-%d'),tdvT['Hora_norm']]

    #crea el índice de tdvPBase
    tdvPBase['Hora_norm']=tdvPBase['Hora_norm'].apply(pd.to_timedelta,unit='H')
    tdvPBase['dia_norm'] = tdvPBase.index.strftime('%Y-%m-%d')
    tdvPBase.index = [tdvPBase['dia_norm'].apply(pd.to_datetime,format='%Y-%m-%d'),tdvPBase['Hora_norm']]
    tdvPBase=tdvPBase.drop('Hora_norm',axis=1)
    tdvPBase=tdvPBase.drop('dia_norm',axis=1)
    tdvPBase=tdvPBase.unstack(level=0)

    valdatapd.index = valdatapd.index.strftime('%Y-%m-%d')
    trdatapd.index = trdatapd.index.strftime('%Y-%m-%d')

    #obtiene el índice interseccion de valdatapd y el primer nivel del índice de tdv
    tdvPdates = tdvP.index.get_level_values(0)
    tdvTdates = tdvT.index.get_level_values(0)

    valdatapd_tdv = valdatapd.index.intersection(tdvPdates)
    trdatapd_tdv = trdatapd.index.intersection(tdvTdates)

    # vuelve a separar los valores de meteo de tdv
    meteoP_norm=tdvP.drop(tdvP.columns[tdvP.columns.str.startswith('tdv')], axis=1)
    meteoT_norm=tdvT.drop(tdvT.columns[tdvT.columns.str.startswith('tdv')], axis=1)

    # elimina los valores de tdv que no estén en valdatapd
    tdvv = tdvP.loc[valdatapd_tdv,valdatapd.columns]
    # elimina los valores de tdv que no estén en trdatapd
    tdvt = tdvT.loc[trdatapd_tdv,trdatapd.columns]

    # unstackea meteoP_norm y meteoT_norm
    meteoP_norm = meteoP_norm.unstack(level=0)
    meteoT_norm = meteoT_norm.unstack(level=0)

    # unstackea tdvv
    tdvv = tdvv.unstack(level=0)
    # unstackea tdvt
    tdvt = tdvt.unstack(level=0)

    # crea un índice para ajustar frecuencias
    tdvv_index_float=pd.Int64Index(np.floor(tdvv.index*1000000000))
    tdvt_index_float=pd.Int64Index(np.floor(tdvt.index*1000000000))
    meteoP_index_float=pd.Int64Index(np.floor(meteoP_norm.index*1000000000))
    meteoT_index_float=pd.Int64Index(np.floor(meteoT_norm.index*1000000000))

    tdvv.index = pd.to_datetime(tdvv_index_float)
    tdvt.index = pd.to_datetime(tdvt_index_float)
    meteoP_norm.index = pd.to_datetime(meteoP_index_float)
    meteoT_norm.index = pd.to_datetime(meteoT_index_float)

    tdvv_orig=tdvv.copy()
    tdvt_orig=tdvt.copy()
    meteoP_norm_orig=meteoP_norm.copy()
    meteoT_norm_orig=meteoT_norm.copy()

    timepreproend=tm.time()
    preprotime=timepreproend-timepreprostart
    print('tdvitems:'+str(tdvitems))
    print('meteoitems:'+str(meteoitems))
    timeloopstart=tm.time()
    ftdv=12/tdvitems
    if meteoitems>0:
        fmeteo=12/meteoitems
    else:
        fmeteo=0
    # convierte el indice a datetime para ajustar frecuencias
    tdvv=tdvv_orig.resample(str(int(ftdv*1000))+'L').mean()
    tdvt=tdvt_orig.resample(str(int(ftdv*1000))+'L').mean()
    if meteoitems>0:
        meteoP_norm=meteoP_norm_orig.resample(str(int(fmeteo*1000))+'L').mean()
        meteoT_norm=meteoT_norm_orig.resample(str(int(fmeteo*1000))+'L').mean()

    # conserva los valores de 1970-01-01 00:00:06.000 a 1970-01-01 00:00:17.900
    tdvv = tdvv.loc[tdvv.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
    tdvt = tdvt.loc[tdvt.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
    tdvv = tdvv.loc[tdvv.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]
    tdvt = tdvt.loc[tdvt.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]

    if meteoitems>0:
        meteoP_norm = meteoP_norm.loc[meteoP_norm.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
        meteoP_norm = meteoP_norm.loc[meteoP_norm.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]
        meteoT_norm = meteoT_norm.loc[meteoT_norm.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
        meteoT_norm = meteoT_norm.loc[meteoT_norm.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]


    # Crea una serie para restaurar el índice
    norm_index=pd.Series(np.arange(6,18,ftdv))
    #recorta norm_index para que coincida con el tamano de tdvt si se ha producido un desajuste al calcular el dataframe
    norm_index=norm_index.loc[norm_index.index<len(tdvt)]
    # Ajusta el índice de tdvv a la serie
    tdvv.index=norm_index
    # Ajusta el índice de tdvt a la serie
    tdvt.index=norm_index

    if meteoitems>0:
        # Crea una serie para restaurar el índice
        norm_index=pd.Series(np.arange(6,18,fmeteo))
        #recorta norm_index para que coincida con el tamano de meteoP_norm si se ha producido un desajuste al calcular el dataframe
        norm_index=norm_index.loc[norm_index.index<len(meteoT_norm)]
        # Ajusta el índice de meteoP_norm a la serie
        meteoP_norm.index=norm_index
        # Ajusta el índice de meteoT_norm a la serie
        meteoT_norm.index=norm_index

        # dropea la columna Hora_norm de meteo
        meteoP_norm = meteoP_norm.drop('Hora_norm',axis=1)
        meteoT_norm = meteoT_norm.drop('Hora_norm',axis=1)

        # stackea meteoP_norm y meteoT_norm
        meteoP_norm = meteoP_norm.stack(level=0)
        meteoT_norm = meteoT_norm.stack(level=0)

        #intercambia los niveles del índice de meteo
        meteoP_norm.index = meteoP_norm.index.swaplevel(0,1)
        meteoT_norm.index = meteoT_norm.index.swaplevel(0,1)

        meteoP_norm=meteoP_norm.dropna(axis=1,how='all')
        meteoT_norm=meteoT_norm.dropna(axis=1,how='all')

        #combina los dos índices de meteo
        meteoP_norm.index = meteoP_norm.index.map('{0[1]}/{0[0]}'.format)
        meteoT_norm.index = meteoT_norm.index.map('{0[1]}/{0[0]}'.format)

        #elimina los indices no comunes de meteo
        meteoP_norm = meteoP_norm.loc[meteoP_norm.index.isin(meteoT_norm.index)]
        meteoT_norm = meteoT_norm.loc[meteoT_norm.index.isin(meteoP_norm.index)]
    else:
        meteoP_norm = pd.DataFrame()
        meteoT_norm = pd.DataFrame()

    #crea un array de numpy en blanco
    array_tdvv=np.empty((len(tdvv)+len(meteoP_norm),0))
    array_tdvt=np.empty((len(tdvt)+len(meteoT_norm),0))

    #por cada elemento en el primer índice de columnas de tdv
    for i in tdvv.columns.levels[0]:
        tdvv_col=tdvv.loc[:,i]
        if meteoitems>0:
            # elimina los valores de meteo que no estén en tdv_col
            meteo_tdv = tdvv_col.columns.intersection(meteoP_norm.columns)
            meteoP_col = meteoP_norm.loc[:,meteo_tdv]

            # combina los valores de tdvv con los de meteo
            merge_tdv_meteo = pd.merge(tdvv.loc[:,i],meteoP_col,how='outer')
        else:
            merge_tdv_meteo = tdvv.loc[:,i]
        # añade la unión al array de numpy
        array_tdvv=np.append(array_tdvv,merge_tdv_meteo.values,axis=1)

    #por cada elemento en el primer índice de columnas de tdv
    for i in tdvt.columns.levels[0]:
        tdvt_col=tdvt.loc[:,i]
        if meteoitems>0:
            # elimina los valores de meteo que no estén en tdv_col
            meteo_tdv = tdvt_col.columns.intersection(meteoT_norm.columns)
            meteoT_col = meteoT_norm.loc[:,meteo_tdv]

            # combina los valores de tdvv con los de meteo
            merge_tdv_meteo = pd.merge(tdvt.loc[:,i],meteoT_col,how='outer')
        else:
            merge_tdv_meteo = tdvt.loc[:,i]
        # añade la unión al array de numpy
        array_tdvt=np.append(array_tdvt,merge_tdv_meteo.values,axis=1)
    # print("tdvt")
    # print(tdvt)
    # print("meteo_tdv")
    # print(meteo_tdv)
    # print("meteoT_norm")
    # print(meteoT_norm)
    # print("tdvt_col")
    # print(tdvt_col)

    # crea los valores X e y para el modelo
    Xtr=array_tdvt.transpose()
    Ytr=trdatapd.unstack().values
    Xv=array_tdvv.transpose()
    Yv=valdatapd.unstack().values

    #print(np.shape(Xtr))
    #print(np.shape(Xv))

    # elimina los valores NaN de Xtr y Xv
    XtrBase = np.nan_to_num(Xtr)
    XvBase = np.nan_to_num(Xv)
    timeloopend=tm.time()
    looptime=timeloopend-timeloopstart
    timeclasstart=tm.time()
    #print('comp:'+str(comp))
    # crea un array de tamaño MaxComp
    #aplica PCA
    pca = skdecomp.PCA(n_components=comp)
    pca.fit(XtrBase)
    Xtr = pca.transform(XtrBase)
    Xv = pca.transform(XvBase)

    # crea el modelo
    clf = sklda.LinearDiscriminantAnalysis(solver='svd')
    # entrena el modelo
    clf.fit(Xtr,Ytr)
    # predice los valores de Yv
    Ypred=clf.predict(Xv)
    # # plotea Yv y Ypred
    # fig, ax = plt.subplots()
    # plt.plot(Ypred, color="#C22F00", marker='+')
    # plt.plot(Yv, color="#4E94EC", marker='x')
    # plt.legend(["Automatic classification","Manual classification"])
    # #plt.grid()
    # plt.xlabel('Sample number')
    # plt.ylabel('Hydric stress level')
    #fig.savefig('ignore/resultadosPCALDA/'+year_data+'.png')
    #plt.show()

    # predice las probabilidades de Yv
    Yprob=clf.predict_proba(Xv)

    # calcula la matriz de confusion

    # confusion_matrix = skmetrics.confusion_matrix(Yv, Ypred)
    # print(confusion_matrix)
    bcm=skmetrics.confusion_matrix(Yv, Ypred)
    print(bcm)
    bcm=skmetrics.confusion_matrix(Yv, Ypred,normalize='true')
    print(bcm)

    # calcula el porcentaje de acierto
    accuracy = skmetrics.balanced_accuracy_score(Yv, Ypred)
    
    res=res.append({'tdv samples':tdvitems,'meteo samples':meteoitems,'year data':year_data,'components from PCA':comp,'fraction from total PCA':np.around(comp/XtrBase.shape[1],1),'accuracy':accuracy},ignore_index=True)
    # res=res.append({'tdv samples':tdvitems,'meteo samples':meteoitems,'year train':year_train,'year data':year_data,'components from PCA':comp,'accuracy':accuracy,'confusion matrix':np.around(bcm,2)},ignore_index=True)
    # res=res.append({'tdv':tdvitems,'meteo':meteoitems,'year train':year_train,'year data':year_data,'comp':comp,'acc':accuracy,'cmatrix':"\\begin{tabular}{ ccc }"+(" \\\\\n".join([" & ".join(map(str,line)) for line in bcm]))+"\\end{tabular}"},ignore_index=True)
    
    timeclasend=tm.time()
    clastime=timeclasend-timeclasstart
    print('Porcentaje de acierto: '+str(accuracy*100)+'%')
    print('frac PCA: '+str(np.around(comp/XtrBase.shape[1],1)))

    times=times.append({'tdv samples':tdvitems,'meteo samples':meteoitems,'year data':year_data,'components from PCA':comp,'fraction from total PCA':np.around(comp/XtrBase.shape[1],1),'preprocess':preprotime,'process':looptime,'classifier':clastime,'total':preprotime+looptime+clastime},ignore_index=True)
res.set_index(['year data','tdv samples','meteo samples','components from PCA','fraction from total PCA'], inplace=True) 
res=res.unstack(level=0)

times.set_index(['year data','tdv samples','meteo samples','components from PCA','fraction from total PCA'], inplace=True) 
times=times.unstack(level=0)

mean=res.mean(numeric_only=True, axis=1)
var=res.var(numeric_only=True, axis=1)
res['total accuracy']=mean
res['accuracy variation']=var
#res=res.sort_values(['total accuracy'], ascending=False)
print(res)
#res.to_csv('ignore/analisisPCALDA/resultadosPCALDAMeteo.csv')
#times.to_csv('ignore/analisisPCALDA/tiemposPCALDAMeteo.csv')
