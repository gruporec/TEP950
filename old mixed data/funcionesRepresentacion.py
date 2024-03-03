import pandas as pd
import plotly.express as px
import numpy as np

def grafica(csv,idsenal,Tini=0,Tfin=0,Tmuestreo=0,MargenMuestreo=pd.to_timedelta("1D"),plot=True,save=False):
    '''Genera una gráfica con los datos de una señal extraida de un archivo csv. Requiere el nombre del archivo csv y
    la etiqueta de la señal idsenal. Puede especificarse además el instante inicial Tini, el instante final Tfin, el tiempo
    de muestreo Tmuestreo (se utilizará, en su caso, interpolación lineal) y un margen de muestreo MargenMuestreo para asegurar 
    que se toman suficientes datos en el borde para el muestreo.'''
    df=pd.read_csv(csv)
    senal=df[["Fecha",idsenal]]
    senal.iloc[:,0]=pd.to_datetime(senal.iloc[:,0]) # Fecha como datetime
    senal=senal.set_index(["Fecha"])

    senal=timecrop(senal,Tini,Tfin,MargenMuestreo)
    if Tmuestreo != 0 :
        senal=senal.resample(rule=Tmuestreo).interpolate("linear")
    senal=timecrop(senal,Tini,Tfin)
    fig = px.line(senal, title=idsenal) 
    if plot: fig.show()
    if save: fig.write_html(idsenal+".html")
    return senal

def compara(csvx,idsenalx,csvy,idsenaly,Tini=0,Tfin=0,Tmuestreo=0,MargenMuestreo=pd.to_timedelta("1D"),plot=True,plot3D=False,save=False,rolx=1,roly=1):
    '''Genera una gráfica con los datos de dos señales extraidas de archivos csv. Requiere el nombre de cada archivo csv y
    la etiqueta de cada señal. Puede especificarse además el instante inicial Tini, el instante final Tfin, el tiempo
    de muestreo Tmuestreo (se utilizará, en su caso, interpolación lineal) y un margen de muestreo MargenMuestreo para asegurar 
    que se toman suficientes datos en el borde para el muestreo.'''
    df=pd.read_csv(csvx)
    df2=pd.read_csv(csvy)

    senal=df[["Fecha",idsenalx]]
    senal.iloc[:,0]=pd.to_datetime(senal.iloc[:,0]) # Fecha como datetime
    senal=senal.set_index(["Fecha"])
    senal=timecrop(senal,Tini,Tfin,MargenMuestreo)

    senal2=df2[["Fecha",idsenaly]]
    senal2.iloc[:,0]=pd.to_datetime(senal2.iloc[:,0]) # Fecha como datetime
    senal2=senal2.set_index(["Fecha"])
    senal2=timecrop(senal2,Tini,Tfin,MargenMuestreo)

    if Tmuestreo != 0 :
        senal=senal.resample(rule=Tmuestreo).interpolate("linear")
    if Tmuestreo != 0 :
        senal2=senal2.resample(rule=Tmuestreo).interpolate("linear")
    
    senal=timecrop(senal,Tini,Tfin)
    senal2=timecrop(senal2,Tini,Tfin)
    senal=senal.rolling(rolx).mean()
    senal2=senal2.rolling(roly).mean()
    print(senal)
    print(senal2)
    senal=senal.merge(senal2,left_index=True,right_index=True).dropna()
    print(senal)
    fig = px.scatter(x=senal.iloc[:,0],y=senal.iloc[:,1], title=idsenalx+" vs "+idsenaly, trendline="ols") 
    if plot: fig.show()
    if save: fig.write_html(idsenalx+" vs "+idsenaly+".html")
    fig = px.scatter_3d(x=senal.index,y=senal.iloc[:,0],z=senal.iloc[:,1], title=idsenalx+" vs "+idsenaly) 
    if plot3D: fig.show()
    if save: fig.write_html(idsenalx+" vs "+idsenaly+" 3D"+".html")
    return senal

def timecrop(df,Tini=0,Tfin=0,MargenMuestreo="0D"):
    '''Corta un dataframe df con indice de tipo datetime, conservando los valores entre Tini y Tend. Si no se especifican, 
    conserva desde el inicio del dataframe o hasta el final del mismo.'''
    if Tfin!=0 :
        if Tini==0 :
            df=df.loc[:pd.to_datetime(Tfin)+pd.to_timedelta(MargenMuestreo) ]
        else:
            df=df.loc[pd.to_datetime(Tini)-pd.to_timedelta(MargenMuestreo):pd.to_datetime(Tfin)+pd.to_timedelta(MargenMuestreo)]
    else :
        if Tini!=0 :
            df=df.loc[pd.to_datetime(Tini)-pd.to_timedelta(MargenMuestreo):]
    return df

def error(csv,primsen,primest,len):
    df=pd.read_csv(csv)
    df.iloc[:,0]=pd.to_datetime(df.iloc[:,0]) # Fecha como datetime
    df=df.drop_duplicates(subset="Fecha")
    df=df.set_index("Fecha")

    dfe1=df.iloc[:,primest:primest+len*2:2]
    dfe2=df.iloc[:,primest+1:primest+len*2:2]
    dfe1.dropna(inplace=True)
    dfe2.dropna(inplace=True)
    dfs=df.iloc[:,primsen:primsen+len]
    dfs.dropna(inplace=True)
    dfs1=dfs2=dfs
    dfs1.columns=dfe1.columns
    ret=abs(dfs1-dfe1).mean()
    dfs2.columns=dfe2.columns
    ret=ret.append(abs(dfs2-dfe2).mean())

    return ret

#print(compara("estadisticosDiarios.csv","Area LTP BOSCH_2_2","estadisticosDiarios.csv","Estado BOSCH_2",rolx=14,plot3D=True,save=False))
#compara("estadisticosDiarios.csv","Area norm LTP BOSCH_2_2","estadisticosDiarios.csv","Estado BOSCH_2",rolx=14,plot3D=True,save=False)
#compara("estadisticosDiarios.csv","Maximo diario LTP BOSCH_2_2","estadisticosDiarios.csv","Estado BOSCH_2",rolx=14,plot3D=True,save=False)
#print(error("estadisticosDiarios.csv",8,20,12,"Area "))
# print(error("estadisticosDiarios.csv",8,32,12))
print(error("estadisticosDiarios.csv",7,107,12))
print(error("estadisticosDiarios.csv",7,119,12))
print(error("estadisticosDiarios.csv",7,131,12))

#print(grafica("rawMinutales.csv","TDV Control_1"))