import pandas as pd

def calculaEstado(x,min,max):
    estado=2
    if x<min:
        estado=3
    if x>max:
        estado=1
    return estado
def calculaError(estimacion,valor):
    error=abs(estimacion-valor)
    return error

# df1 = pd.read_csv("2019 hoja 1.csv",na_values='.') # Meteo diario

# #Preprocesado: rellena precipitaciones con 0
# df1['Precipitación'] = df1['Precipitación'].fillna(0)

# #Preprocesado: rellena riegos con el último valor
# for col in range (4,len(df1.columns)):
#     df1.iloc[:,col]=df1.iloc[:,col].fillna(method='ffill')

# df2 = pd.read_csv("2019 hoja 2.csv",na_values='.') # Estados diarios

# #Combina datos diarios y elimina datos sin fecha
# dfd=pd.merge(df1,df2,how="outer",on="Fecha")
# dfd.dropna(subset = ["Fecha"], inplace=True)
# dfd.iloc[:,0]=pd.to_datetime(dfd.iloc[:,0])

# # Guarda datos RAW
# dfd.to_csv("rawDiarios.csv", index=False)

# df31 = pd.read_csv("2019 hoja 31.csv",na_values='.') # TDV freq variable
# df31.iloc[:,0]=pd.to_datetime(df31.iloc[:,0]) # Fecha como datetime

# # Interpola df31
# df31T=df31.set_index("Fecha").resample(rule="T").interpolate("linear")

# df32 = pd.read_csv("2019 hoja 32.csv",na_values='.') # LTP cada 5 minutos con desviación de 1 minuto

# df32split=df32 # dataframe para recortar y combinar df32
# df32merge = pd.DataFrame() # dataframe para añadir los datos preprocesados
# for x in range(len(df32.columns)//2): # Cada dos columnas
#     df32pair=df32split.iloc[:, :2]
#     df32split=df32split.iloc[:, 2:]
#     df32pair.columns.values[0] = "Fecha"
#     df32pair.iloc[:,0]=pd.to_datetime(df32pair.iloc[:,0]) # Fecha como datetime
#     df32pair=df32pair.drop_duplicates(subset="Fecha")
#     df32pair.dropna(subset = ["Fecha"], inplace=True)
#     df32pair=df32pair.set_index("Fecha").resample(rule="T").interpolate("linear")
#     df32merge=df32merge.join(df32pair, how="outer")

# dfT=df31T.join(df32merge, how="outer") # dataframe (df) minutal (T, según el criterio de la librería pandas). No confundir con la Discrete Fourier Transform

# df4 = pd.read_csv("2019 hoja 4.csv",na_values='.') # Meteo cada 30 minutos
# df4.iloc[:,0]=pd.to_datetime(df4.iloc[:,0]) # Fecha como datetime
# df4.columns.values[0] = "Fecha"

# # Interpola df4
# df4=df4.set_index("Fecha").resample(rule="T").interpolate("linear")

# dfT=dfT.join(df4, how="outer")

# dfT.to_csv("rawMinutales.csv", index=True)

# Carga de datos (en caso de comentar la parte anterior, descomentar)
dfT = pd.read_csv("rawMinutales.csv",na_values='.') 
dfT.iloc[:,0]=pd.to_datetime(dfT.iloc[:,0]) # Fecha como datetime
dfT=dfT.drop_duplicates(subset="Fecha")
dfT.dropna(subset = ["Fecha"], inplace=True)
dfT=dfT.set_index("Fecha")

dfd = pd.read_csv("rawDiarios.csv",na_values='.') 
dfd.iloc[:,0]=pd.to_datetime(dfd.iloc[:,0]) # Fecha como datetime
dfd=dfd.drop_duplicates(subset="Fecha")
dfd.dropna(subset = ["Fecha"], inplace=True)
dfd=dfd.set_index("Fecha")

lpt=dfT.iloc[:,12:36]
lpt0=lpt.resample('1D').first()
lptmax=lpt.resample('1D').max()-lpt0
lpti=lpt.resample('1D').sum()/(60*24)-lpt0
lptin=lpti/(lptmax)
lpti=lpti.add_prefix("Area ")
lptin=lptin.add_prefix("Area norm ")
lptmax=lptmax.add_prefix("Maximo diario ")
print(lpti)
print(lptin)
print(lptmax)

dfdp=pd.merge(dfd,lpti,how="outer",on="Fecha")
dfdp=pd.merge(dfdp,lptin,how="outer",on="Fecha")
dfdp=pd.merge(dfdp,lptmax,how="outer",on="Fecha")
print(dfdp)

estArea=lpti.rolling(14).mean().applymap(calculaEstado,min=0,max=15)
estAreaN=lptin.rolling(14).mean().applymap(calculaEstado,min=-0.4,max=0.2)
estMax=lptmax.rolling(14).mean().applymap(calculaEstado,min=30,max=55)
estArea=estArea.add_prefix("Estado ")
estAreaN=estAreaN.add_prefix("Estado ")
estMax=estMax.add_prefix("Estado ")

print(estArea)
print(estAreaN)
print(estMax)

dfdp=pd.merge(dfdp,estArea,how="outer",on="Fecha")
dfdp=pd.merge(dfdp,estAreaN,how="outer",on="Fecha")
dfdp=pd.merge(dfdp,estMax,how="outer",on="Fecha")

dfdp.to_csv("estadisticosDiarios.csv", index=True)