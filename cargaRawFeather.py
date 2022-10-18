import pandas as pd

year="2014"

df1 = pd.read_csv(year+" hoja 1.csv",na_values='.') # Meteo diario

#Preprocesado: rellena precipitaciones con 0
df1['Precipitación'] = df1['Precipitación'].fillna(0)

#Preprocesado: rellena riegos con el último valor
for col in range (4,len(df1.columns)):
    df1.iloc[:,col]=df1.iloc[:,col].fillna(method='ffill')

df2 = pd.read_csv(year+" hoja 2.csv",na_values='.') # Estados diarios

#Combina datos diarios y elimina datos sin fecha
dfd=pd.merge(df1,df2,how="outer",on="Fecha")
dfd.dropna(subset = ["Fecha"], inplace=True)
dfd.iloc[:,0]=pd.to_datetime(dfd.iloc[:,0])

# Guarda datos RAW
dfd.to_feather("rawDiarios"+year+".feather")

df31 = pd.read_csv(year+" hoja 31.csv",na_values='.') # TDV freq variable
df31.iloc[:,0]=pd.to_datetime(df31.iloc[:,0]) # Fecha como datetime
#elimina valores duplicados de Fecha
df31.drop_duplicates(subset=['Fecha'], keep='first', inplace=True)

# Interpola df31
df31T=df31.set_index("Fecha").resample(rule="T").interpolate("linear")

df32 = pd.read_csv(year+" hoja 32.csv",na_values='.') # LTP cada 5 minutos con desviación de 1 minuto

df32split=df32 # dataframe para recortar y combinar df32
df32merge = pd.DataFrame() # dataframe para añadir los datos preprocesados
for x in range(len(df32.columns)//2): # Cada dos columnas
    df32pair=df32split.iloc[:, :2]
    df32split=df32split.iloc[:, 2:]
    df32pair.columns.values[0] = "Fecha"
    df32pair.iloc[:,0]=pd.to_datetime(df32pair.iloc[:,0]) # Fecha como datetime
    df32pair=df32pair.drop_duplicates(subset="Fecha")
    df32pair.dropna(subset = ["Fecha"], inplace=True)
    df32pair=df32pair.set_index("Fecha").resample(rule="T").interpolate("linear")
    df32merge=df32merge.join(df32pair, how="outer")

dfT=df31T.join(df32merge, how="outer") # dataframe (df) minutal (T, según el criterio de la librería pandas). No confundir con la Discrete Fourier Transform

df4 = pd.read_csv(year+" hoja 4.csv",na_values='.') # Meteo cada 30 minutos
df4.iloc[:,0]=pd.to_datetime(df4.iloc[:,0]) # Fecha como datetime
df4.columns.values[0] = "Fecha"
df4=df4.drop_duplicates(subset="Fecha")
df4=df4.dropna()

# convierte los datos de df4 a numéricos
for col in range (1,len(df4.columns)):
    df4.iloc[:,col]=pd.to_numeric(df4.iloc[:,col])

# Interpola df4
df4=df4.set_index("Fecha").resample(rule="T").interpolate("linear")

dfT=dfT.join(df4, how="outer")

dfT.to_feather("rawMinutales"+year+".feather")