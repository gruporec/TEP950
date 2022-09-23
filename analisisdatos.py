# Import required packages
import pandas as pd

valpdraw=pd.read_csv("validacion.csv")
valpdraw['Fecha'] = pd.to_datetime(valpdraw['Fecha'])
valpd=valpdraw.set_index('Fecha')

calpdraw=pd.read_csv("calibracion3.csv")
calpdraw['Fecha'] = pd.to_datetime(calpdraw['Fecha'])
calpd=calpdraw.set_index('Fecha')

estdatapd=pd.read_csv("estadisticosDiarios.csv")
estdatapd['Fecha'] = pd.to_datetime(estdatapd['Fecha'])
estdatapd=estdatapd.set_index('Fecha')

sav=pd.DataFrame()

for val in valpd.count().index:
    sav.loc['Numero de datos de validacion',val]=valpd.count()[val]
for cal in calpd.count().index:
    sav.loc['Numero de datos de calibracion',cal]=calpd.count()[cal]
for est in estdatapd.count().index:
    sav.loc['Numero de datos de sensores',est]=estdatapd.count()[est]

sav.to_csv('metadatos.csv')