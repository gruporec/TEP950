import pandas as pd

def cargaDatos(year,sufix):
    '''Carga los datos de un año almacenados en los archivos [year][sufix].csv y validacion[year].csv y devuelve una tupla (tdv,ltp,meteo,estado hídrico).'''
    # Carga de datos
    df = pd.read_csv("rawMinutales"+year+sufix+".csv",na_values='.')
    df.loc[:,"Fecha"]=pd.to_datetime(df.loc[:,"Fecha"])# Fecha como datetime
    df=df.drop_duplicates(subset="Fecha")
    df.dropna(subset = ["Fecha"], inplace=True)
    df=df.set_index("Fecha")
    df=df.apply(pd.to_numeric, errors='coerce')

    # separa dfT en tdv y ltp en función del principio del nombre de cada columna y guarda el resto en meteo
    tdv = df.loc[:,df.columns.str.startswith('TDV')]
    ltp = df.loc[:,df.columns.str.startswith('LTP')]
    meteo = df.drop(df.columns[df.columns.str.startswith('TDV')], axis=1)
    meteo = meteo.drop(meteo.columns[meteo.columns.str.startswith('LTP')], axis=1)

    # Carga datos de validacion
    valdatapd=pd.read_csv("validacion"+year+".csv")
    valdatapd.dropna(inplace=True)
    valdatapd['Fecha'] = pd.to_datetime(valdatapd['Fecha'])
    valdatapd.set_index('Fecha',inplace=True)

    return (tdv,ltp,meteo,valdatapd)