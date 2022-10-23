import pandas as pd

def cargaDatos(year,sufix):
    '''Carga los datos de un año almacenados en el archivo [year][sufix].csv y devuelve una tupla (tdv,ltp,meteo,raw).'''
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

    return (tdv,ltp,meteo,df)