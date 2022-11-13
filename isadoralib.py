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

def datosADataframe(ltp:pd.DataFrame,meteo:pd.DataFrame,valdatapd:pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    '''Almacena los datos de ltp y meteo en un dataframe x y los de valdata en una serie y con la forma adecuada para convertirlos a arrays de numpy para scikit o bien para continuar su procesado. X e Y no se reducen a columnas comunes.'''
    ltp['Dia'] = pd.to_datetime(ltp.index).date
    ltp['Delta'] = pd.to_datetime(ltp.index) - pd.to_datetime(ltp.index).normalize()


    meteo['Dia'] = pd.to_datetime(meteo.index).date
    meteo['Delta'] = pd.to_datetime(meteo.index) - pd.to_datetime(meteo.index).normalize()

    # ltpPdia = ltpP.loc[meteoP['R_Neta_Avg']>0]

    ltp=ltp.set_index(['Dia','Delta']).unstack(0)
    meteo=meteo.set_index(['Dia','Delta']).unstack(0).stack(0)
    valdatapd=valdatapd.unstack()

    #common_col = ltp.columns.intersection(valdatapd.index)
    #ltp=ltp[common_col]
    y=valdatapd#[common_col]

    meteoPext=pd.DataFrame(columns=ltp.columns)
    for col in meteoPext:
        meteoPext[col]=meteo[col[1]]
    x=meteoPext.unstack(0)
    x.loc['LTP']=ltp.unstack(0)
    x=x.stack(2)
    return (x, y)