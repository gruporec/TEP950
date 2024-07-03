import pandas as pd
import numpy as np
from datetime import time
import os
import sys
#add the path to the lib folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
# import the isadoralib library
import isadoralib as isl


year_datas=["2014","2015","2016","2019"]
sufix="rht"

meteodata=False
normalizePerDay=False

#pd.set_option('display.max_rows', None)

# create a blank dataframe to save characteristics data (X) and another for class data (Y)
savedfX = pd.DataFrame()
savedfY = pd.DataFrame()

for year_data in year_datas:
    # Load prediction data
    tdvP,ltpP,meteoP,valdatapd=isl.cargaDatosTDV(year_data,sufix)

    if meteodata:
        # add meteo to tdvP
        tdvP=tdvP.join(meteoP)

    # delete all NaN columns from tdv
    tdvP = tdvP.dropna(axis=1,how='all')
    # fill NaN values in tdv with the previous value
    tdvP = tdvP.fillna(method='ffill')
    # fill NaN values in tdv with the next value
    tdvP = tdvP.fillna(method='bfill')

    if normalizePerDay:
        # get the mean value of tdv for each day
        tdv_medioP = tdvP.groupby(tdvP.index.date).mean()

        # get the standard deviation of tdv for each day
        tdv_stdP = tdvP.groupby(tdvP.index.date).std()

        # change the index to datetime
        tdv_medioP.index = pd.to_datetime(tdv_medioP.index)
        tdv_stdP.index = pd.to_datetime(tdv_stdP.index)

        # resample tdv_medio and tdv_std to minutes
        tdv_medioP = tdv_medioP.resample('T').pad()
        tdv_stdP = tdv_stdP.resample('T').pad()

        # tdv normalized per day
        tdvP = (tdvP - tdv_medioP) / tdv_stdP
    # create two columns Fecha and Hora with the date and time of the index as string
    tdvP['Fecha'] = tdvP.index.date.astype(str)
    tdvP['Hora'] = tdvP.index.time.astype(str)

    # substitute the index with the columns Fecha and Hora
    tdvP = tdvP.set_index(['Fecha','Hora'])

    # valdata index as string
    valdatapd.index = valdatapd.index.strftime('%Y-%m-%d')

    # unstack valdatapd
    valdatapd = valdatapd.unstack()

    # unstack the second level of the index of tdvP
    tdvP = tdvP.unstack(level=1)
    #stack the first level of the index of tdvP
    tdvP = tdvP.stack(level=0)

    # swap the levels of the index of tdvP
    tdvP = tdvP.swaplevel()

    # sort the index of tdvP
    tdvP = tdvP.sort_index()

    # sort the index of valdatapd
    valdatapd = valdatapd.sort_index()

    if meteodata:
        # separate the meteo data once again from tdvP as the index that doesn't start with tdv
        meteoT = tdvP.loc[tdvP.index.get_level_values(0).str.startswith('TDV')==False,:]
        # separate the tdv data from tdvP as the index that starts with tdv
        tdvP = tdvP.loc[tdvP.index.get_level_values(0).str.startswith('TDV')==True,:]

        # add a column level to tdvP with the name 'TDV'
        tdvP.columns = pd.MultiIndex.from_product([['TDV'],tdvP.columns])

        # unstack the first level of the index of meteoT
        meteoT = meteoT.unstack(level=0)
        # swap the levels of the columns of meteoT
        meteoT = meteoT.swaplevel(axis=1)
        # sort the columns of meteoT
        meteoT = meteoT.sort_index(axis=1)

        # merge meteo into tdv by the second level of the index in tdv and the only level of the index in meteoT
        tdvP = tdvP.join(meteoT)

    # get the intersection index of valdatapd and tdvP
    valdatapd_tdv = valdatapd.index.intersection(tdvP.index)

    # remove the values of tdv that are not in valdatapd
    tdvv = tdvP.loc[valdatapd_tdv,:]
    
    # print("tdvt")
    # print(tdvt)
    # print("meteo_tdv")
    # print(meteo_tdv)
    # print("meteoT_norm")
    # print(meteoT_norm)
    # print("tdvt_col")
    # print(tdvt_col)

    Xv=tdvv
    Yv=valdatapd

    # Xv as dataframe with the index of Yv
    Xv=pd.DataFrame(Xv,index=Yv.index)

    # add Xv and Yv to their respective dataframes
    savedfX=pd.concat([savedfX,Xv])
    savedfY=pd.concat([savedfY,Yv])

# add savedfY to savedfX as a column named 'Y', subtracting 1 so that the values of Y are between 0 and 2 (as is the usual case when numbering classes)
savedfX['Y']=savedfY-1

# create a string with the last two digits of each year in year_datas
year_datas_str = ''.join(year_datas[-2:] for year_datas in year_datas)
if meteodata:
    year_datas_str = year_datas_str + 'meteo'
# store savedfX in a csv with a name composed of 'db' followed by the last two digits of each year in year_datas
savedfX.to_csv('TDVdb'+year_datas_str+'raw.csv')