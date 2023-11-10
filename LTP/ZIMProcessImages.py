
import pandas as pd
import numpy as np
from datetime import time
import os
import sys
import matplotlib.pyplot as plt
#add the path to the lib folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
# import the isadoralib library
import isadoralib as isl


year_data="2014"
sufix="rht"

sensorPlot="LTP Control_2_1"
dayPlot=year_data+"-07-15"


ltpitems=80
meteoitems=4


#pd.set_option('display.max_rows', None)

# create a blank dataframe to save characteristics data (X) and another for class data (Y)
savedfX = pd.DataFrame()
savedfY = pd.DataFrame()

# Load prediction data
tdvP,ltpP,meteoP,valdatapd=isl.cargaDatos(year_data,sufix)

# select only the sensorPlot column from ltpP
ltpP=ltpP.loc[:,ltpP.columns.str.startswith(sensorPlot)]

# add meteo to ltpP
ltpP=ltpP.join(meteoP)

# extract the dayPlot from ltpP
ltpPPlot=ltpP.loc[dayPlot,:]

# print ltpP
print(ltpPPlot)
# plot ltpP
ltpPPlot.plot(subplots=True,title="Raw data")

# delete all NaN columns from ltp
ltpP = ltpP.dropna(axis=1,how='all')
# fill NaN values in ltp with the previous value
ltpP = ltpP.fillna(method='ffill')
# fill NaN values in ltp with the next value
ltpP = ltpP.fillna(method='bfill')

# apply a rolling mean filter to ltp
ltpP = ltpP.rolling(window=30,center=True).mean()

# extract the dayPlot from ltpP
ltpPPlot=ltpP.loc[dayPlot,:]

# print ltpP
print(ltpPPlot)
# plot ltpP
ltpPPlot.plot(subplots=True, title="Filtered data")

# get the mean value of ltp for each day
ltp_medioP = ltpP.groupby(ltpP.index.date).mean()

# get the standard deviation of ltp for each day
ltp_stdP = ltpP.groupby(ltpP.index.date).std()

# change the index to datetime
ltp_medioP.index = pd.to_datetime(ltp_medioP.index)
ltp_stdP.index = pd.to_datetime(ltp_stdP.index)

# resample ltp_medio and ltp_std to minutes
ltp_medioP = ltp_medioP.resample('T').ffill()
ltp_stdP = ltp_stdP.resample('T').ffill()

# ltp normalized per day
ltpP = (ltpP - ltp_medioP) / ltp_stdP

# extract the dayPlot from ltpP
ltpPPlot=ltpP.loc[dayPlot,:]

# print ltpP
print(ltpPPlot)
# plot ltpP
ltpPPlot.plot(subplots=True,title="Normalized data")

# get every sign change of R_Neta_Avg in meteo
signosP = np.sign(meteoP.loc[:,meteoP.columns.str.startswith('R_Neta_Avg')]).diff()
# get positive to negative sign changes
signos_pnP = signosP<0
# remove false values (not sign changes)
signos_pnP = signos_pnP.replace(False,np.nan).dropna()
# get negative to positive sign changes
signos_npP = signosP>0
# remove false values (not sign changes)
signos_npP = signos_npP.replace(False,np.nan).dropna()

# duplicate the index of signos_np as a new column in signos_np
signos_npP['Hora'] = signos_npP.index
# crop signos_np to the first value of each day
signos_npP = signos_npP.resample('D').first()

# remove days with no sign change
signos_npP=signos_npP.dropna()

# duplicate the index of signos_pn as a new column in signos_pn
signos_pnP['Hora'] = signos_pnP.index
# crop signos_pn to the last value of each day
signos_pnP = signos_pnP.resample('D').last()

# remove days with no sign change
signos_pnP = signos_pnP.dropna()

# get ltp values where the time is 00:00
ltp_00P = ltpP.index.time == time.min
# get ltp values where the time is 23:59
ltp_23P = ltpP.index.time == time(23,59)

# create a column in ltp that is 0 at 00:00
ltpP.loc[ltp_00P,'Hora_norm'] = 0
# make Hora_norm equal to 6 in the indices of signos np
ltpP.loc[signos_npP['Hora'],'Hora_norm'] = 6
# make Hora_norm equal to 18 in the indices of signos pn
ltpP.loc[signos_pnP['Hora'],'Hora_norm'] = 18
# make Hora_norm equal to 24 in the last value of each day
ltpP.loc[ltp_23P,'Hora_norm'] = 24
# make the last value of Hora_norm equal to 24
ltpP.loc[ltpP.index[-1],'Hora_norm'] = 24
# interpolate Hora_norm
ltpP.loc[:,'Hora_norm'] = ltpP.loc[:,'Hora_norm'].interpolate()

# store the values before cropping
ltpPBase=ltpP

# crop ltp to the 6 to 18 time range of hora_norm
ltpP = ltpP.loc[ltpP['Hora_norm']>=6,:]
ltpP = ltpP.loc[ltpP['Hora_norm']<=18,:]

# add the normalized time to the index of ltp
ltpP.index = [ltpP.index.strftime('%Y-%m-%d'),ltpP['Hora_norm']]

# create the index of ltpPBase
ltpPBase['Hora_norm']=ltpPBase['Hora_norm'].apply(pd.to_timedelta,unit='H')
ltpPBase['dia_norm'] = ltpPBase.index.strftime('%Y-%m-%d')
ltpPBase.index = [ltpPBase['dia_norm'].apply(pd.to_datetime,format='%Y-%m-%d'),ltpPBase['Hora_norm']]
ltpPBase=ltpPBase.drop('Hora_norm',axis=1)
ltpPBase=ltpPBase.drop('dia_norm',axis=1)
ltpPBase=ltpPBase.unstack(level=0)

valdatapd.index = valdatapd.index.strftime('%Y-%m-%d')


# extract the dayPlot from ltpP
ltpP=ltpP.loc[dayPlot,:]


# get the intersection index of valdatapd and the first level of the ltp index
ltpPdates = ltpP.index.get_level_values(0)
valdatapd_ltp = valdatapd.index.intersection(ltpPdates)

# drop the Hora_norm column from ltp
ltpPPlot = ltpP.drop('Hora_norm',axis=1)

# print ltpP
print(ltpPPlot)
# plot ltpP
ltpPPlot.plot(subplots=True, title="Cropped data")


# split meteo from ltp again
meteoP_norm=ltpP.drop(ltpP.columns[ltpP.columns.str.startswith('LTP')], axis=1)

# remove the values of ltp that are not in valdatapd (unneeded for representation)
ltpv = ltpP

# unstack meteoP_norm
meteoP_norm = meteoP_norm.unstack(level=0)

# unstack ltpv
ltpv = ltpv.unstack(level=0)

print(ltpv)

# create an index to adjust frequencies
ltpv_index_float=pd.Index(np.floor(ltpv.index*1000000000),dtype='float64')
meteoP_index_float=pd.Index(np.floor(meteoP_norm.index*1000000000),dtype='float64')

ltpv.index = pd.to_datetime(ltpv_index_float)
meteoP_norm.index = pd.to_datetime(meteoP_index_float)

ltpv_orig=ltpv.copy()
meteoP_norm_orig=meteoP_norm.copy()

fltp=12/ltpitems
if meteoitems>0:
    fmeteo=12/meteoitems
else:
    fmeteo=0
# index as datetime to adjust frequencies
ltpv=ltpv_orig.resample(str(int(fltp*1000))+'L').mean()
if meteoitems>0:
    meteoP_norm=meteoP_norm_orig.resample(str(int(fmeteo*1000))+'L').mean()

# keep values from 1970-01-01 00:00:06.000 to 1970-01-01 00:00:17.900 (data was converted as time from epoch; this will keep dailight time only)
ltpv = ltpv.loc[ltpv.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
ltpv = ltpv.loc[ltpv.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]

if meteoitems>0:
    meteoP_norm = meteoP_norm.loc[meteoP_norm.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
    meteoP_norm = meteoP_norm.loc[meteoP_norm.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]


# create a series to restore the index
norm_index=pd.Series(np.arange(6,18,fltp))
# adjust the index of ltpv to the series
ltpv.index=norm_index

if meteoitems>0:
    # create a series to restore the index
    norm_index=pd.Series(np.arange(6,18,fmeteo))
    # crop norm_index to match the size of meteoP_norm if there has been a mismatch when calculating the dataframe
    norm_index=norm_index.loc[norm_index.index<len(meteoP_norm)]
    # adjust the index of meteoP_norm to the series
    meteoP_norm.index=norm_index

    # drop the Hora_norm column from meteo
    meteoP_norm = meteoP_norm.drop('Hora_norm',axis=1)

    # stack meteoP_norm
    meteoP_norm = meteoP_norm.stack(level=0)

    # swap the levels of the index of meteo
    meteoP_norm.index = meteoP_norm.index.swaplevel(0,1)

    meteoP_norm=meteoP_norm.dropna(axis=1,how='all')

    # merge the two levels of the index of meteo
    meteoP_norm.index = meteoP_norm.index.map('{0[1]}/{0[0]}'.format)

else:
    meteoP_norm = pd.DataFrame()

# create an empty numpy array
array_ltpv=np.empty((len(ltpv)+len(meteoP_norm),0))

print(ltpv.columns.levels[0])

# for each element in the first column index of ltp
for i in ltpv.columns.levels[0]:
    ltpv_col=ltpv.loc[:,i]
    if meteoitems>0:
        # remove the values of meteo that are not in ltp_col
        meteo_ltp = ltpv_col.columns.intersection(meteoP_norm.columns)
        meteoP_col = meteoP_norm.loc[:,meteo_ltp]

        # merge ltpv with meteo
        merge_ltp_meteo = pd.merge(ltpv.loc[:,i],meteoP_col,how='outer')
    else:
        merge_ltp_meteo = ltpv.loc[:,i]
    # add the resulting dataframe to the numpy array
    array_ltpv=np.append(array_ltpv,merge_ltp_meteo.values,axis=1)

print(array_ltpv)
sys.exit()

print(ltpv)
# extract the dayPlot from ltpv
ltpPPlot=ltpv

# unstack the second column index of ltpv
ltpPPlot = ltpPPlot.stack(level=1)

# select the dayPlot from the second index of ltpPPlot
ltpPPlot=ltpPPlot.xs(dayPlot, level=1, drop_level=False)

# print ltpP
print(ltpPPlot)

# remove the second index

# drop the Hora_norm column from ltp
ltpPPlot = ltpPPlot.drop('Hora_norm')

# print ltpP
print(ltpPPlot)
# plot ltpP
ltpPPlot.plot(subplots=True, title="Cropped data")

# print("ltpt")
# print(ltpt)
# print("meteo_ltp")
# print(meteo_ltp)
# print("meteoT_norm")
# print(meteoT_norm)
# print("ltpt_col")
# print(ltpt_col)

Xv=array_ltpv.transpose()
Yv=valdatapd.unstack()

# Xv as dataframe with the index of Yv
#Xv=pd.DataFrame(Xv,index=Yv.index)

# add Xv and Yv to their respective dataframes
#savedfX=pd.concat([savedfX,Xv])
#savedfY=pd.concat([savedfY,Yv])
