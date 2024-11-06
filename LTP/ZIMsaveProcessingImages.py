
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


day_data="15"
month_data="07"
year_data="2014"
sufix="rht"
sensor="LTP Control_2_1"

cutDate=pd.to_datetime(year_data+"-"+month_data+"-"+day_data)

ltpitems=80
meteoitems=4

#pd.set_option('display.max_rows', None)

# create a blank dataframe to save characteristics data (X) and another for class data (Y)
savedfX = pd.DataFrame()
savedfY = pd.DataFrame()

#create blank dataframes for the plots
ZIMplots = pd.DataFrame()
MeteoPlots = pd.DataFrame()

# Load prediction data
tdvP,ltpP,meteoP,valdatapd=isl.cargaDatos(year_data,sufix)

# cut the data to the date
tdvP=tdvP.loc[tdvP.index>=cutDate,:]
ltpP=ltpP.loc[ltpP.index>=cutDate,:]
meteoP=meteoP.loc[meteoP.index>=cutDate,:]
valdatapd=valdatapd.loc[valdatapd.index>=cutDate,:]

# cut all data after 00:00 of next day
tdvP=tdvP.loc[tdvP.index<pd.to_datetime(cutDate+pd.DateOffset(1)),:]
ltpP=ltpP.loc[ltpP.index<pd.to_datetime(cutDate+pd.DateOffset(1)),:]
meteoP=meteoP.loc[meteoP.index<pd.to_datetime(cutDate+pd.DateOffset(1)),:]
valdatapd=valdatapd.loc[valdatapd.index<pd.to_datetime(cutDate+pd.DateOffset(1)),:]

# get only the data from the sensor
ltpP=ltpP.loc[:,ltpP.columns.str.startswith(sensor)]

#merge ltpP and meteoP for plotting
plotData = pd.merge(ltpP,meteoP,how='outer',left_index=True,right_index=True)

# rename the column with the sensor name to ZIM measurements
plotData = plotData.rename(columns={sensor: 'ZIM measurements'})
# rename T_Amb_Avg to Ambient Temperature
plotData = plotData.rename(columns={'T_Amb_Avg': 'Ambient Temperature'})
# rename R_Neta_Avg to Net Radiation
plotData = plotData.rename(columns={'R_Neta_Avg': 'Net Radiation'})
# rename H_Relat_Avg to Relative Humidity
plotData = plotData.rename(columns={'H_Relat_Avg': 'Relative Humidity'})
# rename VPD_Avg to Vapor Pressure Deficit
plotData = plotData.rename(columns={'VPD_Avg': 'Vapor Pressure Deficit'})

# Add the ZIM measurements to the plot dataframe as "raw data"
ZIMplots['raw'] = plotData['ZIM measurements']
# Add the ambient temperature to the plot dataframe as "raw data"
MeteoPlots['raw'] = plotData['Ambient Temperature']

# add meteo to ltpP
ltpP=ltpP.join(meteoP)

# delete all NaN columns from ltp
ltpP = ltpP.dropna(axis=1,how='all')
# fill NaN values in ltp with the previous value
ltpP = ltpP.fillna(method='ffill')
# fill NaN values in ltp with the next value
ltpP = ltpP.fillna(method='bfill')

# apply a rolling mean filter to ltp
ltpP_mf = ltpP.rolling(window=240,center=True).mean()

# # where there is no data in ltpP_mf, use the original data
# ltpP_mf = ltpP_mf.fillna(ltpP)

#store the data in ltpP
ltpP = ltpP_mf

# rename the columns of ltpP as we did before with plotData
ltpP = ltpP.rename(columns={sensor: 'ZIM measurements'})
ltpP = ltpP.rename(columns={'T_Amb_Avg': 'Ambient Temperature'})
ltpP = ltpP.rename(columns={'R_Neta_Avg': 'Net Radiation'})
ltpP = ltpP.rename(columns={'H_Relat_Avg': 'Relative Humidity'})
ltpP = ltpP.rename(columns={'VPD_Avg': 'Vapor Pressure Deficit'})

# Add the ZIM measurements to the plot dataframe as "filtered data"
ZIMplots['filtered'] = ltpP['ZIM measurements']
# Add the ambient temperature to the plot dataframe as "filtered data"
MeteoPlots['filtered'] = ltpP['Ambient Temperature']

#get the zim measurements column
ltpP_zim = ltpP['ZIM measurements']

# get the mean value of the zim measurements
ltp_medioP = ltpP_zim.groupby(ltpP_zim.index.date).mean()

# get the standard deviation of ltp for each day
ltp_stdP = ltpP_zim.groupby(ltpP_zim.index.date).std()

#get the only value of ltp_medioP as a scalar
ltp_medioP = ltp_medioP.iloc[0]
ltp_stdP = ltp_stdP.iloc[0]


# ltp normalized per day (only zim measurements)
ltpP_zim_norm = (ltpP_zim-ltp_medioP)/ltp_stdP

# save the normalized zim measurements in ltpP
ltpP['ZIM measurements'] = ltpP_zim_norm

# plot ltpP normalized
# ltpP.plot(subplots=True)

# add the normalized zim measurements to the plot dataframe
ZIMplots['normalized'] = ltpP['ZIM measurements']

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

ltpP_wo_hn = ltpP.drop('Hora_norm',axis=1)

# plot ltpP without Hora_norm
# ltpP_wo_hn.plot(subplots=True)

# get the hour at which hora_norm is 6
ltpP_6 = ltpP.loc[ltpP['Hora_norm']==6,:]
# get the hour at which hora_norm is 18
ltpP_18 = ltpP.loc[ltpP['Hora_norm']==18,:]

# add to the plot a vertical line at the hour where hora_norm is 6 in every subplot
for ax in plt.gcf().get_axes():
    ax.axvline(x=ltpP_6.index[0],color='gray')
# add to the plot a vertical line at the hour where hora_norm is 18 in every subplot
for ax in plt.gcf().get_axes():
    ax.axvline(x=ltpP_18.index[0],color='gray')

# crop ltp to the 6 to 18 time range of hora_norm
ltpP = ltpP.loc[ltpP['Hora_norm']>=6,:]
ltpP = ltpP.loc[ltpP['Hora_norm']<=18,:]

# add an copy of the normalized zim measurements to do a plot with night hours marked
ZIMplots['night'] = ZIMplots['normalized']
MeteoPlots['night'] = MeteoPlots['filtered']

# drop the hora_norm column
ltpP = ltpP.drop('Hora_norm',axis=1)
cropData = ltpP.copy()

#separate the zim columns from the meteo columns
ltpP_zim = ltpP['ZIM measurements']
ltpP_meteo = ltpP.drop('ZIM measurements',axis=1)

# take group the zim data in 16 groups with the same number of elements and get the mean of each group
ltpP_zim_grouped = ltpP_zim.groupby(pd.qcut(ltpP_zim.index,16)).mean()

# take group the meteo data in 4 groups with the same number of elements and get the mean of each group
ltpP_meteo_grouped = ltpP_meteo.groupby(pd.qcut(ltpP_meteo.index,4)).mean()


# change the index of the grouped data to the mean of the group
ltpP_zim_grouped.index = ltpP_zim_grouped.index.map(lambda x: x.mid)
ltpP_meteo_grouped.index = ltpP_meteo_grouped.index.map(lambda x: x.mid)

# convert the index of the grouped data to a datetime object
ltpP_zim_grouped.index = pd.to_datetime(ltpP_zim_grouped.index)
ltpP_meteo_grouped.index = pd.to_datetime(ltpP_meteo_grouped.index)

# plot the grouped data
# ltpP_zim_grouped.plot()
# ltpP_meteo_grouped.plot(subplots=True)
# get only the data from temperature
ltpP_meteo_grouped_temp = ltpP_meteo_grouped['Ambient Temperature']


#create a figure with 6 subplots
ax = plt.gcf().add_subplot(511)
# plot the raw data
ZIMplots['raw'].plot(ax=ax)
# set y label as "raw data"
ax.set_ylabel('Raw')
# remove the x axis tags
ax.set_xticks([])
ax = plt.gcf().add_subplot(512)
# plot the filtered data
ZIMplots['filtered'].plot(ax=ax)
# set y label as "filtered data"
ax.set_ylabel('Filtered')
# remove the x axis tags
ax.set_xticks([])
ax = plt.gcf().add_subplot(513)
# plot the normalized data
ZIMplots['normalized'].plot(ax=ax)
# set y label as "normalized data"
ax.set_ylabel('Normalized')
# remove the x axis tags
ax.set_xticks([])
ax = plt.gcf().add_subplot(514)
# plot the data with night hours marked
ZIMplots['night'].plot(ax=ax)
# place a vertical line at the hour where hora_norm is 6
ax.axvline(x=ltpP_6.index[0],color='gray')
# place a vertical line at the hour where hora_norm is 18
ax.axvline(x=ltpP_18.index[0],color='gray')
# shadow the night hours from 00:00 to 6:00 and from 18:00 to 24:00
ax.axvspan(ltpPBase.index[0],ltpP_6.index[0],color='gray',alpha=0.5)
ax.axvspan(ltpP_18.index[0],ltpPBase.index[-1],color='gray',alpha=0.5)

# set y label as "crop"
ax.set_ylabel('Crop')

# remove the x axis tags
ax.set_xticks([])

ax = plt.gcf().add_subplot(515)

ZIMcrop=cropData['ZIM measurements'].dropna()
# remove the date from the index of the base zim data
ZIMcrop.index = ZIMcrop.index.time
# plot the base zim data
ZIMcrop.plot(ax=ax)


# remove the date from the index of the grouped zim data
ltpP_zim_grouped.index = ltpP_zim_grouped.index.time
# plot the grouped zim data
ltpP_zim_grouped.plot(ax=ax, marker='x', linestyle='')

#set x ticks to be every 3 hours
ax.set_xticks([time(6,0),time(9,0),time(12,0),time(15,0),time(18,0)])
# remove the x axis label
ax.set_xlabel('')
# set y label as "averages"
ax.set_ylabel('Averages')

#make the vertical space between the subplots equal to 0.4
plt.subplots_adjust(hspace=0.4)

plt.figure()
ax = plt.gcf().add_subplot(411)

# plot the raw data
MeteoPlots['raw'].plot(ax=ax)
# set y label as "raw data"
ax.set_ylabel('Raw')
# remove the x axis tags
ax.set_xticks([])
ax = plt.gcf().add_subplot(412)
# plot the filtered data
MeteoPlots['filtered'].plot(ax=ax)
# set y label as "filtered data"
ax.set_ylabel('Filtered')
# remove the x axis tags
ax.set_xticks([])
ax = plt.gcf().add_subplot(413)
# plot the data with night hours marked
MeteoPlots['night'].plot(ax=ax)
# place a vertical line at the hour where hora_norm is 6
ax.axvline(x=ltpP_6.index[0],color='gray')
# place a vertical line at the hour where hora_norm is 18
ax.axvline(x=ltpP_18.index[0],color='gray')
# shadow the night hours from 00:00 to 6:00 and from 18:00 to 24:00
ax.axvspan(ltpPBase.index[0],ltpP_6.index[0],color='gray',alpha=0.5)
ax.axvspan(ltpP_18.index[0],ltpPBase.index[-1],color='gray',alpha=0.5)
# remove the x axis tags
ax.set_xticks([])
# set y label as "crop"
ax.set_ylabel('Crop')

# add another ax to the plot
ax = plt.gcf().add_subplot(414)

# meteo crop
meteoCrop=cropData['Ambient Temperature'].dropna()
# remove the date from the index of the base meteo data
meteoCrop.index = meteoCrop.index.time
# plot the base meteo data
meteoCrop.plot(ax=ax)

# remove the date from the index of the grouped meteo data
ltpP_meteo_grouped_temp.index = ltpP_meteo_grouped_temp.index.time
# plot the grouped meteo data
ltpP_meteo_grouped_temp.plot(ax=ax, marker='x', linestyle='')

# set x ticks to be every 3 hours
ax.set_xticks([time(6,0),time(9,0),time(12,0),time(15,0),time(18,0)])
# remove the x axis label
ax.set_xlabel('')
# set y label as "averages"
ax.set_ylabel('Averages')

#make the vertical space between the subplots equal to 0.3
plt.subplots_adjust(hspace=0.3)

plt.show()

sys.exit()