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

target_Y=3

# Load data
tdvP,ltpP,meteoP,valdatapd=isl.cargaDatos(year_data,sufix)

# melt the dataframes to have a single column with the values and a column with the sensor name
ltpP = ltpP.melt(ignore_index=False,var_name='Sensor',value_name='Valor')
valdatapd = valdatapd.melt(ignore_index=False,var_name='Sensor',value_name='Valor')

#divide Fecha into Fecha and Time
ltpP['Time'] = ltpP.index.time
ltpP['Fecha'] = ltpP.index.date

#convert fecha to datetime
ltpP['Fecha'] = pd.to_datetime(ltpP['Fecha'])

#set Fecha and sensor as index and unmelt the hour
ltpP = ltpP.set_index(['Fecha', 'Time','Sensor'])
ltpP = ltpP.unstack(level=1)

#add sensor as index to valdatapd, keeping also the date
valdatapd['Fecha'] = valdatapd.index.date
valdatapd['Fecha'] = pd.to_datetime(valdatapd['Fecha'])
valdatapd = valdatapd.set_index(['Fecha','Sensor'])


#remove indices from valdatapd that are not in ltpP
valdatapd = valdatapd.loc[valdatapd.index.intersection(ltpP.index)]

# select a random day from valdatapd where Valor is target_Y
valdatapd = valdatapd[valdatapd['Valor']==target_Y]
valdatapd = valdatapd.sample(n=1)

#remove indices from ltpP that are not in valdatapd
ltpP = ltpP.loc[ltpP.index.intersection(valdatapd.index)]

#remove the first level of the column index
ltpP.columns = ltpP.columns.droplevel(0)

#stack the columns
ltpP = ltpP.stack()

# get the first values of Time and sensor
Time = ltpP.index.get_level_values('Time')[0]
sensor = ltpP.index.get_level_values('Sensor')[0]


ltpP.index = ltpP.index.droplevel([0, 1])

#plot the data
ltpP.plot()
#add Y lavel as "ZIM probe value"
plt.ylabel('ZIM probe value')

#add the title as "ZIM curve with stress level "+target_Y
plt.title('ZIM curve with stress level '+str(target_Y))

plt.show()
#print the indices
print(ltpP)

print(valdatapd)