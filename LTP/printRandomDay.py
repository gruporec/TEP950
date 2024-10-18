import pandas as pd
import numpy as np
from datetime import time
import os
import sys
import random

#add the path to the lib folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
# import the isadoralib library
import isadoralib as isl


year_datas=["2014","2015","2016","2019"]
sufix="rht"

meteodata=True
normalizePerDay=False

target_valdata=2

#pd.set_option('display.max_rows', None)

# create a blank dataframe to save characteristics data (X) and another for class data (Y)
savedfX = pd.DataFrame()
savedfY = pd.DataFrame()

for year_data in year_datas:
    # Load prediction data
    tdvP,ltpP,meteoP,valdatapd=isl.cargaDatos(year_data,sufix)

    #separate yyyy-mm-dd and hh:mm:ss in two columns from the index of ltpP
    ltpP['Fecha'] = ltpP.index.date
    ltpP['hora'] = ltpP.index.time

    #make it a multiindex dataframe
    ltpP.set_index(['Fecha','hora'],inplace=True)

    #melt the columns of valdatapd to rows
    valdatapd=valdatapd.melt(ignore_index=False)

    #melt the columns of ltpP to rows
    ltpP=ltpP.melt(ignore_index=False)

    #rename variable column to sensor
    ltpP.rename(columns={'variable':'sensor'},inplace=True)
    valdatapd.rename(columns={'variable':'sensor'},inplace=True)

    #choose all rows from valdatapd where the value is target_valdata
    valdatapd=valdatapd[valdatapd['value']==target_valdata]

    #choose a row at random
    random_row=random.choice(valdatapd.index)
    print(random_row)
    #turn fecha into a string
    random_row['Fecha']=random_row['Fecha'].strftime('%Y-%m-%d')
    ltpP['Fecha']=ltpP.index.get_level_values('Fecha').strftime('%Y-%m-%d')

    #choose data from ltpP with the same index as the random row
    ltpP=ltpP.loc[random_row]
    
    print(valdatapd)
    print(ltpP)
    sys.exit()