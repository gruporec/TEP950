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


year_data="2014"
sufix="rht"

meteodata=True
normalizePerDay=False

target_valdata=2

#pd.set_option('display.max_rows', None)

# create a blank dataframe to save characteristics data (X) and another for class data (Y)
savedfX = pd.DataFrame()
savedfY = pd.DataFrame()

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

#get a single sample from valdatapd
sample=valdatapd.sample()


print(valdatapd)
print(sample)
print(ltpP)
sys.exit()