import sys
from time import time
from matplotlib.markers import MarkerStyle
import matplotlib
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp
import isadoralib as isl
ndias=5
year_train="2014"
year_data="2019"
sufix="rht"
comp=4
fltp=4
fmeteo=4
saveFolder="ignore/figures/PCALDAMETEOresults"+str(ndias)+"/"+year_train+"-"+year_data+"-"+sufix+"/"
n_dias_print=5
#matplotlib.use("Agg")

# Carga de datos de entrenamiento
tdvT,ltpT,meteoT,trdatapd=isl.cargaDatos(year_train,sufix)

# Carga de datos de predicción
tdvP,ltpP,meteoP,valdatapd=isl.cargaDatos(year_data,sufix)

# guarda la información raw para plots
ltpPlot = ltpP.copy()
meteoPlot = meteoP.copy()

# aplica un filtro de media móvil a ltp
ltpT = ltpT.rolling(window=240,center=True).mean()
ltpP = ltpP.rolling(window=240,center=True).mean()

#interpola los datos meteo
meteoT=meteoT.resample('1T').interpolate('linear')
meteoP=meteoP.resample('1T').interpolate('linear')

xt,yt=isl.datosADataframe(ltpT,meteoT,trdatapd)
xp,yp=isl.datosADataframe(ltpP,meteoP,valdatapd)

print(xp)
print(valdatapd)
sys.exit()