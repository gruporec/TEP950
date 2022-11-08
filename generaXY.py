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
year="2015"
sufix=""
comp=4
fltp=4
fmeteo=4
saveFolder="ignore/xycsv/"
#matplotlib.use("Agg")

# Carga de datos de entrenamiento
tdvT,ltpT,meteoT,trdatapd=isl.cargaDatos(year,sufix)

# aplica un filtro de media móvil a ltp
ltpT = ltpT.rolling(window=240,center=True).mean()

#interpola los datos meteo
meteoT=meteoT.resample('1T').interpolate('linear')
xt,yt=isl.datosADataframe(ltpT,meteoT,trdatapd)

xt.to_csv(saveFolder+"x"+year+".csv")
yt.to_csv(saveFolder+"y"+year+".csv")