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
csv_folder="ignore/xycsv/"
year_train="2014"
year_test="2015"
year_data="2019"
# sufix="rht"
# comp=4
# fltp=4
# fmeteo=4
saveFolder="ignore/figures/PCALDAMETEOresults"+str(ndias)+"/"+year_train+"-"+year_data+"-"+year_test+"/"
n_dias=5
#matplotlib.use("Agg")

# Carga de datos de entrenamiento
xtr=pd.read_csv(csv_folder+"x"+year_train+".csv",index_col=[0,1],header=[0,1])
xtr.index = xtr.index.set_levels(pd.to_timedelta(xtr.index.levels[1]), level=1)
xtr.columns = xtr.columns.set_levels(pd.to_datetime(xtr.columns.levels[1]), level=1)
ytr=pd.read_csv(csv_folder+"y"+year_train+".csv",index_col=[0,1])

# Carga de datos de validación
xv=pd.read_csv(csv_folder+"x"+year_data+".csv",index_col=[0,1],header=[0,1])
xv.index = xv.index.set_levels(pd.to_timedelta(xv.index.levels[1]), level=1)
xv.columns = xv.columns.set_levels(pd.to_datetime(xv.columns.levels[1]), level=1)
yv=pd.read_csv(csv_folder+"y"+year_data+".csv",index_col=[0,1])

# Carga de datos de test
xts=pd.read_csv(csv_folder+"x"+year_test+".csv",index_col=[0,1],header=[0,1])
xts.index = xts.index.set_levels(pd.to_timedelta(xts.index.levels[1]), level=1)
xts.columns = xts.columns.set_levels(pd.to_datetime(xts.columns.levels[1]), level=1)
yts=pd.read_csv(csv_folder+"y"+year_test+".csv",index_col=[0,1])
#xtr.index = xtr.index.set_levels(pd.to_timedelta(xtr.index.levels[1])+pd.Timedelta(1,"day"), level=1)
print(xtr)

xtrn=xtr.copy()
for i in range(ndias+1):
    xtrn.index=xtrn.index.set_levels(xtrn.index.levels[1]-pd.Timedelta(1,"day"), level=1)
    
    xtrn=pd.concat([xtrn,xtr])
print(xtrn)