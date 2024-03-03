import sys
from time import time
from matplotlib.markers import MarkerStyle
import matplotlib
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import time
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp

year_data="2019"
sufix="rht"
comp=4
fltp=4
fmeteo=4
saveFolder="ignore/figures/rawdata"
n_dias_print=5
plotSensor='LTP Control_2_1'
plotDate=pd.to_datetime('2019-05-07')
#matplotlib.use("Agg")

# Carga de datos de predicción
dfP = pd.read_csv("rawMinutales"+year_data+sufix+".csv",na_values='.')
dfP.loc[:,"Fecha"]=pd.to_datetime(dfP.loc[:,"Fecha"])# Fecha como datetime
dfP=dfP.drop_duplicates(subset="Fecha")
dfP.dropna(subset = ["Fecha"], inplace=True)
dfP=dfP.set_index("Fecha")
dfP=dfP.apply(pd.to_numeric, errors='coerce')

# guarda la información raw para plots
tdvPlot = dfP.loc[:,dfP.columns.str.startswith('TDV')]
ltpPlot = dfP.loc[:,dfP.columns.str.startswith('LTP')]
meteoPlot = dfP.drop(dfP.columns[dfP.columns.str.startswith('TDV')], axis=1)
meteoPlot = meteoPlot.drop(meteoPlot.columns[meteoPlot.columns.str.startswith('LTP')], axis=1)

# for column in ltpPlot:
#     for idx, day in ltpPlot[column].groupby(ltpPlot[column].index.date):

meteod=meteoPlot[plotDate-pd.Timedelta(n_dias_print-1,'d'):plotDate+pd.Timedelta(1,'d')]
ltpd=ltpPlot[plotSensor][plotDate-pd.Timedelta(n_dias_print-1,'d'):plotDate+pd.Timedelta(1,'d')]

fig, [ax1,ax2,ax3,ax4,ax5] = plt.subplots(5,1)

ax1.plot(ltpd)
ax1.set(xlabel='Hora', ylabel='LTP', title=plotSensor+' '+str(plotDate))
ax1.fill_between(meteod.index, ltpd.min(), ltpd.max(), where=(meteod['R_Neta_Avg'] < 0), alpha=0.5,color=(232/255, 222/255, 164/255, 0.5))
ax1.grid()

ax2.plot(meteod['T_Amb_Avg'])
ax2.set(xlabel='Hora', ylabel='T amb (ºC)')
ax2.fill_between(meteod.index, meteod['T_Amb_Avg'].min(), meteod['T_Amb_Avg'].max(), where=(meteod['R_Neta_Avg'] < 0), alpha=0.5,color=(232/255, 222/255, 164/255, 0.5))
ax2.grid()

ax3.plot(meteod['H_Relat_Avg'])
ax3.set(xlabel='Hora', ylabel='H rel (%)')
ax3.fill_between(meteod.index, meteod['H_Relat_Avg'].min(), meteod['H_Relat_Avg'].max(), where=(meteod['R_Neta_Avg'] < 0), alpha=0.5,color=(232/255, 222/255, 164/255, 0.5))
ax3.grid()

ax4.plot(meteod['VPD_Avg'])
ax4.set(xlabel='Hora', ylabel='VPD')
ax4.fill_between(meteod.index, meteod['VPD_Avg'].min(), meteod['VPD_Avg'].max(), where=(meteod['R_Neta_Avg'] < 0), alpha=0.5,color=(232/255, 222/255, 164/255, 0.5))
ax4.grid()

ax5.plot(meteod['R_Neta_Avg'])
ax5.set(xlabel='Hora', ylabel='Rad Neta')
ax5.fill_between(meteod.index, meteod['R_Neta_Avg'].min(), meteod['R_Neta_Avg'].max(), where=(meteod['R_Neta_Avg'] < 0), alpha=0.5,color=(232/255, 222/255, 164/255, 0.5))
ax5.grid()

plt.show()

        # plt.savefig(currFolder+'/'+str(idx+pd.Timedelta(n_dias_print-1,'d'))+'.png')
        # plt.close()
