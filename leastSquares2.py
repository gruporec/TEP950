# Import required packages
from datetime import timedelta
from math import inf
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Design variables
minbakcwindow=1
maxbakcwindow=16
savefile='lsqr.csv'

# Load calibration output
calpdraw=pd.read_csv("calibracion3.csv")
calpdraw['Fecha'] = pd.to_datetime(calpdraw['Fecha'])
calpd=calpdraw.set_index('Fecha')

dates=calpd.index
cal=calpd.stack().to_numpy()
cal=np.atleast_2d(cal).T

# Load calibration input
caldatapd=pd.read_csv("estadisticosDiarios.csv")
caldatapd['Fecha'] = pd.to_datetime(caldatapd['Fecha'])
caldatapd=caldatapd.set_index('Fecha')

# Load calibration data with windows
calAreapd=pd.DataFrame()
calMaxpd=pd.DataFrame()

for i in range(maxbakcwindow):
    aux=caldatapd[caldatapd.columns.intersection(calpd.add_prefix('Area norm ').columns)][caldatapd.index.isin(calpd.index-timedelta(days=i))]
    aux.index=aux.index+timedelta(days=i)
    aux=aux.stack()
    calAreapd[i]=aux
    
    aux=caldatapd[caldatapd.columns.intersection(calpd.add_prefix('Maximo diario ').columns)][caldatapd.index.isin(calpd.index-timedelta(days=i))]
    aux.index=aux.index+timedelta(days=i)
    aux=aux.stack()
    calMaxpd[i]=aux


calArea=calAreapd.to_numpy()
calMax=calMaxpd.to_numpy()

# Load validation output
valpdraw=pd.read_csv("validacion.csv")
valpdraw.fillna(0,inplace=True)
valpdraw['Fecha'] = pd.to_datetime(valpdraw['Fecha'])
valpd=valpdraw.set_index('Fecha')

datesval=valpd.index
val=valpd.stack().to_numpy()
val=np.atleast_2d(val).T

# Load validation input
valdatapd=pd.read_csv("estadisticosDiarios.csv")
valdatapd.replace([np.inf, -np.inf], 0, inplace=True)
valdatapd.fillna(0,inplace=True)
valdatapd['Fecha'] = pd.to_datetime(valdatapd['Fecha'])
valdatapd.set_index('Fecha',inplace=True)

valAreapd=valdatapd[valdatapd.columns.intersection(valpd.add_prefix('Area norm ').columns)][valdatapd.index.isin(valpd.index)]
valMaxpd=valdatapd[valdatapd.columns.intersection(valpd.add_prefix('Maximo diario ').columns)][valdatapd.index.isin(valpd.index)]

# Load validation data with windows
valAreapd=pd.DataFrame()
valMaxpd=pd.DataFrame()

for i in range(maxbakcwindow):
    aux=valdatapd[valdatapd.columns.intersection(valpd.add_prefix('Area norm ').columns)][valdatapd.index.isin(valpd.index-timedelta(days=i))]
    aux.index=aux.index+timedelta(days=i)
    aux=aux.stack()
    valAreapd[i]=aux
    
    aux=valdatapd[valdatapd.columns.intersection(valpd.add_prefix('Maximo diario ').columns)][valdatapd.index.isin(valpd.index-timedelta(days=i))]
    aux.index=aux.index+timedelta(days=i)
    aux=aux.stack()
    valMaxpd[i]=aux


valArea=valAreapd.to_numpy()
valMax=valMaxpd.to_numpy()

# Data matrices
valerror=np.zeros(maxbakcwindow-minbakcwindow)
calerror=np.zeros(maxbakcwindow-minbakcwindow)

# Saving
sav=pd.DataFrame()

for bw in range(minbakcwindow,maxbakcwindow):
    calAreabw=calArea[:,:bw]
    calMaxbw=calMax[:,:bw]

    valAreabw=valArea[:,:bw]
    valMaxbw=valMax[:,:bw]
    

    #calArea=np.atleast_2d(calArea).T
    #calMax=np.atleast_2d(calMax).T

    matCal=np.asarray(np.append(calAreabw,calMaxbw,axis=1))
    matVal=np.asarray(np.append(valAreabw,valMaxbw,axis=1))
    
    cols=['area k-'+str(i) for i in range(bw)]
    cols.extend(['max k-'+str(i) for i in range(bw)])

    onesRow=np.full([np.shape(calAreabw)[0],1],1)
    calAreabw=np.asarray(np.append(calAreabw,onesRow,axis=1))
    calMaxbw=np.asarray(np.append(calMaxbw,onesRow,axis=1))
    matCal=np.asarray(np.append(matCal,onesRow,axis=1)) #area max 1

    
    cols.extend(['1'])

    onesRow=np.full([np.shape(valAreabw)[0],1],1)
    valAreabw=np.asarray(np.append(valAreabw,onesRow,axis=1))
    valMaxbw=np.asarray(np.append(valMaxbw,onesRow,axis=1))
    matVal=np.asarray(np.append(matVal,onesRow,axis=1)) #area max 1
    print('Calculating bAM with window '+str(bw)+':')
    # bArea=np.matmul(np.linalg.inv(np.matmul(calArea.T,calArea)),np.matmul(calArea.T,cal))
    # bMax=np.matmul(np.linalg.inv(np.matmul(calMax.T,calMax)),np.matmul(calMax.T,cal))
    bAM=np.matmul(np.linalg.inv(np.matmul(matCal.T,matCal)),np.matmul(matCal.T,cal))
    xcalc=np.matmul(matCal,bAM)
    print('bAM:')
    print(bAM)
    
    print('Average calibration square error:')
    
    calerror[bw-1]=np.sum(np.square(xcalc-cal))/np.size(cal)
    print(calerror[bw-1])

    # fig = plt.figure()
    # plt.scatter(range(cal),cal)
    # plt.scatter(range(xcalc),xcalc)
    # plt.xlabel('Sample')
    # plt.ylabel('Stress')
    # fig.suptitle('Window '+str(bw)+' calibration:',fontsize=16)

    fig = plt.figure()
    plt.scatter(cal,xcalc)
    plt.xlabel('Real stress')
    plt.ylabel('Stimated stress')
    fig.suptitle('Window '+str(bw)+' calibration:',fontsize=16)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # Axes3D.scatter(ax,maxVector,areavector,zs=cal)
    # Axes3D.scatter(ax,maxVector,areavector,zs=xcalc)

    print('Calculating validation with window '+str(bw)+':')

    xvalc=np.matmul(matVal,bAM)

    print('Average validation square error:')

    valerror[bw-1]=np.sum(np.square(xvalc[val>0]-val[val>0])/np.size(val))

    print(valerror[bw-1])
    # fig = plt.figure()
    # plt.scatter(datesval[[val>0][0].T[0]],val[val>0])
    # plt.scatter(datesval[[val>0][0].T[0]],xvalc[val>0])
    # plt.xlabel('Sample')
    # plt.ylabel('Stress')
    # fig.suptitle('Window '+str(bw)+' validation:',fontsize=16)
    fig = plt.figure()
    plt.scatter(val[val>0],xvalc[val>0])
    plt.xlabel('Real stress')
    plt.ylabel('Stimated stress')
    fig.suptitle('Window '+str(bw)+' validation:',fontsize=16)

    auxdf=pd.DataFrame(bAM.T, columns = cols)
    auxdf['backwards window']=bw
    auxdf['validation error']=valerror[bw-1]
    auxdf['calibration error']=calerror[bw-1]
    sav=pd.concat([sav,auxdf],ignore_index=True)
sav=sav.fillna(0)
sav.to_csv(savefile,index_label="index")

fig = plt.figure()
valplot=plt.plot(np.arange(minbakcwindow,maxbakcwindow),valerror)
calplot=plt.plot(np.arange(minbakcwindow,maxbakcwindow),calerror)
plt.xlabel('Backward window')
plt.ylabel('Error')
plt.legend(['Validation','Calibration'])
plt.show()