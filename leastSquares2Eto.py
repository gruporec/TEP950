# Import required packages
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Design variables
minbakcwindow=1
maxbakcwindow=16
savefile='lsqretotest.csv'

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
calEtopd=pd.DataFrame()

for i in range(maxbakcwindow):
    aux=caldatapd[caldatapd.columns.intersection(calpd.add_prefix('Area norm ').columns)][caldatapd.index.isin(calpd.index-timedelta(days=i))]
    numColumns=len(aux.columns)
    aux.index=aux.index+timedelta(days=i)
    aux=aux.stack()
    calAreapd[i]=aux
    
    aux=caldatapd[caldatapd.columns.intersection(calpd.add_prefix('Maximo diario ').columns)][caldatapd.index.isin(calpd.index-timedelta(days=i))]
    aux.index=aux.index+timedelta(days=i)
    aux=aux.stack()
    calMaxpd[i]=aux
    
    aux=caldatapd['ETo'][caldatapd.index.isin(calpd.index-timedelta(days=i))]
    aux.to_frame()
    aux=aux.iloc[np.repeat(np.arange(len(aux)), numColumns)]
    aux=aux.reset_index(drop=True)
    calEtopd[i]=aux
calEtopd.index=calAreapd.index

calArea=calAreapd.to_numpy()
calMax=calMaxpd.to_numpy()
calEto=calEtopd.to_numpy()

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
valEtopd=pd.DataFrame()

for i in range(maxbakcwindow):
    aux=valdatapd[valdatapd.columns.intersection(valpd.add_prefix('Area norm ').columns)][valdatapd.index.isin(valpd.index-timedelta(days=i))]
    numColumns=len(aux.columns)
    aux.index=aux.index+timedelta(days=i)
    aux=aux.stack()
    valAreapd[i]=aux
    
    aux=valdatapd[valdatapd.columns.intersection(valpd.add_prefix('Maximo diario ').columns)][valdatapd.index.isin(valpd.index-timedelta(days=i))]
    aux.index=aux.index+timedelta(days=i)
    aux=aux.stack()
    valMaxpd[i]=aux

    aux=valdatapd['ETo'][valdatapd.index.isin(valpd.index-timedelta(days=i))]
    aux.to_frame()
    aux=aux.iloc[np.repeat(np.arange(len(aux)), numColumns)]
    aux=aux.reset_index(drop=True)
    valEtopd[i]=aux
    
valEtopd.index=valAreapd.index

valArea=valAreapd.to_numpy()
valMax=valMaxpd.to_numpy()
valEto=valEtopd.to_numpy()

# Data matrices
valerror=np.zeros(maxbakcwindow-minbakcwindow)
calerror=np.zeros(maxbakcwindow-minbakcwindow)

# Saving
sav=pd.DataFrame()

for bw in range(minbakcwindow,maxbakcwindow):
    calAreabw=calArea[:,:bw]
    calMaxbw=calMax[:,:bw]
    calEtobw=calEto[:,:bw]

    valAreabw=valArea[:,:bw]
    valMaxbw=valMax[:,:bw]
    valEtobw=valEto[:,:bw]
    

    #calArea=np.atleast_2d(calArea).T
    #calMax=np.atleast_2d(calMax).T

    matCal=np.asarray(np.append(calAreabw,calMaxbw,axis=1))
    matVal=np.asarray(np.append(valAreabw,valMaxbw,axis=1))
    
    cols=['area k-'+str(i) for i in range(bw)]
    cols.extend(['max k-'+str(i) for i in range(bw)])
    
    matCal=np.asarray(np.append(matCal,calEtobw,axis=1))
    matVal=np.asarray(np.append(matVal,valEtobw,axis=1))
    
    cols.extend(['ETo k-'+str(i) for i in range(bw)])

    onesRow=np.full([np.shape(calAreabw)[0],1],1)
    calAreabw=np.asarray(np.append(calAreabw,onesRow,axis=1))
    calMaxbw=np.asarray(np.append(calMaxbw,onesRow,axis=1))
    matCal=np.asarray(np.append(matCal,onesRow,axis=1)) #area max ETo 1

    
    cols.extend(['1'])

    onesRow=np.full([np.shape(valAreabw)[0],1],1)
    valAreabw=np.asarray(np.append(valAreabw,onesRow,axis=1))
    valMaxbw=np.asarray(np.append(valMaxbw,onesRow,axis=1))
    matVal=np.asarray(np.append(matVal,onesRow,axis=1)) #area max ETo 1

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
    
    # print('Calibration failed ratio:')
    # xcalcest=np.rint(xcalc.clip(min=1,max=3))
    # calerrorest[bw-1]=np.sum(np.square(xcalcest-cal))/np.size(cal)
    # print(calerror[bw-1])

    fig = plt.figure()
    plt.scatter(range(cal.size),cal)
    plt.scatter(range(cal.size),xcalc)
    plt.xlabel('Sample')
    plt.ylabel('Stress')
    fig.suptitle('Window '+str(bw)+' validation:',fontsize=16)

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

    print(matVal)
    xvalc=np.matmul(matVal,bAM)

    print('Average validation square error:')

    valerror[bw-1]=np.sum(np.square(xvalc[val>0]-val[val>0])/np.size(val))

    print(valerror[bw-1])

    fig = plt.figure()
    plt.scatter(range(val[val>0].size),val[val>0])
    plt.scatter(range(val[val>0].size),xvalc[val>0])
    plt.xlabel('Sample')
    plt.ylabel('Stress')
    fig.suptitle('Window '+str(bw)+' validation:',fontsize=16)

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