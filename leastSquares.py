# Import required packages
from math import inf
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load calibration output
calpdraw=pd.read_csv("calibracion.csv")
calpd=calpdraw.set_index('Fecha')

# Load calibration input
caldatapd=pd.read_csv("estadisticosDiarios.csv")
caldatapd=caldatapd.set_index('Fecha')
calAreapd=caldatapd
calAreapd=caldatapd[(caldatapd.columns) & (calpd.add_prefix('Area norm ').columns)][caldatapd.index.isin(calpd.index)]
calMaxpd=caldatapd[(caldatapd.columns) & (calpd.add_prefix('Maximo diario ').columns)][caldatapd.index.isin(calpd.index)]

# Load validation output
valpdraw=pd.read_csv("validacion.csv")
valpdraw.fillna(0,inplace=True)
valpd=valpdraw.set_index('Fecha')

# Load validation input
valdatapd=pd.read_csv("estadisticosDiarios.csv")
valdatapd.replace([np.inf, -np.inf], 0, inplace=True)
valdatapd.fillna(0,inplace=True)
valdatapd.set_index('Fecha',inplace=True)
valAreapd=valdatapd[(valdatapd.columns) & (valpd.add_prefix('Area norm ').columns)][valdatapd.index.isin(valpd.index)]
valMaxpd=valdatapd[(valdatapd.columns) & (valpd.add_prefix('Maximo diario ').columns)][valdatapd.index.isin(valpd.index)]


# Data matrices
minbakcwindow=1
maxbakcwindow=16

valerror=np.zeros(maxbakcwindow-minbakcwindow)
calerror=np.zeros(maxbakcwindow-minbakcwindow)

calAreapd.reset_index(inplace=True, drop=True)
valAreapd.reset_index(inplace=True, drop=True)
arraylen=calAreapd.shape[0]
arraylenval=valAreapd.shape[0]
for bw in range(minbakcwindow,maxbakcwindow):
    cal=calpd.iloc[bw:,0].append(calpd.iloc[bw:,1]).append(calpd.iloc[bw:,2]).append(calpd.iloc[bw:,3])
    dates=cal.index
    calArea=calAreapd.iloc[bw:arraylen,0].append(calAreapd.iloc[bw:arraylen,1]).append(calAreapd.iloc[bw:arraylen,2]).append(calAreapd.iloc[bw:arraylen,3])
    calArea.reset_index(inplace=True, drop=True)
    calMax=calMaxpd.iloc[bw:,0].append(calMaxpd.iloc[bw:,1]).append(calMaxpd.iloc[bw:,2]).append(calMaxpd.iloc[bw:,3])
    calMax.reset_index(inplace=True, drop=True)
    for i in range(1,1+bw):
        conArea=calAreapd.iloc[bw-i:arraylen-i,0].append(calAreapd.iloc[bw-i:arraylen-i,1]).append(calAreapd.iloc[bw-i:arraylen-i,2]).append(calAreapd.iloc[bw-i:arraylen-i,3])
        conArea.reset_index(inplace=True, drop=True)
        calArea=pd.concat([calArea,conArea],axis=1,ignore_index=True)

        conMax=calMaxpd.iloc[bw-i:arraylen-i,0].append(calMaxpd.iloc[bw-i:arraylen-i,1]).append(calMaxpd.iloc[bw-i:arraylen-i,2]).append(calMaxpd.iloc[bw-i:arraylen-i,3])
        conMax.reset_index(inplace=True, drop=True)
        calMax=pd.concat([calMax,conMax],axis=1,ignore_index=True)
    cal.reset_index(inplace=True, drop=True)
    calArea.reset_index(inplace=True, drop=True)
    calMax.reset_index(inplace=True, drop=True)

    val=valpd.iloc[bw:,0].append(valpd.iloc[bw:,1]).append(valpd.iloc[bw:,2]).append(valpd.iloc[bw:,3]).append(valpd.iloc[bw:,4]).append(valpd.iloc[bw:,5])


    datesval=val.index
    valArea=valAreapd.iloc[bw:,0].append(valAreapd.iloc[bw:,1]).append(valAreapd.iloc[bw:,2]).append(valAreapd.iloc[bw:,3]).append(valAreapd.iloc[bw:,4]).append(valAreapd.iloc[bw:,5])
    valMax=valMaxpd.iloc[bw:,0].append(valMaxpd.iloc[bw:,1]).append(valMaxpd.iloc[bw:,2]).append(valMaxpd.iloc[bw:,3]).append(valMaxpd.iloc[bw:,4]).append(valMaxpd.iloc[bw:,5])

    valArea.reset_index(inplace=True, drop=True)
    valMax.reset_index(inplace=True, drop=True)
    for i in range(1,1+bw):
        conArea=valAreapd.iloc[bw-i:arraylenval-i,0].append(valAreapd.iloc[bw-i:arraylenval-i,1]).append(valAreapd.iloc[bw-i:arraylenval-i,2]).append(valAreapd.iloc[bw-i:arraylenval-i,3]).append(valAreapd.iloc[bw-i:arraylenval-i,4]).append(valAreapd.iloc[bw-i:arraylenval-i,5])
        
        conArea.reset_index(inplace=True, drop=True)
        valArea=pd.concat([valArea,conArea],axis=1,ignore_index=True)

        conMax=valMaxpd.iloc[bw-i:arraylenval-i,0].append(valMaxpd.iloc[bw-i:arraylenval-i,1]).append(valMaxpd.iloc[bw-i:arraylenval-i,2]).append(valMaxpd.iloc[bw-i:arraylenval-i,3]).append(valMaxpd.iloc[bw-i:arraylenval-i,4]).append(valMaxpd.iloc[bw-i:arraylenval-i,5])
        
        conMax.reset_index(inplace=True, drop=True)
        valMax=pd.concat([valMax,conMax],axis=1,ignore_index=True)
    val.reset_index(inplace=True, drop=True)
    valArea.reset_index(inplace=True, drop=True)
    valMax.reset_index(inplace=True, drop=True)

    cal=cal.to_numpy()
    calArea=calArea.to_numpy()
    calMax=calMax.to_numpy()

    val=val.to_numpy()
    valArea=valArea.to_numpy()
    valMax=valMax.to_numpy()

    cal=np.atleast_2d(cal).T
    val=np.atleast_2d(val).T

    #calArea=np.atleast_2d(calArea).T
    #calMax=np.atleast_2d(calMax).T

    matCal=np.asarray(np.append(calArea,calMax,axis=1))
    matVal=np.asarray(np.append(valArea,valMax,axis=1))

    onesRow=np.full([np.shape(calArea)[0],1],1)
    calArea=np.asarray(np.append(calArea,onesRow,axis=1))
    calMax=np.asarray(np.append(calMax,onesRow,axis=1))
    matCal=np.asarray(np.append(matCal,onesRow,axis=1)) #area max 1

    onesRow=np.full([np.shape(valArea)[0],1],1)
    valArea=np.asarray(np.append(valArea,onesRow,axis=1))
    valMax=np.asarray(np.append(valMax,onesRow,axis=1))
    matVal=np.asarray(np.append(matVal,onesRow,axis=1)) #area max 1



    print('Calculating bAM with window '+str(bw)+':')
    try:
        if bw==1:
            print(matCal)
            print(np.matmul(matCal.T,matCal))
            exit()
        # bArea=np.matmul(np.linalg.inv(np.matmul(calArea.T,calArea)),np.matmul(calArea.T,cal))
        # bMax=np.matmul(np.linalg.inv(np.matmul(calMax.T,calMax)),np.matmul(calMax.T,cal))
        bAM=np.matmul(np.linalg.inv(np.matmul(matCal.T,matCal)),np.matmul(matCal.T,cal))

        xcalc=np.matmul(matCal,bAM)


        print('bAM:')
        print(bAM)
        fig = plt.figure()
        
        print('Average error:')
        
        print('Average error:')
        calerror[bw-1]=np.sum(np.square(xcalc-cal))/np.size(cal)
        print(calerror[bw-1])
        plt.scatter(dates,cal)
        plt.scatter(dates,xcalc)
        plt.xlabel('Sample')
        plt.ylabel('Stress')
        fig.suptitle('Window '+str(bw)+' calibration:',fontsize=16)

        fig = plt.figure()
        plt.scatter(cal,xcalc)
        plt.xlabel('Real stress')
        plt.ylabel('Stimated stress')
        fig.suptitle('Window '+str(bw)+' calibration:',fontsize=16)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # Axes3D.scatter(ax,maxVector,areavector,zs=cal)
        # Axes3D.scatter(ax,maxVector,areavector,zs=xcalc)0
        print('Calculating validation with window '+str(bw)+':')

        xvalc=np.matmul(matVal,bAM)

        print('Average error:')
        valerror[bw-1]=np.sum(np.square(xvalc[val>0]-val[val>0])/np.size(val))
        print(valerror[bw-1])

        fig = plt.figure()
        plt.scatter(datesval[[val>0][0].T[0]],val[val>0])
        plt.scatter(datesval[[val>0][0].T[0]],xvalc[val>0])
        plt.xlabel('Sample')
        plt.ylabel('Stress')
        fig.suptitle('Window '+str(bw)+' validation:',fontsize=16)

        fig = plt.figure()
        plt.scatter(val[val>0],xvalc[val>0])
        plt.xlabel('Real stress')
        plt.ylabel('Stimated stress')
        fig.suptitle('Window '+str(bw)+' validation:',fontsize=16)
    except Exception as e:
        print(e)

fig = plt.figure()
valplot=plt.plot(np.arange(minbakcwindow,maxbakcwindow),valerror)
calplot=plt.plot(np.arange(minbakcwindow,maxbakcwindow),calerror)
plt.xlabel('Backward window')
plt.ylabel('Error')
plt.legend([valplot,calplot],['Validation','Calibration'])
plt.show()