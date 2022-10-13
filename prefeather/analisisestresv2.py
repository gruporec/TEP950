# Import required packages
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import sys

#Datos disponibles para Control_1_1, Control_2_1, RDI45_1_2, RDI45_2_1, RDI30_2_1, RDI30_3_2
sens=5
forgettingFactor=0.8
window=5
maxZscore=4

tipos=pd.DataFrame(np.array([['Control', 1, 1],['Control', 2, 1], ['RDI45', 1, 2], ['RDI45', 2, 1], ['RDI30', 2, 1], ['RDI30', 3, 2]]),columns=['Tipo','numero','sensor']) #sensores disponibles

# dataframe de almacenamiento de resultados
res=pd.DataFrame(columns=['Estimador','Ecuación','Factor de olvido','Tamaño de ventana'])
# estimador como índice de res
res=res.set_index('Estimador')

# carga ecuaciones de estadísticos diarios
ecuaciones=pd.read_csv("estDiariosEcuaciones.csv")


# carga estadísticos diarios
estdataraw=pd.read_csv("estadisticosDiarios.csv")
estdataraw['Fecha'] = pd.to_datetime(estdataraw['Fecha'])
estdataraw=estdataraw.set_index('Fecha')
estdatami=estdataraw.copy()
estdatami.columns = pd.MultiIndex.from_tuples(estdatami.columns.str.split(' ').tolist())



valpdraw=pd.read_csv("validacion.csv")
valpdraw['Fecha'] = pd.to_datetime(valpdraw['Fecha'])
valpdraw=valpdraw.set_index('Fecha')
valpdraw=valpdraw.replace(-np.inf, np.nan)
valpdraw=valpdraw.replace(np.inf, np.nan)
valmean=valpdraw.mean()
valstd=valpdraw.std()
valpdraw=(valpdraw-valmean)/valstd

calpdraw=pd.read_csv("calibracion3.csv")
calpdraw['Fecha'] = pd.to_datetime(calpdraw['Fecha'])
calpdraw=calpdraw.set_index('Fecha')
calpdraw=calpdraw.replace(-np.inf, np.nan)
calpdraw=calpdraw.replace(np.inf, np.nan)
idx=pd.date_range(start=calpdraw.first_valid_index(), end=calpdraw.last_valid_index(), freq='D')
calpdraw=(calpdraw-valmean)/(valstd)
calpdraw=calpdraw.dropna(axis=1)
calpdraw = calpdraw.reindex(idx, fill_value=np.nan)


# bucle
for forgettingFactor in [0]:
    for window in [1]:
        for i in ecuaciones.columns:
            res=res.append(pd.Series([ecuaciones.loc[0,i],forgettingFactor,window],index=res.columns[0:3], name=i+'_'+str(forgettingFactor)+'_'+str(window)))
        for sens in range(7):
            if(sens<6):
                tipo=tipos.loc[sens]['Tipo']
                numero=tipos.loc[sens]['numero']
                sensor=tipos.loc[sens]['sensor']
                sensstr=str(tipo)+'_'+str(numero)+'_'+str(sensor)


                # guarda solo los estadísticos LTP que acaben en tipo_numero_sensor
                estdatapd=estdataraw.loc[:,estdataraw.columns.str.endswith(sensstr)].copy()

                valpd=valpdraw[["LTP "+tipo+"_"+numero+"_"+sensor]].copy()
                valpd=valpd.add_prefix("validación ")

                if "LTP "+tipo+"_"+numero+"_"+sensor in calpdraw.columns:
                    calpd=calpdraw[["LTP "+tipo+"_"+numero+"_"+sensor]].copy()
                else:
                    calpd=pd.DataFrame()
                calpd=calpd.add_prefix("calibración ")
            else:
                estdatapd=estdatami.sort_index(axis=0,level=1)

                valpd=valpdraw
                #valpd.columns=valpd.columns.str.replace('LTP ','')
                valpd=valpd.stack()
                valpd=valpd.sort_index(axis=0,level=1)

                sensstr='Global'

            estdatapd=estdatapd.replace(-np.inf, np.nan)
            estdatapd=estdatapd.replace(np.inf, np.nan)
            estdatapd=estdatapd[(np.abs(stats.zscore(estdatapd.dropna()))<maxZscore)] #elimina outliers


            #implementación del factor de olvido
            estdata=estdatapd.copy()
            for i in range(1,window+1):
                estdata=estdata.shift(1)*forgettingFactor
                estdatapd=estdatapd+estdata

            estdatapd=(estdatapd-estdatapd.mean())/(estdatapd.std())

            if(sens==6):
                estdatapd=estdatapd.stack()
                estdatapd.columns=estdatapd.columns.droplevel(0)
                estdatapd=estdatapd.add_prefix('LTP ')
                estdatapd.rename(mapper=lambda x: f'LTP {x}', axis='index', level=1, inplace=True)

            # correlación
            corrval=estdatapd.corrwith(valpd.squeeze(),drop=True)
            # añade los resultados al dataframe de resultados
            #print(corrval)
            for i in corrval.index:
                res.loc[i.replace(" "+sensstr,"")+'_'+str(forgettingFactor)+'_'+str(window),sensstr+" en validación"]=corrval[i].copy()
            if sens<6:
                corrcal=estdatapd.corrwith(calpd.squeeze(),drop=True)
                for i in corrcal.index:
                    res.loc[i.replace(" "+sensstr,"")+'_'+str(forgettingFactor)+'_'+str(window),sensstr+" en calibración"]=corrcal[i].copy()
            #print(res)
            if sens < 6:
                estdatapd.columns+=[" correlación="]
                estdatapd.columns+=[str(a) for a in corrval.tolist()]
                output=pd.DataFrame()
                output=pd.concat([estdatapd,valpd,calpd],axis=1)
            # print(output)

            # output.to_csv('test.csv')

                #fig drawing
                fig = px.line(title='Datos')
                for ind in output.columns:#len(df.columns)//2
                    fig.add_trace(go.Scatter(x=output.index, y=output[ind],
                        mode='lines',
                        name=ind))
                fig.write_html("graficasCorrelacion/visualizaSensores"+sensstr+"_olvido_"+str(forgettingFactor)+"_ventana_"+str(window)+"SO.html")


# guarda res en un archivo
res.to_csv("correlacionEstimadoresBase.csv")

#ToDo: correlación global
# output=pd.DataFrame()
    # output=pd.concat([estdatapd,valpd,calpd],axis=1)
    # # print(output)

    # # output.to_csv('test.csv')

    # #fig drawing
    # fig = px.line(title='Datos')
    # for ind in output.columns:#len(df.columns)//2
    #     fig.add_trace(go.Scatter(x=output.index, y=output[ind],
    #         mode='lines',
    #         name=ind))
    # #df.dtypes
    # fig.show()
    # fig.write_html("visualizaSensores"+tipo+"_"+numero+"_"+sensor+"_olvido_"+str(forgettingFactor)+"_ventana_"+str(window)+"SO.html")