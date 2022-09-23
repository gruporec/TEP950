import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
"""
Script para leer sensores Zim a partir de un archivo csv y representarlos
graficamente con interfaz interactiva que permita mostrar u ocultar cada sensor
"""
def on_pick(event):
    """
    Funcion para capturar el evento de pulsar en la leyenda y ocultar o mostrar esa grafica
    """
    legend = event.artist
    isVisible = legend.get_visible()

    graphs[legend].set_visible(not isVisible)
    legend.set_visible(not isVisible)

    fig.canvas.draw()

#Leemos en un dataframe los datos de los sensores Zim a partir de un csv:
zims=pd.read_csv("allZims2019.csv")
zimArr=[]
#Para graficar cada sensor con matplotlib necesitamos separarlos
# 1º convertimos el dataframe en un array. 
# Cada elemento del array representa un sensor, en forma de pareja de columnas (fecha, valor)
for col in range(1,len(zims.columns),2):
    #print(str(col)+": "+zims.columns[col])
    zimArr.append(zims.iloc[:,col:col+2])
#Creamos la figura que albergara las graficas:
fig=plt.figure('Sondas Zim 2019')
#y dentro de la figura se crea un subplot:
ax=fig.add_subplot(1,1,1)

zArr=[] #Array que albergará la referencia a cada linea 
zimArr2=[] #Vamos a modificar cada pareja de columnas a una sola columna 
#con los valores del sensor y convirtiendo la primera columna (convertida a tipo datetime) en indice
#Hay que hacer todo esto para que se dibuje correctamente con matplotlib
for zim in zimArr:
    #La primera columna la convertimos de String a datetime
    zim.iloc[:,0]=pd.to_datetime(zim.iloc[:,0], infer_datetime_format=True)
    #Convertimos la primera columna en un indice
    zim=zim.set_index(zim.columns[0])
    #Añadimos el sensor convertido al array
    zimArr2.append(zim)
    #dibujamos la linea correspondiente al sensor
    z,=ax.plot(zim.index,zim,label=zim.columns[0])
    #y la añadimos al array
    zArr.append(z) 
#Lo siguiente es necesario para crear una leyenda interactiva que permita
#mostrar u ocultar cada linea al pulsar en su leyenda
legend = plt.legend(loc='upper right')
lgs=legend.get_lines()
for lg in lgs:
    lg.set_picker(True)
    lg.set_pickradius(10)
graphs = {}
for i in range(0,len(lgs)):
    graphs[lgs[i]]=zArr[i]
plt.connect('pick_event',on_pick)
plt.show()