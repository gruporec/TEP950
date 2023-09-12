import sklearn.datasets as skdata
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


dataset = skdata.load_digits()
#dato inicial
dini=0
#dato final
dfin=1797
#muestra el tamaño del dataset
print(dataset.data.shape)

# calcula el cuadrado más cercano a dfin-dini
n = np.ceil(np.sqrt(dfin-dini))
m=n
# calcula si se van a quedar filas o columnas vacías
print(n*m)
print(dfin-dini+n-1)
while n*m > dfin-dini+n-1:
    m = m-1
# crea una figura con un tamaño de m x n
fig, ax = plt.subplots(int(m), int(n), figsize=(10, 10))
#para los datos entre dini y dfin
for i in range(dini,dfin):
    #extrae las características del dato
    x_0 = dataset.data[i]
    # convierte la imagen a 8x8
    x_0 = np.reshape(x_0, [8, 8])
    # muestra la imagen en la posición correspondiente
    ax[int(i/n), int(i%n)].imshow(x_0, cmap='gray')
    # Oculta los ejes
    ax[int(i/n), int(i%n)].axis('off')
plt.show()