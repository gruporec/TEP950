import pandas as pd

#crea un dataframe con una columna por cada característica a considerar
df = pd.DataFrame(columns=['max-max sens','max-min sens','stress','max-max cont', 'max-min control','max-min ratio','norm'])

#agrega filas al dataframe asignando valores booleanos hasta tener todas las combinaciones posibles utilizando un for anidado
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for n in range(2):
                        for o in range(2):
                            df = df.append({'max-max sens': bool(i),'max-min sens': bool(j),'stress': bool(k),'max-max cont': bool(l),'max-min control': bool(m),'max-min ratio': bool(n),'norm': bool(o)}, ignore_index=True)
#elimina la fila con todos los valores en False
df = df.drop([0])
#elimina la fila que solo contiene True en la columna 'stress'
df = df.drop([1])
#localiza las filas que tienen False en max-max sens y max-min sens a la vez y las elimina
df = df.drop(df[(df['max-max sens'] == False) & (df['max-min sens'] == False)].index)

#reinicia el índice del dataframe
df = df.reset_index(drop=True)

#añade una columna 'done' con valores False
df['done'] = False

#añade una columna 'numChars' con el número de características activas; la normalización no se considera
df['numChars'] = df['max-max sens'].astype(int) + df['max-min sens'].astype(int) + df['stress'].astype(int) + df['max-max cont'].astype(int) + df['max-min control'].astype(int) + df['max-min ratio'].astype(int)

#ordena el dataframe por el número de características activas
df = df.sort_values(by=['numChars'])

#elimina la columna 'numChars'
df = df.drop(columns=['numChars'])

#llama ID al índice del dataframe
df.index.name = 'ID'
#guarda el dataframe en un archivo csv
df.to_csv('ignore/resultadosTDV/batch/programmedTests3.csv',index=True)
print(df)