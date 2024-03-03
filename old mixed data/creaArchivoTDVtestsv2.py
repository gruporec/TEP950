import pandas as pd

#crea un dataframe con una columna por cada característica a considerar
df = pd.DataFrame(columns=['tdv_prev_max','tdv_prev_min','tdv_prev_diff','tdv_prev_inc','tdv_prev_max_norm','tdv_prev_min_norm','tdv_prev_diff_norm','tdv_prev_inc_norm','stress'])

#agrega filas al dataframe asignando valores booleanos hasta tener todas las combinaciones posibles utilizando un for anidado
for i in range(2):
    for j in range(2):
        for k in range(2):
            df = df.append({'tdv_prev_max':False,'tdv_prev_min':False,'tdv_prev_diff':bool(i),'tdv_prev_inc':bool(j),'tdv_prev_max_norm':False,'tdv_prev_min_norm':False,'tdv_prev_diff_norm':False,'tdv_prev_inc_norm':False,'stress':bool(k)},ignore_index=True)
#elimina la fila con todos los valores en False
df = df.drop([0])
#elimina la fila que solo contiene True en la columna 'stress'
df = df.drop([1])

#añade una columna 'done' con valores False
df['done'] = False

#añade una columna 'numChars' con el número de características activas
df['numChars'] = df['tdv_prev_max'].astype(int) + df['tdv_prev_min'].astype(int) + df['tdv_prev_diff'].astype(int) + df['tdv_prev_inc'].astype(int) + df['tdv_prev_max_norm'].astype(int) + df['tdv_prev_min_norm'].astype(int) + df['tdv_prev_diff_norm'].astype(int) + df['tdv_prev_inc_norm'].astype(int) + df['stress'].astype(int)

#ordena el dataframe por el número de características activas
df = df.sort_values(by=['numChars'])

#elimina la columna 'numChars'
df = df.drop(columns=['numChars'])

#llama ID al índice del dataframe
df.index.name = 'ID'
#guarda el dataframe en un archivo csv
df.to_csv('ignore/resultadosTDV/batch/programmedTests.csv',index=True)
print(df)