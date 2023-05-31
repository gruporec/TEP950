import pandas as pd
import numpy as np
import isadoralib as isl
import multiprocessing as mp
import os
import matplotlib
import matplotlib.pyplot as plt


# grafica una variable sum_i(a*x^(b^i)), siendo x el valor max-max, a el peso (1 para positivos, variable para negativos)
# y b el exponente (menor que 1, menor peso a valores antiguos)
# frente al número de días con x negativo, asignando colores según la clase

# Define una función que por cada valor entre 1 y days, devuelve un dataframe con los valores de days días antes
def get_prev_days(df, days):
    #crea un dataframe vacío
    temp_df = df.copy()
    for i in range(1,days,1):
        #copia df en un dataframe temporal
        df_temp = df.copy()
        #añade i a la columna Carac
        df_temp['Carac'] = i
        #añade i días al índice
        df_temp.index = df_temp.index + pd.Timedelta(days=i)
        #añade el dataframe temporal al dataframe de dias anteriores
        temp_df = temp_df.append(df_temp)
    return temp_df

def process_el(args):
    (weight,exp,count_w,tdv_tests_max_inc_prev,data_tests,years,days,save_folder)=args

            #Crea una lista que cuenta el número de elementos con x negativo para cada día y columna
    tdv_tests_max_inc_prev_neg = []
    for i in range(len(tdv_tests_max_inc_prev)):
        tdv_temp=tdv_tests_max_inc_prev[i].copy()

        #intercambia el índice de columnas y el segundo índice de filas
        tdv_temp = tdv_temp.stack(0).unstack(1)

        # iguala los valores negativos a 1 y el resto a 0
        tdv_temp[tdv_temp>=0] = 0
        tdv_temp[tdv_temp<0] = 1

        #multiplica cada columna por el peso elevado a la potencia de la columna
        for col in tdv_temp.columns:
            tdv_temp[col] = tdv_temp[col]*count_w**(col)
        
        #suma los elementos negativos por fila y vuelve a stackear el índice
        tdv_tests_max_inc_prev_neg.append(tdv_temp.sum(axis=1).unstack(1))

    for i in range(len(tdv_tests_max_inc_prev)):
        tdv_temp=tdv_tests_max_inc_prev[i].copy()

        #intercambia el índice de columnas y el segundo índice de filas
        tdv_temp = tdv_temp.stack(0).unstack(1)

        #multiplica los valores negativos por el peso
        tdv_temp[tdv_temp<0] = tdv_temp[tdv_temp<0]*weight

        #por cada columna
        for col in tdv_temp.columns:
            #eleva los valores a la exp elevado a la potencia de la columna más uno
            tdv_temp[col] = tdv_temp[col].pow(exp**(col+1))
        #suma los valores de cada fila y vuelve a unstakear el segundo índice de filas
        tdv_sum=tdv_temp.sum(axis=1).unstack(1)

        #steackea tdv_sum, tdv_tests_max_inc_prev_neg and data_tests
        tdv_sum_stacked = tdv_sum.stack(0)
        tdv_tests_max_inc_prev_neg_stacked = tdv_tests_max_inc_prev_neg[i].stack(0)
        data_tests_stacked = data_tests[i].stack(0)

        #encuentra los índices comunes entre los tres
        common_index = tdv_sum_stacked.index.intersection(tdv_tests_max_inc_prev_neg_stacked.index).intersection(data_tests_stacked.index)

        #recorta los tres dataframes a los índices comunes
        tdv_sum_stacked = tdv_sum_stacked.loc[common_index]
        tdv_tests_max_inc_prev_neg_stacked = tdv_tests_max_inc_prev_neg_stacked.loc[common_index]
        data_tests_stacked = data_tests_stacked.loc[common_index]

        colors = ["red", "blue", "green"]

        #grafica tdv_sum_stacked frente a tdv_tests_max_inc_prev_neg_stacked, asignando colores según data_tests_stacked
        plt.scatter(tdv_sum_stacked, tdv_tests_max_inc_prev_neg_stacked, c=data_tests_stacked.apply(lambda x: colors[int(x-1)]), marker=',', s=0.1)
        plt.xlabel("Suma ponderada de los valores de los últimos "+str(days)+" días")
        plt.ylabel("Dias negativos en los últimos "+str(days)+" días")
        plt.title("Peso de la suma: "+str(weight)+", Exponente: "+str(exp)+",\nPeso de los días negativos: "+str(count_w)+", Año: "+str(years[i]))
        plt.savefig(save_folder+"CompWMM_"+str(weight)+"_"+str(exp)+"_"+str(count_w)+"_"+str(years[i])+".png")
        #cierra la figura
        plt.close()

        
        
    

if __name__=='__main__':
    years = ["2014","2015","2016","2019"]
    #usa agg backend para que no intente abrir ventanas
    matplotlib.use("Agg")

    # valores iniciales de los parámetros
    #PCA_components = 4
    TDV_days = [1,2,3,4,5,6,7]
    Neg_weight = [0.5,1,1.5,2]
    Time_exp = [0.1,0.5,1]
    Neg_count_w = [0.25,0.5,1,2]

    save_folder = 'ignore/resultadosTDV/batch/CompWMM/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    #crea dos listas vacías para los datos
    tdv_tests=[]
    data_tests=[]
    #por cada año de test 
    for year_test in years:
        if not os.path.isfile("rawDiarios"+year_test+".csv") or not os.path.isfile("rawMinutales"+year_test+".csv"):
            os.system("python3 cargaRaw.py")
        #carga los datos de test
        tdv_test,ltp_test,meteo_test,data_test=isl.cargaDatosTDV(year_test,"rht")
        #añade los datos de test a las listas
        tdv_tests.append(tdv_test.copy())
        data_tests.append(data_test.copy())

    # obtiene el máximo de tdv de cada día
    tdv_tests_max = []
    for i in range(len(tdv_tests)):
        tdv_tests_max.append(tdv_tests[i].groupby(tdv_tests[i].index.date).max())

    # obtiene el incremento del máximo respecto al día anterior de cada día de test
    tdv_tests_max_inc = []
    for i in range(len(tdv_tests_max)):
        tdv_tests_max_inc.append(tdv_tests_max[i].diff())
        # elimina los valores nulos
        tdv_tests_max_inc[i] = tdv_tests_max_inc[i].dropna()
    
    #añade la columna Carac de valor max_max a cada día de test
    for i in range(len(tdv_tests_max_inc)):
        tdv_tests_max_inc[i]['Carac'] = 0
    

    #por cada valor de días
    for days in TDV_days:
        tdv_tests_max_inc_prev = []
        #añade los días anteriores a los datos
        for i in range(len(tdv_tests_max_inc)):
            tdv_tests_max_inc_prev.append(get_prev_days(tdv_tests_max_inc[i], days))
        #convierte los índices a datetime
        for i in range(len(tdv_tests_max_inc_prev)):
            tdv_tests_max_inc_prev[i].index = pd.to_datetime(tdv_tests_max_inc_prev[i].index)
            #convierte el índice de datos a datetime
            data_tests[i].index = pd.to_datetime(data_tests[i].index)
            
        for i in range(len(tdv_tests_max_inc_prev)):
            #obtiene los índices comunes entre tdv_tests_max_inc_prev y data_tests
            common_index = tdv_tests_max_inc_prev[i].index.intersection(data_tests[i].index)
            #recorta tdv_tests_max_inc_prev para que sólo contenga índices comunes con data_tests
            tdv_tests_max_inc_prev[i] = tdv_tests_max_inc_prev[i].loc[common_index]
            #recorta data_tests para que sólo contenga índices comunes con tdv_tests_max_inc_prev
            data_tests[i] = data_tests[i].loc[common_index]

        for i in range(len(tdv_tests_max_inc_prev)):
            #añade la columna carac como un segundo nivel de índice
            tdv_tests_max_inc_prev[i].set_index('Carac',append=True,inplace=True)

        inputs=[]
        #por cada valor de peso
        for weight in Neg_weight:
            #por cada valor de exponente
            for exp in Time_exp:
                for count_w in Neg_count_w:
                    #añade el peso, el exponente y una copia de los datos a la lista de inputs
                    inputs.append([weight,exp,count_w,tdv_tests_max_inc_prev.copy(),data_tests.copy(),years,days,save_folder])
        
        #crea una pool de procesos con tantos procesos como años de test multiplicado por el número de combinaciones de parámetros
        pool = mp.Pool(processes=len(Neg_weight)*len(Time_exp))

        #lanza los procesos
        results = pool.map(process_el,inputs)

        #cierra la pool
        pool.close()



    # #crea una pool de procesos con tantos procesos como años de test
    # pool = mp.Pool(processes=len(years_test))

    # #lanza los procesos
    # results = pool.map(process_el,inputs)

    # #cierra la pool
    # pool.close()