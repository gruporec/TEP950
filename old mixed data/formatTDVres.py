import pandas as pd

orig_file='ignore/resultadosTDV/PCA_LDA_results_temp_char.csv'
res_file='ignore/resultadosTDV/PCA_LDA_results_temp_char_form.csv'

#carga el fichero original
df=pd.read_csv(orig_file)

#elimina entradas duplicadas
df.drop_duplicates(inplace=True)

#convierte el dataframe en multiindex con los índices year_test, PCA_components, TDV_days, meteo_days, TDV_sampling y meteo_sampling
df.set_index(['year_test','PCA_components','TDV_days','meteo_days','TDV_sampling','meteo_sampling'],inplace=True)

#elimina índices duplicados
df=df[~df.index.duplicated(keep='first')]

#unstackea year_test
df=df.unstack(level=0)

#añade una columna con la media de las columnas train_score del segundo nivel del multiindex
df['train_score_mean']=df['train_score'].mean(axis=1)

#añade una columna con la media de las columnas test_score del segundo nivel del multiindex
df['test_score_mean']=df['test_score'].mean(axis=1)

#combina los dos niveles del multiindex de columnas en uno solo
df.columns=df.columns.map('{0[0]} {0[1]}'.format)


#guarda el resultado
df.to_csv(res_file)