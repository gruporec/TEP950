import pandas as pd

folder = 'ignore/resultadosTDV/batch/PCALDA/'
testsFile = folder + 'programmedTests3.csv'
resultsFile = folder + 'meta2.csv'

# Carga el archivo de tests
tests = pd.read_csv(testsFile)
# Carga el archivo de resultados
results = pd.read_csv(resultsFile)

# Asigna como índice la columna ID
tests = tests.set_index('ID')
results = results.set_index('ID')

# Combina los dos dataframes donde la ID es la misma
tests = pd.concat([tests, results], axis=1, join='inner')

# Por cada ID, carga el archivo del test
for index, row in tests.iterrows():
    testdata = pd.read_csv(folder + str(index) + '.csv')
    # encuentra la fila que tiene el mejor resultado (test_score es mayor)
    best = testdata.loc[testdata['test_score'].idxmax()]
    # Por cada columna del archivo del test, asigna el valor a la fila del test en una columna con el mismo nombre, excepto por la columna test_score
    for column in testdata:
        if column != 'test_score':
            tests.at[index, column] = best[column]

# Ordena los tests según la columna best acc, de mayor a menor
tests = tests.sort_values(by=[' best acc'], ascending=False)
# Guarda el archivo de tests
tests.to_csv(folder + 'TestResults.csv')