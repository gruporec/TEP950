import pandas as pd

file='classifierTesting/dbv2/0.csv'

#load data
df = pd.read_csv(file)

#count data per class
print(df['Y'].value_counts())