import pandas as pd
import sys

csv_file ='results/ZIM/AnalisysMixedBalanced-KNN-LDA-lSVM-RF-CNN-meteoraw-HC/AnalisysMixedBalanced-KNN-LDA-lSVM-RF-CNN-meteoraw-HC.csv'

df = pd.read_csv(csv_file)

# Round the values to 2 decimal places
df = df.round(2)

# Create a latex table from the csv file printing using 2 decimal places
tablestr = df.to_latex(index=False, float_format="%.2f")

print(tablestr)