import pandas as pd

file="classifierTesting\\results\\all.csv"

df = pd.read_csv(file)

# create a string to store the latex table
latex = df.to_latex(index=False)

# print the latex table
print(latex)