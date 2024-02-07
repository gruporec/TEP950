import pandas as pd
import os

# print the current directory
print("Current directory: "+os.getcwd())
# list available folders
print("Available folders:")
# list all folders in the current directory
folders=[f for f in os.listdir('classifierTesting') if os.path.isdir('classifierTesting/'+f)]

for fol in folders:
    print(fol)
# ask for the folder and file by console
folder=input("Folder: ")

# list available files
print("Available files:")
# list all files in the current directory
files=[f for f in os.listdir('classifierTesting/'+folder) if os.path.isfile('classifierTesting/'+folder+'/'+f)]
for fil in files:
    print(fil)
file=input("File: ")

path='classifierTesting/'+folder+'/'+file

#load data
df = pd.read_csv(path)

#count data per class
print(df['Y'].value_counts())