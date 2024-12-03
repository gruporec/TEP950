import pandas as pd
import matplotlib.pyplot as plt
import os

# I'm going to use this code to load the data from the IRNAS API and save it in a csv file that will be loaded in the classification script.
# In implementation, it will not be necessary to save it, but the classification script can be executed directly from the API data.

prevDays=60
saveFolder='implementation/db'
saveFile='IrnasData.csv'

# load data from http://gruporec.csic.es/Isadora/read_file.php?file=ISADORA3_zims.dat&days=60
url = 'http://gruporec.csic.es/Isadora/read_file.php?file=ISADORA3_zims.dat&days='+str(prevDays)

# load the data in JSON format
data = pd.read_json(url)

#set the index to the timestamp
data.set_index('TIMESTAMP', inplace=True)

# make sure the folder exists
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)
# save the data in a csv file
data.to_csv(saveFolder+'/'+saveFile)