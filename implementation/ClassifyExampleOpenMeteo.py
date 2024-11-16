import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Add the implementation/ZIM directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..','implementation/ZIM'))

import functions as f  # Now you can import functions.py

#openmeteo has no data available for 2014 to 2019. I'll use the available data from IRNAS-CSIC to emulate it
# to simplify the script, I'll use the meteoraw database to get the data

#load the data from the IRNAS API (see implementation/loadApiData.py)
dataFile='implementation/db/IrnasData.csv'

# load the data from the openmeteo API (see implementation/loadOpenMeteoData.py). Data was saved so the limit of requests is not exceeded.
openmeteoFile='implementation/db/openmeteo_hourly_data.csv'
# daily data doesn't seem to work so I'll use the hourly radiation data to recover the sun hours

# save folder for the classification results
saveFolder='implementation/results'

# save file for the classification results
saveFile='classificationResults.csv'

# where to load the classifier from
classifierFolder='classifiers/OM'
classifierFile='classifier.clas'

#what sensor data to use
sensors=['1','2']

# Read data file. use the first column as index.
origdata = pd.read_csv(dataFile, sep=',', decimal='.', index_col=[0])

# Read openmeteo data. Format of request should be the same as in loadOpenMeteoData.py and return date, temperature_2m, relative_humidity_2m and terrestrial_radiation
openmeteodata = pd.read_csv(openmeteoFile, sep=',', decimal='.', index_col=[0])

# get the full name of each sensor as zim_Avg([sensor]) where [sensor] is the sensor number and place it in a list
fullnameSensors = ['zim_Avg(' + sensor + ')' for sensor in sensors]

# select the sensors from the data
origdata = origdata[fullnameSensors]

# remove rows with NaN values
origdata = origdata.dropna()

# separate the TIMESTAMP and the date columns from the data into day and time columns
origdata['date'] = pd.to_datetime(origdata.index)
origdata['day'] = origdata['date'].dt.date
origdata['time'] = origdata['date'].dt.time

openmeteodata['date'] = pd.to_datetime(openmeteodata.index)
openmeteodata['day'] = openmeteodata['date'].dt.date
openmeteodata['time'] = openmeteodata['date'].dt.time

# drop the date column
origdata = origdata.drop(columns=['date'])
openmeteodata = openmeteodata.drop(columns=['date'])

#drop the days that are not in both datasets
origdata = origdata.loc[origdata['day'].isin(openmeteodata['day'])]
openmeteodata = openmeteodata.loc[openmeteodata['day'].isin(origdata['day'])]

# separate the temperature and radiation data from the openmeteo data into separate dataframes, keeping the day and time columns
openmeteo_radiation = openmeteodata[['day','time','terrestrial_radiation']]
openmeteo_temperature = openmeteodata[['day','time','temperature_2m']]
openmeteo_humidity = openmeteodata[['day','time','relative_humidity_2m']]

# set the day and time columns as the index for the openmeteo data
openmeteo_radiation.set_index(['day','time'], inplace=True)
openmeteo_temperature.set_index(['day','time'], inplace=True)
openmeteo_humidity.set_index(['day','time'], inplace=True)

# unstack the time index to columns for the openmeteo data and leave the values of the only column as the values of the dataframe
openmeteo_radiation = openmeteo_radiation.unstack('time').droplevel(0, axis=1)
openmeteo_temperature = openmeteo_temperature.unstack('time').droplevel(0, axis=1)
openmeteo_humidity = openmeteo_humidity.unstack('time').droplevel(0, axis=1)

# create a sunrise and sunset dataset from the radiation data: date as index, sunrise and sunset as columns, and the values as the time of the day for the first and last non-zero radiation value
sunriseTime = openmeteo_radiation.apply(lambda x: x[x>0].index[0], axis=1)
sunsetTime = openmeteo_radiation.apply(lambda x: x[x>0].index[-1], axis=1)

# process the meteorological data from the openmeteo data using the same method as in TrainEmulateOpenMeteo.py
processedHumidity = f.processRawMeteoData(openmeteo_humidity, sunriseTime, sunsetTime)
processedTemperature = f.processRawMeteoData(openmeteo_temperature, sunriseTime, sunsetTime)

# load the classifier from a file
clf = f.load(classifierFolder+'/'+classifierFile)

# create a dataframe to store the classification results, with a column for each sensor and a row for each day
classificationResults = pd.DataFrame(index=origdata['day'].unique(), columns=fullnameSensors)

# for each sensor in the data
for sensor in fullnameSensors:
    # get the sensor data keeping the day and time columns
    sensorData = origdata[['day','time',sensor]]

    # set the day and time columns as the index for the sensor data
    sensorData.set_index(['day','time'], inplace=True)

    # remove accidental duplicates
    sensorData = sensorData[~sensorData.index.duplicated()]

    # unstack the time index to columns for the sensor data and leave the values of the only column as the values of the dataframe
    sensorData = sensorData.unstack('time').droplevel(0, axis=1)

    # process the raw ZIM data using the sunrise and sunset times and the number of samples and filter window
    processedData = f.processRawZIMData(sensorData, sunriseTime, sunsetTime)
    
    # combine the processed ZIM data with the processed meteorological data
    processedData = f.combineZIMMeteoData(processedData, processedHumidity, processedTemperature)

    # predict the stress level for the processed data
    classificationResults[sensor] = f.predict(clf, processedData)

# add one to the stress level to make it 1-based as IRNAS usually does
classificationResults = classificationResults + 1

# make sure the folder exists
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)
# save the classification results to a file
classificationResults.to_csv(saveFolder+'/'+saveFile)
print(classificationResults)
