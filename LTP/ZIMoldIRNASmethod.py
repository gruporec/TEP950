import pandas as pd
import numpy as np
from datetime import time
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#add the path to the lib folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
# import the isadoralib library
import isadoralib as isl

year_datas=["2014","2015","2016","2019"]
sufix="rht"

#NOTE: saved data is already intepolated but unfiltered. original data was 1 sample per 5 minutes, interpolated to 1 sample per minute. Relevant for the rolling mean filter.

# create a blank dataframe to save characteristics data (X) and another for class data (Y). both are multiindex dataframes with indices "Fecha" and "Sensor"
savedfX = pd.DataFrame()
savedfY = pd.DataFrame()

# X coumns will correspond to the indicators I to VIII:
    # I: the slope of the linear regression of the values between the maximum and minimum Pp (slope1)
    # II: the slope of the linear regression of the Pp values between 16.15 GMT and the end of the day (slope2)
    # III: the moment of the day when started the biggest fall of Pp value (max1)
    # IV: the duration of the fall starting at max1 (pmax1)
    # V: the ratio between the beginning and the end of the fall of Pp values starting at max1 and lasting pmax1 (rat1)
    # VI: the moment of the day when started the second biggest fall of Pp value (max2)
    # VII: the duration of the fall starting at max2 (pmax2)
    # VIII: ratio between the beginning and the end of the fall on Pp values starting at max2 and lasting pmax2 (rat2)
# Where Pp is the turgor pressure (the loaded data)
# Name the columns accordingly
savedfX["slope1"] = np.nan
savedfX["slope2"] = np.nan
savedfX["max1"] = np.nan
savedfX["pmax1"] = np.nan
savedfX["rat1"] = np.nan
savedfX["max2"] = np.nan
savedfX["pmax2"] = np.nan
savedfX["rat2"] = np.nan


# add index columns to the dataframes
savedfX["Fecha"] = np.nan
savedfX["Sensor"] = np.nan
savedfX = savedfX.set_index(["Fecha","Sensor"])

savedfY["Y"] = np.nan
savedfY["Fecha"] = np.nan
savedfY["Sensor"] = np.nan
savedfY = savedfY.set_index(["Fecha","Sensor"])

for year_data in year_datas:
    # Load data
    tdv,ltp,meteo,valdatapd=isl.cargaDatos(year_data,sufix)

    # this method does not use meteo data

    # delete all NaN columns from ltp
    ltp = ltp.dropna(axis=1,how='all')
    # fill NaN values in ltp with the previous value
    ltp = ltp.ffill()
    # fill NaN values in ltp with the next value
    ltp = ltp.bfill()
    # apply a rolling mean filter to the data. Original filter was 20 samples, centered, with a sample per 5 minutes. The data is already interpolated to 1 sample per minute, so the filter will be 100 samples, centered.
    ltp = ltp.rolling(window=100,center=True).mean()

    # drop all NaN values
    ltp = ltp.dropna()

    # separate the index of the data into a date and a time column
    ltp["Hora"] = ltp.index.time
    ltp["Fecha"] = ltp.index.date

    #turn ltp into a multiindex dataframe
    ltp = ltp.set_index(["Fecha","Hora"])

    # remove all rows with dates that are not in valdatapd
    ltp = ltp.loc[valdatapd.index]

    # create a dataframe with the increments of ltp
    ltpdiff = ltp.diff()

    # convert the index of valdatapd to a datetime.date format
    valdatapd.index = pd.to_datetime(valdatapd.index).date

    # for each column (sensor) in the data
    for sensor in ltp.columns:
        print("year: ",year_data," sensor: ",sensor)

        # make sure the sensor is in valdatapd
        if sensor not in valdatapd.columns:
            print("sensor not in valdatapd, skipping")
            continue
        # slice ltp and ltpdiff to only include the data for the current sensor
        ltp_sensor = ltp[sensor]
        ltpdiff_sensor = ltpdiff[sensor]

        # slice valdatapd to only include the data for the current sensor
        valdatapd_sensor = valdatapd[sensor]

        #for each date in the data
        for day in ltp_sensor.index.get_level_values(0).unique():

            # make sure there is data in valdatapd for the current day and sensor
            if day not in valdatapd_sensor.index:
                print("no data in valdatapd for day: ",day,", skipping")
                continue
            
            #slice the data to only include the current day
            ltp_day = ltp_sensor.loc[day]

            # make sure there is data in ltp_day
            if ltp_day.empty:
                print("no data in ltp_day for day: ",day,", skipping")
                continue

            #add valdatapd to Y according to the date and sensor
            savedfY.loc[(day,sensor),"Y"] = valdatapd_sensor.loc[day]

            
            # convert the index to a numeric value
            ltp_day.index = [time.hour + time.minute/60 for time in ltp_day.index]

            # calculate the slope of the linear regression of the values between the maximum and minimum Pp (slope1)
            # find the maximum and minimum values of the data
            maxPp = ltp_day.idxmax()
            minPp = ltp_day.idxmin()
            # slice the data to only include the values between the maximum and minimum
            # if the minimum is before the maximum
            if minPp < maxPp:
                minmaxcrop = ltp_day.loc[minPp:maxPp]
            # if the maximum is before the minimum
            else:
                minmaxcrop = ltp_day.loc[maxPp:minPp]
            slope1 = np.polyfit(minmaxcrop.index, minmaxcrop.values, 1)[0]

            # add slope1 to X
            savedfX.loc[(day,sensor),"slope1"] = slope1

            # calculate the slope of the linear regression of the Pp values between 16.15 GMT and the end of the day (slope2)
            # find the index of the time 16:15
            time1615 = ltp_day.index.get_loc(16.25)
            # slice the data to only include the values after 16:15
            ltp1615 = ltp_day.iloc[time1615:]
            slope2 = np.polyfit(ltp1615.index, ltp1615.values, 1)[0]

            # add slope2 to X
            savedfX.loc[(day,sensor),"slope2"] = slope2

            # calculate the moment of the day when started the biggest fall of Pp value (max1)
            # get the times of the local maximums and minimums of the data as lists
            maxs = []
            mins = []

            #check the first value of the data, if higher than the next value, it is a local maximum
            if ltp_day.iloc[0] >= ltp_day.iloc[1]:
                maxs.append(ltp_day.index[0])
            else:
                mins.append(ltp_day.index[0])
            # for each value in the data
            for i in range(1,len(ltp_day)-1):
                # if the value is greater than the previous and the next value, it is a local maximum
                if ltp_day.iloc[i] >= ltp_day.iloc[i-1] and ltp_day.iloc[i] > ltp_day.iloc[i+1]:
                    maxs.append(ltp_day.index[i])
                # if the value is smaller than the previous and the next value, it is a local minimum
                if ltp_day.iloc[i] <= ltp_day.iloc[i-1] and ltp_day.iloc[i] < ltp_day.iloc[i+1]:
                    mins.append(ltp_day.index[i])
            #check the last value of the data, if higher than the previous value, it is a local maximum
            if ltp_day.iloc[-1] >= ltp_day.iloc[-2]:
                maxs.append(ltp_day.index[-1])
            else:
                mins.append(ltp_day.index[-1])
            
            # merge the lists of maximums and minimums
            lims = maxs + mins
            # sort the list of maximums and minimums
            lims.sort()
            # create a list of the differences between the values in ltp_day corresponding to the maximums and minimums
            limsdiff = [ltp_day.loc[lims[i+1]] - ltp_day.loc[lims[i]] for i in range(len(lims)-1)]
            
            # find the index of the greatest negative difference in limsdiff and the corresponding limit in lims
            maxPp = lims[np.argmin(limsdiff)]

            # add max1 to X
            savedfX.loc[(day,sensor),"max1"] = maxPp

            # calculate the duration of the fall starting at max1 (pmax1)
            # find the next limit after maxPp
            nextlim = lims[np.argmin(limsdiff)+1]

            #find the difference in time between maxPp and nextlim
            pmax1 = nextlim - maxPp

            # add pmax1 to X
            savedfX.loc[(day,sensor),"pmax1"] = pmax1

            # calculate the ratio between the beginning and the end of the fall of Pp values starting at max1 and lasting pmax1 (rat1)
            # find the value of ltp_day at maxPp
            maxPpval = ltp_day.loc[maxPp]
            # find the value of ltp_day at nextlim
            nextlimval = ltp_day.loc[nextlim]

            # calculate the ratio between the values
            rat1 = maxPpval/nextlimval

            # add rat1 to X
            savedfX.loc[(day,sensor),"rat1"] = rat1

            # calculate the moment of the day when started the second biggest fall of Pp value (max2)
            # find the index of the second greatest negative difference in limsdiff and the corresponding limit in lims
            maxPp2 = lims[np.argsort(limsdiff)[1]]
            # print(np.argsort(limsdiff))
            # print(limsdiff)

            # add max2 to X
            savedfX.loc[(day,sensor),"max2"] = maxPp2

            # calculate the duration of the fall starting at max2 (pmax2)
            # find the next limit after maxPp2
            nextlim2 = lims[np.argsort(limsdiff)[1]+1]

            #find the difference in time between maxPp2 and nextlim2
            pmax2 = nextlim2 - maxPp2

            # add pmax2 to X
            savedfX.loc[(day,sensor),"pmax2"] = pmax2

            # calculate the ratio between the beginning and the end of the fall on Pp values starting at max2 and lasting pmax2 (rat2)
            # find the value of ltp_day at maxPp2
            maxPpval2 = ltp_day.loc[maxPp2]
            # find the value of ltp_day at nextlim2
            nextlimval2 = ltp_day.loc[nextlim2]

            # calculate the ratio between the values
            rat2 = maxPpval2/nextlimval2

            # add rat2 to X
            savedfX.loc[(day,sensor),"rat2"] = rat2


            # #plot the data for the day
            # plt.plot(ltp_day.index, ltp_day.values)
            # #plot the crop for slope1 and a line with slope1 starting at the same point as the data is at the minimum
            # #plt.plot(minmaxcrop.index, minmaxcrop.values)
            # plt.plot(minmaxcrop.index, slope1*minmaxcrop.index + minmaxcrop.loc[minPp] - slope1*minPp)
            # #plot the crop for slope2 and a line with slope2 starting at 16:15 in the same point as the data is at 16:15
            # #plt.plot(ltp1615.index, ltp1615.values)
            # plt.plot(ltp1615.index, slope2*ltp1615.index + ltp1615.loc[16.25] - slope2*16.25)
            # #plot the lims
            # #plt.scatter(lims, [ltp_day.loc[lim] for lim in lims])
            # #plot a vertical line at each lim with the corresponding limsdiff, except for the last lim, starting at the y value of ltp_day at the lim and ending at y of ltp_day + limsdiff
            # #for i in range(len(lims)-1):
            # #    plt.plot([lims[i],lims[i]], [ltp_day.loc[lims[i]],ltp_day.loc[lims[i]]+limsdiff[i]], color='red')
            # # plot maxPp
            # plt.scatter(maxPp, ltp_day.loc[maxPp])
            # # plot a horizontal line starting at maxPp and with a length of pmax1
            # plt.plot([maxPp,maxPp+pmax1],[ltp_day.loc[maxPp],ltp_day.loc[maxPp]])
            # # plot maxPp2
            # plt.scatter(maxPp2, ltp_day.loc[maxPp2])
            # # plot a horizontal line starting at maxPp2 and with a length of pmax2
            # plt.plot([maxPp2,maxPp2+pmax2],[ltp_day.loc[maxPp2],ltp_day.loc[maxPp2]])

            # # add a legend
            # #plt.legend(["Pp","slope1","slope2","max1","pmax1","max2","pmax2"])

            # print("slope1: ",slope1)
            # print("slope2: ",slope2)
            # print("max1: ",maxPp)
            # print("pmax1: ",pmax1)
            # print("rat1: ",rat1)
            # print("max2: ",maxPp2)
            # print("pmax2: ",pmax2)
            # print("rat2: ",rat2)

            # plt.show()
            # sys.exit()

# swap the levels of the multiindex
savedfX = savedfX.swaplevel()
savedfY = savedfY.swaplevel()

# add Y to X as a column
savedfX["Y"] = savedfY["Y"]

# save the data
# create a string with the last two digits of each year in year_datas
year_datas_str = ''.join(year_datas[-2:] for year_datas in year_datas)
savedfX.to_csv('db\ZIMdb'+year_datas_str+'oldIRNAS.csv')