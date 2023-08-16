import sys
from time import time
import traceback
from matplotlib.markers import MarkerStyle
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import time
import isadoralib as isl
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp
import multiprocessing as mp
import matplotlib.patches as mpatches

matplotlib.use('Agg')


#years_test = ["2014","2015","2016","2019"]
#years_test = ["2015"]
years_test = ["2014","2015","2016","2019"]
              
save_folder = '../ignore/resultadosTDV/batch/GraficaMes/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#create a folder to save the results if it doesn't exist for each test year
for year_test in years_test:
    if not os.path.exists(save_folder+year_test):
        os.makedirs(save_folder+year_test)



#create three empty lists for the data: tdv, data, and sun
tdv_tests=[]
data_tests=[]
sun_tests=[]

# for each test year
for year_test in years_test:

    #load the test data
    tdv_test,ltp_test,meteo_test,data_test=isl.cargaDatosTDV(year_test,"rht")

    #from meteo, only keep radiation info (R_Neta_Avg)
    meteo_test=meteo_test["R_Neta_Avg"]

    #create a rad_sign dataframe with value 1 where meteo is positive and -1 where meteo is negative
    rad_sign=meteo_test.copy()
    rad_sign[rad_sign>0]=1
    rad_sign[rad_sign<0]=-1
    #create a rad_sign_diff dataframe with the difference between consecutive values of rad_sign
    rad_sign_diff=rad_sign.diff()
    #remove zero values from rad_sign_diff
    rad_sign_diff=rad_sign_diff[rad_sign_diff!=0]
    #remove nan values from rad_sign_diff
    rad_sign_diff=rad_sign_diff.dropna()
    #separate the positive and negative values of rad_sign_diff
    rad_sign_diff_pos=rad_sign_diff[rad_sign_diff>0]
    rad_sign_diff_neg=rad_sign_diff[rad_sign_diff<0]
    # separate date and hour from the index of rad_sign_diff_pos into two columns in a new dataframe
    sun=pd.DataFrame()
    #get date
    sun["date"]=rad_sign_diff_pos.index.date
    #get hour format hh:mm:ss
    sun["sunrise"]=rad_sign_diff_pos.index.time
    # make date the index of sun
    sun=sun.set_index("date")
    # remove duplicate dates keeping the first one
    sun=sun[~sun.index.duplicated(keep='first')]
    # separate date and hour from the index of rad_sign_diff_neg into two columns in a new dataframe
    sunAux=pd.DataFrame()
    #get date
    sunAux["date"]=rad_sign_diff_neg.index.date
    #get hour format hh:mm:ss
    sunAux["sunset"]=rad_sign_diff_neg.index.time
    # make date the index of sunAux
    sunAux=sunAux.set_index("date")
    # remove duplicate dates keeping the last one
    sunAux=sunAux[~sunAux.index.duplicated(keep='last')]
    # merge sun and sunAux into a single dataframe
    sun=sun.merge(sunAux,how="outer",left_index=True,right_index=True)
    # add a column with next day's sunrise
    sun["sunriseNext"]=sun["sunrise"].shift(-1)

    # convert sun index to datetime
    sun.index=pd.to_datetime(sun.index)

    # remove any row with a nan value
    sun=sun.dropna()

    #add the test data to the lists
    tdv_tests.append(tdv_test.copy())
    data_tests.append(data_test.copy())
    sun_tests.append(sun.copy())


# for each element in tdv_tests
for i in range(len(tdv_tests)):
    year_test=years_test[i]
    # for every column in tdv_tests[i]
    for j in range(len(tdv_tests[i].columns)):
        # get the maximum value in column j of tdv_tests[i]
        max_an=tdv_tests[i].iloc[:,j].max()
        # get the minimum value in column j of tdv_tests[i]
        min_an=tdv_tests[i].iloc[:,j].min()

        #for every month in the index of tdv_tests[i]
        for month in tdv_tests[i].index.month.unique():
            # create a figure with the right size for a horizontal A4 
            plt.figure(figsize=(11.69,8.27))
            # get the dataframe containing the data of column j of month month as a copy
            tdv_test_month=tdv_tests[i].loc[tdv_tests[i].index.month==month].iloc[:,j].copy()
            # add the last day of the previous month to the dataframe
            #tdv_test_month=tdv_test_month.append(tdv_tests[i].loc[tdv_tests[i].index.month==month-1].iloc[:,j].copy().tail(1440))

            # order the data by date
            tdv_test_month=tdv_test_month.sort_index()

            # get the name of column j
            col=tdv_tests[i].columns[j]

            # get the days from tdv_tests[i] indices for column j of month month
            days=tdv_tests[i].loc[tdv_tests[i].index.month==month].index.day.unique()

            # get the dates of tdv_tests[i] indices for column j of month month adding the year and the month
            days_date=[str(year_test)+"-"+str(month)+"-"+str(day) for day in days]
            # for each day
            for iday in range(len(days)):
                day=days[iday]
                day_date=days_date[iday]
                # if it exists, get the value of column "real" corresponding to day day in the first index and column j in the second index
                if (str(day_date),col) in data_tests[i].stack().index:
                    real=str(int(data_tests[i].stack().loc[(str(day_date),col)]))
                else:
                    real=""

                # order the data of tdv_test_month of day day by hour
                tdv_test_month_day=tdv_test_month.loc[tdv_test_month.index.day==day].sort_index().dropna()

                # plot the data of tdv_tests[i] of column j of month month of day day with the color corresponding to real
                if real=="1":
                    plt.plot(tdv_test_month_day,label=tdv_tests[i].columns[j],color='blue')
                elif real=="2":
                    plt.plot(tdv_test_month_day,label=tdv_tests[i].columns[j],color='green')
                elif real=="3":
                    plt.plot(tdv_test_month_day,label=tdv_tests[i].columns[j],color='red')
                else:
                    plt.plot(tdv_test_month_day,label=tdv_tests[i].columns[j],color='black')
                #plt.plot(tdv_test_month,label=tdv_tests[i].columns[j])

            # limit the vertical axis to the annual maximum above and the annual minimum plus 5% below
            plt.ylim(min_an-(max_an-min_an)*0.05,max_an)
            bottom, top = plt.ylim()
            # expand the vertical axis downwards to fit text
            # bottom, top = plt.ylim()
            # plt.ylim(bottom-(top-bottom)*0.1,top)

            # get the indices of the test data of column j of month month
            days=data_tests[i].loc[(data_tests[i].index.month==month)].index

            # for each day
            for day in days:
                #add a green area between 5:00 and 8:00
                plt.axvspan(day+pd.Timedelta(hours=5),day+pd.Timedelta(hours=8),color='green',alpha=0.1)
            
            #get the indices of the sun data of month month
            days=sun_tests[i].loc[sun_tests[i].index.month==month].index

            #for each day
            for day in days:
                #get the sunrise and sunset times
                sunrise=sun_tests[i].loc[day,"sunrise"]
                sunset=sun_tests[i].loc[day,"sunset"]
                #add a gray area between 00:00 and sunrise
                plt.axvspan(day,day+pd.Timedelta(hours=sunrise.hour,minutes=sunrise.minute,seconds=sunrise.second),color='gray',alpha=0.1)
                #add a gray area between sunset and 00:00 of the next day
                plt.axvspan(day+pd.Timedelta(hours=sunset.hour,minutes=sunset.minute,seconds=sunset.second),day+pd.Timedelta(days=1),color='gray',alpha=0.1)

            # get the indices of data_tests
            days=data_tests[i].loc[data_tests[i].index.month==month].index
            days_date=days.date

            # for each day
            for iday in range(len(days)):
                day=days[iday]
                day_date=days_date[iday]

                # if it exists, get the value of column "real" corresponding to day day in the first index and column j in the second index
                if (str(day_date),col) in data_tests[i].stack().index:
                    real=str(int(data_tests[i].stack().loc[(str(day_date),col)]))
                else:
                    real=""
                
                xpos=pd.to_datetime(str(day_date)+" 12:00:00",format="%Y-%m-%d %H:%M:%S")
                # if there is data in tdv_test_month, assign the minimum to ypos
                if day in tdv_test_month.index:
                    ypos=min_an-(max_an-min_an)*0.03
                # otherwise, assign 0 to ypos
                else:
                    ypos=0


                day_text=real
                # write the value of real in the graph in red, green or blue depending on whether it is 3, 2 or 1
                if real=="1":
                    plt.text(xpos,ypos,day_text,horizontalalignment='center',verticalalignment='top',fontsize=8,color='blue')
                elif real=="2":
                    plt.text(xpos,ypos,day_text,horizontalalignment='center',verticalalignment='top',fontsize=8,color='green')
                elif real=="3":
                    plt.text(xpos,ypos,day_text,horizontalalignment='center',verticalalignment='top',fontsize=8,color='red')
                else:
                    plt.text(xpos,ypos,day_text,horizontalalignment='center',verticalalignment='top',fontsize=8,color='black')

            # add a title
            plt.title("Sensor: "+str(col)+" mes: "+str(month))

            # create a folder to save the plots if it doesn't exist
            if not os.path.isdir(save_folder+years_test[i]+"/"+str(tdv_tests[i].columns[j])):
                os.makedirs(save_folder+years_test[i]+"/"+str(tdv_tests[i].columns[j]))

            # save the plot
            plt.savefig(save_folder+years_test[i]+"/"+str(tdv_tests[i].columns[j])+"/"+f'{month:02}'+".png")

            # close the plot
            plt.close()