import sys
import pandas as pd
import os
import numpy as np
from datetime import time

#add the path to the lib folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
# import the isadoralib library
import isadoralib as isl

years=["2014","2015","2016","2019"]
stress_mismatch=[-1,0,0,0]

usemeteo=True

# Create an empty dataframe to save the processed data
savedf=pd.DataFrame()

# For each year in the list
for i in range(0,len(years),1):
    # Get the year
    year=years[i]
    # Get the stress mismatch
    mismatch=stress_mismatch[i]

    # Load the data
    tdv,ltp,meteo,stress=isl.cargaDatosTDV(year,"")

    # Remove the nan values
    tdv=tdv.dropna()
    ltp=ltp.dropna()
    meteo=meteo.dropna()
    stress=stress.dropna()

    # Get the values of tdv between 5 and 8 of each day
    tdv_5_8 = tdv.between_time(time(5,0),time(8,0))
    # Get the maximum of tdv between 5 and 8 of each day
    tdv_5_8_max = tdv_5_8.groupby(tdv_5_8.index.date).max()
    # Get the difference between the maximum of tdv between 5 and 8 of each day
    tdv_5_8_max_diff = tdv_5_8_max.diff(periods=1).dropna()
    # Get the sign of the difference between the maximum of tdv between 5 and 8 of each day
    tdv_5_8_max_diff_sign = tdv_5_8_max_diff.apply(np.sign)
    # Change the negative values to 0
    tdv_5_8_max_diff_sign[tdv_5_8_max_diff_sign<0]=0
    # Create a dataframe that is 1 when tdv_5_8_max_diff_sign is 0 and 0 when it is 1
    tdv_5_8_max_diff_sign_inv=tdv_5_8_max_diff_sign.apply(lambda x: 1-x)

    # Create two dataframes with the size of tdv_5_8_max_diff_sign and values 0
    pk0=pd.DataFrame(np.zeros(tdv_5_8_max_diff_sign.shape),index=tdv_5_8_max_diff_sign.index,columns=tdv_5_8_max_diff_sign.columns)
    pk1=pd.DataFrame(np.zeros(tdv_5_8_max_diff_sign.shape),index=tdv_5_8_max_diff_sign.index,columns=tdv_5_8_max_diff_sign.columns)
    # For each day in tdv_5_8_max_diff_sign
    for i in tdv_5_8_max_diff_sign.index:
        # If it is the first row
        if i==tdv_5_8_max_diff_sign.index[0]:
            # Add to pk0 the value of tdv_5_8_max_diff_sign_inv
            pk0.loc[i]=tdv_5_8_max_diff_sign_inv.loc[i]
            # Add to pk1 the value of tdv_5_8_max_diff_sign
            pk1.loc[i]=tdv_5_8_max_diff_sign.loc[i]
        # If it is not the first row
        else:
            # Get the previous index by subtracting one day
            i_ant=i-pd.Timedelta(days=1)
            # Add to pk0 the value of the previous row of pk0 plus the value of the row of tdv_5_8_max_diff_sign_inv, multiplied by the value of the row of tdv_5_8_max_diff_sign_inv
            pk0.loc[i]=(pk0.loc[i_ant]+tdv_5_8_max_diff_sign_inv.loc[i])*tdv_5_8_max_diff_sign_inv.loc[i]
            # Add to pk1 the value of the previous row of pk1 plus the value of the row of tdv_5_8_max_diff_sign, multiplied by the value of the row of tdv_5_8_max_diff_sign
            pk1.loc[i]=(pk1.loc[i_ant]+tdv_5_8_max_diff_sign.loc[i])*tdv_5_8_max_diff_sign.loc[i]
    # Substract pk0 from pk1 to get the number of days since the last change of trend, with the sign of the current trend
    pk=pk1-pk0

    # add 1 to the value of the first row of tdv_5_8_max_diff_sign. This way, bk will be calculated as if all the past days were of the same trend
    tdv_5_8_max_diff_sign.iloc[0]=tdv_5_8_max_diff_sign.iloc[0]+1
    # create a new dataframe fk with the exponentially weighted moving average of tdv_5_8_max_diff_sign
    bk=tdv_5_8_max_diff_sign.ewm(alpha=0.5).sum()
    # substract 1 to the value of the first row of tdv_5_8_max_diff_sign to restore its original value
    tdv_5_8_max_diff_sign.iloc[0]=tdv_5_8_max_diff_sign.iloc[0]-1
    
    # Get the current trend into a new dataframe
    trend=tdv_5_8_max_diff_sign.copy()
    
    # Remove the nan values
    bk=bk.dropna()
    trend=trend.dropna()

    # Create a dataframe with diff tdv_5_8_max_diff_sign that represents the changes of trend
    ctend=pd.DataFrame(tdv_5_8_max_diff_sign.diff(periods=1).dropna())

    # Equal to 1 the non-null values
    ctend[ctend!=0]=1

    # Get the values of the maximum in the time slot when there is a change of trend
    max_ctend=tdv_5_8_max[ctend!=0]
    # Fill the null values with the previous value
    max_ctend=max_ctend.fillna(method='ffill')
    # When there is no previous value, fill with 0
    max_ctend=max_ctend.fillna(0)
    # Add one day to the date
    max_ctend.index = max_ctend.index + pd.Timedelta(days=1)
    # Get the difference between the maximum of tdv between the current maximum and the maximum in the last change of trend
    max_ctend_diff=tdv_5_8_max-max_ctend
    # Remove the nan values
    max_ctend_diff=max_ctend_diff.dropna()

    # Apply a mismatch offset to valdatapd
    stress.index = stress.index + pd.Timedelta(days=mismatch)

    # Convert the indices of tdv_5_8_max, pk, bk, max_ctend_diff and valdatapd to datetime
    tdv_5_8_max.index = pd.to_datetime(tdv_5_8_max.index)
    pk.index = pd.to_datetime(pk.index)
    bk.index = pd.to_datetime(bk.index)
    trend.index = pd.to_datetime(trend.index)
    max_ctend_diff.index = pd.to_datetime(max_ctend_diff.index)
    stress.index = pd.to_datetime(stress.index)

    # Crop the dataframes tdv_5_8_max, pk, bk, bk1, max_ctend_diff and valdatapd to have the same size and indices
    common_index = tdv_5_8_max.index.intersection(pk.index).intersection(bk.index).intersection(max_ctend_diff.index).intersection(stress.index).intersection(trend.index)
    tdv_5_8_max = tdv_5_8_max.loc[common_index]
    pk = pk.loc[common_index]
    bk = bk.loc[common_index]
    trend = trend.loc[common_index]
    max_ctend_diff = max_ctend_diff.loc[common_index]
    stress = stress.loc[common_index]

    # Stack the dataframes
    tdv_max_stack=tdv_5_8_max.stack()
    pk_stack=pk.stack()
    bk_stack=bk.stack()
    bk1_stack=trend.stack()
    ctend_stack=max_ctend_diff.stack()
    data_stack_val=stress.stack()

    # Create a dataframe with the values of tdv_max_stack, pk_stack, bk_stack and ctend_stack as columns
    #data_val=pd.DataFrame({'tdv_max':tdv_max_stack.copy(),'pk':pk_stack.copy(),'bk':bk_stack.copy(),'ctend':ctend_stack.copy()})
    data_val=pd.DataFrame({'pk':pk_stack.copy(),'bk':bk_stack.copy(),'ctend':ctend_stack.copy(),'bk1':bk1_stack.copy()})

    # Crop data_val to the indices of data_stack_val
    data_val=data_val.loc[data_stack_val.index]

    # Rename the first index of data_val to "date" and the second to "id"
    data_val.index.names=["date","id"]

    # If usemeteo is True
    if usemeteo:
        # Get the average daily values of meteo
        meteo_avg=meteo.groupby(meteo.index.date).mean()

        # Rename the index of meteo_avg to "date"
        meteo_avg.index.names=["date"]

        # Create a new dataframe
        meteo_avg_data=pd.DataFrame()

        # For every unique value in the second level of the index of data_val
        for i in data_val.index.get_level_values(1).unique():
            # create an auxiliary dataframe with the data of meteo_avg
            aux=pd.DataFrame(meteo_avg,index=meteo_avg.index,columns=meteo_avg.columns)
            # Add a column with the value of i
            aux["id"]=i
            # Add the new column to the index
            aux=aux.set_index("id",append=True)

            #add the values of aux to meteo_avg_data
            meteo_avg_data=pd.concat([meteo_avg_data,aux],axis=0)

        # Combine the data of meteo_avg_data and data_val
        data_val=pd.merge(data_val,meteo_avg_data,how="left",left_index=True,right_index=True)


    # Add a column Y with the values of data_stack_val, substrating 1 to the values so that the clases are 0, 1 and 2
    data_val["Y"]=data_stack_val-1

    # Add to savedf the values of data_val preserving the same columns
    savedf=pd.concat([savedf,data_val],axis=0)

    # Remove empty columns
    savedf=savedf.dropna(axis=1,how="all")

    # Remove the rows with nan values
    savedf=savedf.dropna()

# Swap the index levels
savedf=savedf.swaplevel()

# create a string with the last two digits of each year in years
year_datas_str = ''.join(year[-2:] for year in years)

# if meteo is True, add "Meteo" to the string
if usemeteo:
    year_datas_str=year_datas_str+"Meteo"

# store savedf in a csv with a name composed of 'TDVdb' followed by the last two digits of each year in year_datas
savedf.to_csv('db\TDVdb'+year_datas_str+'.csv')   