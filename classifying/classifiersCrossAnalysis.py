import pandas as pd
import os
import sys

# Report file in results/ZIM/ of root directory, the name of the file is the name of the classifier
reportFolder='results/ZIM/fullAnalysis/'
reportFile='accuracies.csv'

# Load the accuracies
accuracies = pd.read_csv(reportFolder+reportFile)

# Set the index to the first column
accuracies.set_index(accuracies.columns[0], inplace=True)

# Get the column names
columns = accuracies.columns

# remove the last 9 characters of the column names
columns = [column[:-9] for column in columns]

# get the unique values of the processed columns
columns = list(set(columns))

# create a new dataframe with the same index as accuracies
mean_accuracies = pd.DataFrame(index=accuracies.index)

# create another
diff_accuracies = pd.DataFrame(index=accuracies.index)

#drop BEST
accuracies.drop('BEST', axis=0, inplace=True)

# for each unique value in columns
for column in columns:
    # get the columns that contain the unique value
    cols = [col for col in accuracies.columns if column in col]

    # get the mean of the columns
    mean_accuracies[column] = accuracies[cols].mean(axis=1)
    # get the difference between the maximum and minimum of the columns
    diff_accuracies[column] = accuracies[cols].max(axis=1) - accuracies[cols].min(axis=1)
# get the best mean for each column and add it to a new row
mean_accuracies.loc['BEST MEAN'] = mean_accuracies.max(axis=0)

# get the best difference for each column and add it to a new row
diff_accuracies.loc['BEST DIFF'] = diff_accuracies.min(axis=0)

# save the mean accuracies in a csv file
mean_accuracies.to_csv(reportFolder+'mean_accuracies.csv')

# save the difference accuracies in a csv file
diff_accuracies.to_csv(reportFolder+'diff_accuracies.csv')