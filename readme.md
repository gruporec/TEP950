# File index and info
A comprehensive list of all the relevant and/or useful files and folders available in this github is available in the following lines.

## Tools (folder)
Contains tools that are useful to various task, not limited to a single part of the project but related to all, or at least several, of them

### folderToPDF.py
Given a list of directories to be inputed as folders=["dir1","dir2",etc], each inputed directory and each of its subdirectories will be scaned in search of png images. Every image found will be added to a PDF file per subfolder, as well as a general PDF file per main directory provided. The PDF files will be saved in each of the subdirectories and in the main directory respectively. The name of the PDF files will be the same as the name of the folder they are in.

### plotTDVMonthColor.py
Given a list of available years and a save directory path, plots TDV data and colors it according to manual clasification.

## LTP (folder)
Contains scripts related to LTP and ZIM probes.

### ZIMMulticlasifierAccPlot.py
Configurable script that tests accuracy of multiple configurable classifiers over a preprocessed ZIM probe database. Available clasifiers include linear and quadratic discriminant analysis from sklearn and krigging based classifiers from isadoralib, developped under this project. The script generates a plot with the measured accuracies versus the fraction of data that was used for training. It can also print relevant information such as the current configuration of the classifier, its confusion matrices and its balanced accuracy to console.

### ZIMsaveDataset.py
This script generates a preprocessed database to be used by **ZIMMulticlasifierAccPlot.py**. The applied preprocess is intended for ZIM probes. Some parameters can be adjusted via variables in the first lines of the script.

## TDV (folder)
Contains scripts related to TDV and dendrometers.

### plotClassifierTests.py
Plots some useful information about LTP characteristics to be used in the classifiers and saves the plots.

## db (folder)
Contains preprocesed data used by scripts.

### ZIMdb14151619.csv
Database containing proprocesed ZIM data from years 2014, 2015, 2016 and 2019. Each row contains data for one day and sensor. Contains sensor name and date (labeled "Fecha") to be used as indexes, a series of numbered columns containing relevant data for classification and an Y column containing manual clasification for that day and sensor. Relevant data includes 80 samples of ZIM sensor and 4 samples of each meteorological data, which includes net radiation, humidity and temperature, all measured during daytime. 