import os
import sys
from fpdf import FPDF

folders=["ignore\\resultadosTDV\\batch\\GraficaMes\\2019"]

#define a function to scan the directory and return a list with the subdirectories
def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

for folder in folders:
    #scan the directory and return a list with the subdirectories
    subfolders = fast_scandir(folder)

    #create a general pdf in the main directory
    gen_pdf = FPDF()
    #for the pdf name, get the name of the main directory
    mainfolder = folder.split("\\")[-1]

    #for each subdirectory, locates the png files and adds them to a pdf per subdirectory and to the general pdf
    for subfolder in subfolders:
        #scan the directory and return a list with the png files
        files = [f for f in os.listdir(subfolder) if f.endswith('.png')]
        #order the list of files by name
        files.sort()
        #if there are png files, create a pdf
        if len(files)>0:
            #create a pdf with the name of the subdirectory
            pdf = FPDF()
            #for the pdf name, get the name of each subdirectory
            subfoldersplit=subfolder.split("\\")
            #for each png file, add a page to the local pdf and to the general pdf
            for file in files:
                pdf.add_page("L")
                pdf.image(subfolder+"\\"+file,0,0,297,210)
                gen_pdf.add_page("L")
                gen_pdf.image(subfolder+"\\"+file,0,0,297,210)
            
            #save the pdf in the subdirectory
            pdf.output(subfolder+"\\"+subfoldersplit[-2]+"-"+subfoldersplit[-1]+".pdf","F")

            print("Created PDF in "+subfolder+".pdf")

    #save the general pdf in the main directory
    gen_pdf.output(folder+"\\"+mainfolder+".pdf","F")

