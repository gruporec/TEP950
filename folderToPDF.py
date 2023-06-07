import os
import sys
from fpdf import FPDF

folder="ignore\\resultadosTDV\\batch\\PCALDA6am\\IMG\\OriginalRiego\\2014"

#escanea el directorio y devuelve una lista con los subdirectorios
def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders
subfolders = fast_scandir(folder)

#crea un pdf general en el directorio principal
gen_pdf = FPDF()
#para el nombre del pdf, obtiene el nombre del directorio principal
mainfolder = folder.split("\\")[-1]

#por cada subdirectorio, localiza los archivos png y los añade a un pdf por subdirectorio y a un pdf general
for subfolder in subfolders:
    #escanea el directorio y devuelve una lista con los archivos png
    files = [f for f in os.listdir(subfolder) if f.endswith('.png')]
    #ordena la lista de archivos por nombre
    files.sort()
    #si hay archivos png, crea un pdf
    if len(files)>0:
        #crea un pdf con el nombre del subdirectorio
        pdf = FPDF()
        #para el nombre del pdf, obtiene el nombre de cada subdirectorio
        subfoldersplit=subfolder.split("\\")
        #por cada archivo png, añade una página al pdf local y al pdf general
        for file in files:
            pdf.add_page("L")
            pdf.image(subfolder+"\\"+file,0,0,297,210)
            gen_pdf.add_page("L")
            gen_pdf.image(subfolder+"\\"+file,0,0,297,210)
        
        #guarda el pdf en el directorio
        pdf.output(subfolder+"\\"+subfoldersplit[-3]+"-"+subfoldersplit[-2]+"-"+subfoldersplit[-1]+".pdf","F")

        print("PDF creado en "+subfolder+".pdf")

#guarda el pdf general en el directorio principal
gen_pdf.output(folder+"\\"+mainfolder+".pdf","F")

