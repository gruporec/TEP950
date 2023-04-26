import os
import sys
from fpdf import FPDF

folder="ignore\\resultadosTDV\\comp"

#escanea el directorio y devuelve una lista con los subdirectorios
def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders
subfolders = fast_scandir(folder)

#por cada subdirectorio, localiza los archivos png y los añade a un pdf
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
        print(subfoldersplit[-2]+"-"+subfoldersplit[-3]+"d-"+subfoldersplit[-1])
        #por cada archivo png, añade una página al pdf
        for file in files:
            pdf.add_page()
            pdf.image(subfolder+"\\"+file,0,0,210,297)
        
        #guarda el pdf en el directorio
        pdf.output(subfolder+"\\"+subfoldersplit[-2]+"-"+subfoldersplit[-3]+"d-"+subfoldersplit[-1]+".pdf","F")

        print("PDF creado en "+subfolder+".pdf")
# #guarda el pdf en el directorio principal
# pdf.output(folder+"\\comparacion.pdf","F")

