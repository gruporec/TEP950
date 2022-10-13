import pandas as pd
import PySimpleGUI as sg
import os.path

file_list_column = [
    [
        sg.Text("Carpeta de datos"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(button_text='Abrir')
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
    [sg.Text("Seleccionar archivo",  key='-c-')]
]

layout = [
    [
        sg.Column(file_list_column)
    ]
]

window = sg.Window('Matplotlib', layout, finalize=True)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".csv"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            #fig generation

            df = pd.read_csv(filename,na_values='.')

            #date formatting
            maxIndex=len(df.columns)/2*len(df)
            for x in range(len(df.columns)//2):
                coli=x*2 #column index of time columns
                for rowi in range(len(df)): #row index
                    index=(coli*len(df)/2+rowi)*100/maxIndex
                    window['-c-'].update("Formateando celda "+str(rowi)+","+str(coli)+". Completado: "+str(index)+"%.")
                    hashour=False
                    if " " in str(df.iloc[rowi,coli]):
                        splited=df.iloc[rowi,coli].split(" ")
                        date=splited[0]
                        hour=splited[1]
                        hashour=True
                    else:
                        date=df.iloc[rowi,coli]
                        hour=""
                    if "/" in str(date):
                        splited=date.split("/")
                        day=splited[0]
                        mon=splited[1]
                        yea=splited[2]
                        if hashour:
                            df.iloc[rowi,coli]=yea+"-"+mon+"-"+day+" "+hour
                        else:
                            df.iloc[rowi,coli]=yea+"-"+mon+"-"+day
                #print("Formateada celda "+str(rowi)+","+str(coli)+". Completado: "+str(index)+"%.")

            #save raw
            df.to_csv("rawOutput.csv", index=False)

        except:
            pass

window.close()
