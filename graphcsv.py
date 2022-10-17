import pandas as pd
import plotly.express as px
import PySimpleGUI as sg
import os.path
import plotly.graph_objects as go

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

            
            #fig drawing
            fig = px.line(title='Datos')
            for x in range(0,len(df.columns)//2):#len(df.columns)//2
                coli=x*2 #column index of time columns
                fig.add_trace(go.Scatter(x=df.iloc[:,coli], y=df.iloc[:,coli+1],
                    mode='lines',
                    name=df.columns[coli+1]))
            #df.dtypes
            fig.show()
            fig.write_html("output.html")
        except:
            pass

window.close()
