# data_viewer.py

import PySimpleGUI as sg
import os.path
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib as mpl
import matplotlib.pyplot as plt

n=1
fig, ax = plt.subplots()

mpl.use("TkAgg")

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

# First the window layout in 2 columns

file_list_column = [
    [
        sg.Text("Data Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

# ToDo: select data

data_list_column = [
    [
        sg.Text("Data Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose a file from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Graph((640, 480), (0, 0), (640, 480), key='Graph')],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(data_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window('Matplotlib', layout, finalize=True)

canvas = FigureCanvasTkAgg(fig, window['Graph'].Widget)
plot_widget = canvas.get_tk_widget()
plot_widget.grid(row=0, column=0)

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
            window["-TOUT-"].update(filename)

            #fig generation

            fig = mpl.figure.Figure(figsize=(5, 4), dpi=100)
            x = np.arange(0, 3, .01)
            y = n * np.sin(2 * np.pi * x)

            ax.cla()
            ax.set_title("Sensor Data")
            ax.set_xlabel("X axis")
            ax.set_ylabel("Y axis")
            ax.set_xscale('log')
            ax.grid()

            plt.plot(x, y)      # Plot new curve
            fig.canvas.draw()   # Draw curve really
            n=n+1
        

        except:
            pass

window.close()