import PySimpleGUI as sg

layout = [  [sg.Text("Pulsa",  key='-c-')],
            [sg.Button('Pulsa para Iniciar', size=(60, 12), font='Arial 15', key='-b-')] ]
window = sg.Window('Test', layout, size=(550, 350))

clicks = 0
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    elif event == '-b-':
        text = " " if clicks < 1 else str(clicks)
        window['-b-'].Update(text)
        clicks += 1
        window['-c-'].update(str(clicks))

window.close()