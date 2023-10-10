import tkinter as tk
from tkinter.constants import *
from tkinter import ttk
from titanic.kpiswindow import KpisWindow


class MainWindow:
    def __init__(self):
        root = tk.Tk()
        root.title("KPIs del Titanic")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        mainframe = ttk.Frame(root, width=250, height=250, padding="10 10 12 12")
        mainframe.pack(expand=TRUE, fill=BOTH)

        show_kpis_button = ttk.Button(mainframe, text="Show KPIs", command=self.show_kpis_window)
        show_kpis_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        root.mainloop()

    def show_kpis_window(self):
        KpisWindow()
