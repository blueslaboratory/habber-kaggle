import tkinter as tk
from tkinter import ttk
from tkinter.constants import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from titanic.graph import *


class KpisWindow:
    kpis = {
        "Sex": get_graph_sex,
        "Class": get_graph_pclass,
        "Relatives": get_graph_relatives,
        # "Parents and Childrens": get_graph_parch,
        "Age": get_graph_age
    }

    kpis_default_value = 'Sex'

    def __init__(self):
        root = tk.Toplevel()
        root.title("KPI´s Graphics")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        mainframe = ttk.Frame(root, padding="20 20 20 20")
        mainframe.pack(expand=TRUE, fill=BOTH)

        figure = KpisWindow.kpis[KpisWindow.kpis_default_value]()
        self.canvas = FigureCanvasTkAgg(figure=figure, master=mainframe)

        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(expand=TRUE, fill=BOTH)

        self.kpi_var = tk.StringVar(value=KpisWindow.kpis_default_value)
        kpi_select_box = ttk.Combobox(mainframe, textvariable=self.kpi_var)
        kpi_select_box.bind('<<ComboboxSelected>>', self.__change_graphic)
        kpi_select_box['values'] = list(KpisWindow.kpis.keys())
        kpi_select_box['state'] = 'readonly'

        kpi_select_box.pack()

    def __change_graphic(self, event):
        self.canvas.figure = KpisWindow.kpis[self.kpi_var.get()]()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas.draw()
