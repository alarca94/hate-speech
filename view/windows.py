import os

import pandas as pd

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QSize
from PyQt5 import QtGui


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Annotator for linguists")

        gen_layout = QtWidgets.QVBoxLayout()
        gen_layout.setAlignment(Qt.AlignCenter)

        open_button = QtWidgets.QPushButton('Open .csv', self)
        open_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        open_button.clicked.connect(self.open_file)

        self.statusBar = QtWidgets.QStatusBar(self)
        self.default_styleSheet = self.statusBar.styleSheet()
        self.setStatusBar(self.statusBar)

        gen_layout.addWidget(open_button)

        widget = QtWidgets.QWidget()
        widget.setLayout(gen_layout)

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(widget)

    def setStyle(self, style):
        if style == 'error':
            self.statusBar.setStyleSheet(
                "QStatusBar{color:red;font-weight:bold;}")
        elif style == 'default':
            self.statusBar.setStyleSheet(self.default_styleSheet)

    def open_file(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                            "All Files (*);;Python Files (*.py)", options=options)
        if filename:
            if filename.endswith('xlsx'):
                data = pd.read_excel(filename)
            else:
                self.setStyle('error')
                self.statusBar.showMessage('Selected File does not have the correct format (.csv file)', 5000)

    def read_excel(self, filename):
        sheets = ['ECONOMÍA', 'INMIGRACIÓN', 'POLÍTICA', 'RELIGIÓN']

        useful_columns = ['ID', 'Column1', 'COM. CONSTRUCTIU', 'COM. TÒXIC', 'GRAU TOXICITAT', 'Sarcasme/Ironia',
                          'Burla/ridiculització', 'Insults', 'Argumentació/Diàleg', 'Llenguatge negatiu/tòxic']

        columns = ['TextData', 'Constructive', 'Toxic', 'ToxicityDegree', 'Sarcasm/Irony',
                   'Mockery/Ridicule', 'Insults', 'Argument/Discussion', 'NegativeToxicLanguage']

        data = pd.DataFrame()
        for sheet in sheets:
            # Read data
            new_data = pd.read_excel(os.path.join(filename), sheet_name=sheet, index_col='ID',
                                     usecols=useful_columns)
            # Rename the columns
            new_data.columns = columns

            data = pd.concat([data, new_data], ignore_index=True)


