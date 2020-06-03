from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QSize
from PyQt5 import QtGui


# Subclass QMainWindow to customise your application's main window
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("My Awesome App")
        self.setGeometry(0, 0, self.width, self.height)

        label = QtWidgets.QLabel("This is a PyQt5 window!")

        # The `Qt` namespace has a lot of attributes to customise
        # widgets. See: http://doc.qt.io/qt-5/qt.html
        label.setAlignment(Qt.AlignCenter)

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(label)

        toolbar = QtWidgets.QToolBar("My main toolbar")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        button_action = QtWidgets.QAction(QtGui.QIcon("./assets/bug.png"), "Your button", self)
        button_action.setStatusTip("This is your button")
        button_action.triggered.connect(self.onMyToolBarButtonClick)
        button_action.setCheckable(True)
        toolbar.addAction(button_action)

        toolbar.addSeparator()

        button_action2 = QtWidgets.QAction(QtGui.QIcon("./assets/bug.png"), "Your button2", self)
        button_action2.setStatusTip("This is your button2")
        button_action2.triggered.connect(self.onMyToolBarButtonClick)
        button_action2.setCheckable(True)
        toolbar.addAction(button_action2)

        toolbar.addWidget(QtWidgets.QLabel("Hello"))
        toolbar.addWidget(QtWidgets.QCheckBox())

        self.setStatusBar(QtWidgets.QStatusBar(self))

    def onMyToolBarButtonClick(self, s):
        print("click", s)


# Subclass QMainWindow to customise your application's main window
class MainWindow2(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow2, self).__init__(*args, **kwargs)

        self.setWindowTitle("My Awesome App")

        combo_box = QtWidgets.QComboBox()
        combo_box.addItems(["One", "Two", "Three"])

        label = QtWidgets.QLabel("Hello")
        font = label.font()
        font.setPointSize(30)
        label.setFont(font)
        label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        checkbox = QtWidgets.QCheckBox()
        checkbox.setCheckState(Qt.Checked)

        checkbox.setCheckState(Qt.PartiallyChecked)
        # Or: widget.setTriState(True)
        checkbox.stateChanged.connect(self.show_state)

        list_widget = QtWidgets.QListWidget()
        list_widget.addItems(["One", "Two", "Three"])

        line_edit = QtWidgets.QLineEdit()
        line_edit.setMaxLength(10)
        line_edit.setPlaceholderText("Enter your text")

        # widget.setReadOnly(True) # uncomment this to make readonly

        line_edit.returnPressed.connect(self.return_pressed)

        layout = QtWidgets.QVBoxLayout()
        widgets = [checkbox,
                   combo_box,
                   list_widget,
                   line_edit,
                   QtWidgets.QDateEdit(),
                   QtWidgets.QDateTimeEdit(),
                   QtWidgets.QDial(),
                   QtWidgets.QDoubleSpinBox(),
                   QtWidgets.QFontComboBox(),
                   QtWidgets.QLCDNumber(),
                   label,
                   QtWidgets.QLineEdit(),
                   QtWidgets.QProgressBar(),
                   QtWidgets.QPushButton(),
                   QtWidgets.QRadioButton(),
                   QtWidgets.QSlider(),
                   QtWidgets.QSpinBox(),
                   QtWidgets.QTimeEdit()
                   ]

        for w in widgets:
            layout.addWidget(w)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(widget)

    def show_state(self, s):
        print(s == Qt.Checked, s)

    def return_pressed(self):
        print("Return pressed!")
        for child in self.centralWidget().children():
            if type(child) == QtWidgets.QLineEdit:
                child.setText("BOOM!")


class Color(QtWidgets.QWidget):

    def __init__(self, color, *args, **kwargs):
        super(Color, self).__init__(*args, **kwargs)
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(color))
        self.setPalette(palette)


class MainWindow3(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow3, self).__init__(*args, **kwargs)

        self.setWindowTitle("My Awesome App")

        layout1 = QtWidgets.QHBoxLayout()
        layout2 = QtWidgets.QVBoxLayout()
        layout3 = QtWidgets.QVBoxLayout()

        layout1.setContentsMargins(0, 0, 0, 0)
        layout1.setSpacing(20)

        layout2.addWidget(Color('red'))
        layout2.addWidget(Color('yellow'))
        layout2.addWidget(Color('purple'))

        layout1.addLayout(layout2)

        layout1.addWidget(Color('green'))

        layout3.addWidget(Color('red'))
        layout3.addWidget(Color('purple'))

        layout1.addLayout(layout3)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout1)

        self.setCentralWidget(widget)


class CustomDialog(QtWidgets.QDialog):

    def __init__(self, *args, **kwargs):
        super(CustomDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("HELLO!")

        QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


class MainWindow4(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow4, self).__init__(*args, **kwargs)

        toolbar = QtWidgets.QToolBar("My main toolbar")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        button_action = QtWidgets.QAction(QtGui.QIcon("./assets/bug.png"), "Your button", self)
        button_action.setStatusTip("This is your button")
        button_action.triggered.connect(self.onMyToolBarButtonClick)
        toolbar.addAction(button_action)

    def onMyToolBarButtonClick(self, s):
        print("click", s)

        dlg = CustomDialog(self)
        if dlg.exec_():
            print("Success!")
        else:
            print("Cancel!")


class MainWindow5(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow5, self).__init__(*args, **kwargs)
        self.title = 'PyQt5 file dialogs - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.openFileNameDialog()
        self.openFileNamesDialog()
        self.saveFileDialog()

    def openFileNameDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)

    def openFileNamesDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "",
                                                "All Files (*);;Python Files (*.py)", options=options)
        if files:
            print(files)

    def saveFileDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            print(fileName)
