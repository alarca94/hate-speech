import sys

from PyQt5.QtWidgets import QApplication
# from view.windows_tests import MainWindow, MainWindow2, MainWindow3, MainWindow4, MainWindow5
from view.windows import MainWindow


def main():
    app = QApplication(sys.argv)
    screen_resolution = app.desktop().screenGeometry()
    width, height = screen_resolution.width(), screen_resolution.height()

    window = MainWindow()
    window.showMaximized()

    app.exec_()


if __name__ == '__main__':
    main()
