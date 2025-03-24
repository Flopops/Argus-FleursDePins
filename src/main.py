from ui.window_main import MainWindow
import os
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

# Ajouter ces lignes si Linux sinon supprimer
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/qt5/plugins"

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    iconpath = "icon.ico"
    window.setWindowIcon(QIcon(iconpath))
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
