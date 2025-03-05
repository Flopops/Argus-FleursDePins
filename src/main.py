from ui.window_main import MainWindow
import os
import sys
from PyQt5.QtWidgets import QApplication

# Ajouter ces lignes avant de créer l'application QT
os.environ["QT_QPA_PLATFORM"] = "xcb"
# Spécifier le chemin correct des plugins Qt
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/qt5/plugins"

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
