
from PyQt5.QtWidgets import ( QMainWindow, QAction, QStackedWidget)

# Utiliser des imports relatifs avec le point
from .window_predict import PredictUI
from .window_cl import ContinuousLearningUI

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Application de Prédiction d'Images")
        self.setGeometry(100, 100, 800, 600)

        # Menu Bar
        menu_bar = self.menuBar()

        # Menu pour la prédiction
        predict_menu = menu_bar.addMenu("Prédiction")
        predict_action = QAction("Interface de Prédiction", self)
        predict_action.triggered.connect(self.show_predict_ui)
        predict_menu.addAction(predict_action)

        # Menu pour l'apprentissage continu
        learning_menu = menu_bar.addMenu("Apprentissage Continu")
        learning_action = QAction("Interface d'Apprentissage Continu", self)
        learning_action.triggered.connect(self.show_learning_ui)
        learning_menu.addAction(learning_action)

        # Widget central avec QStackedWidget pour basculer entre les vues
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Initialisation des interfaces
        self.predict_ui = PredictUI()
        self.learning_ui = ContinuousLearningUI()

        self.stacked_widget.addWidget(self.predict_ui)
        self.stacked_widget.addWidget(self.learning_ui)

    def show_predict_ui(self):
        self.stacked_widget.setCurrentWidget(self.predict_ui)

    def show_learning_ui(self):
        self.stacked_widget.setCurrentWidget(self.learning_ui)


