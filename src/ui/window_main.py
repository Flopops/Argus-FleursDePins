from PyQt5.QtWidgets import ( QMainWindow, QAction, QStackedWidget, QMessageBox)

# Utiliser des imports relatifs avec le point
from .window_predict import PredictUI
from .window_cl import ContinuousLearningUI
from .window_model import ModelSelectorUI
from .window_dataset import DatasetUI
from utils.utils_predict import update_model
import shutil
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

        # Menu pour le dataset
        dataset_menu = menu_bar.addMenu("Dataset")
        dataset_action = QAction("Gestion du Dataset", self)
        dataset_action.triggered.connect(self.show_dataset_ui)
        dataset_menu.addAction(dataset_action)

        # Menu pour l'apprentissage continu
        learning_menu = menu_bar.addMenu("Apprentissage Continu")
        learning_action = QAction("Interface d'Apprentissage Continu", self)
        learning_action.triggered.connect(self.show_learning_ui)
        learning_menu.addAction(learning_action)

        # Menu pour la sélection du modèle
        model_menu = menu_bar.addMenu("Modèle")
        model_action = QAction("Sélection du modèle", self)
        model_action.triggered.connect(self.show_model_ui)
        model_menu.addAction(model_action)

        # Widget central avec QStackedWidget pour basculer entre les vues
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Initialisation des interfaces
        self.predict_ui = PredictUI()
        self.learning_ui = ContinuousLearningUI()
        self.model_ui = ModelSelectorUI()
        self.dataset_ui = DatasetUI()

        # Connecter le signal de sélection du modèle
        self.model_ui.model_selected.connect(self.on_model_selected)

        self.stacked_widget.addWidget(self.predict_ui)
        self.stacked_widget.addWidget(self.learning_ui)
        self.stacked_widget.addWidget(self.model_ui)
        self.stacked_widget.addWidget(self.dataset_ui)

    def show_predict_ui(self):
        self.stacked_widget.setCurrentWidget(self.predict_ui)

    def show_learning_ui(self):
        self.stacked_widget.setCurrentWidget(self.learning_ui)

    def show_model_ui(self):
        self.stacked_widget.setCurrentWidget(self.model_ui)

    def show_dataset_ui(self):
        self.stacked_widget.setCurrentWidget(self.dataset_ui)

    def on_model_selected(self, model_path):
        """Gère la sélection d'un nouveau modèle"""
        if update_model(model_path):
            QMessageBox.information(self, "Succès", f"Modèle chargé avec succès: {model_path}")
        else:
            QMessageBox.critical(self, "Erreur", f"Erreur lors du chargement du modèle: {model_path}")


