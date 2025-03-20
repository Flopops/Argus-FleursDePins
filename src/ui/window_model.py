from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QPushButton, QFileDialog, QLabel
)
from PyQt5.QtCore import pyqtSignal
import os
import json
import shutil
class ModelSelectorUI(QWidget):
    model_selected = pyqtSignal(str)  # Signal émis quand un nouveau modèle est sélectionné
    CONFIG_FILE = "config/model_config.json"

    def __init__(self):
        super().__init__()
        self.models_dir = None
        self.init_ui()
        self.load_config()
        self.clear_subdirectories()  # Supprime les sous-dossiers au démarrage

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Label d'information
        self.info_label = QLabel("Sélectionnez un dossier contenant les modèles", self)
        layout.addWidget(self.info_label)

        # Bouton pour sélectionner le dossier des modèles
        self.select_dir_button = QPushButton("Sélectionner le dossier des modèles", self)
        self.select_dir_button.clicked.connect(self.select_models_directory)
        layout.addWidget(self.select_dir_button)

        # ComboBox pour la sélection du modèle
        self.model_combo = QComboBox(self)
        self.model_combo.currentTextChanged.connect(self.on_model_selected)
        layout.addWidget(self.model_combo)

        # Label pour afficher le modèle actuellement sélectionné
        self.current_model_label = QLabel("Aucun modèle sélectionné", self)
        layout.addWidget(self.current_model_label)

        layout.addStretch()

    def load_config(self):
        """Charge la configuration depuis le fichier JSON"""
        try:
            # Créer le dossier config s'il n'existe pas
            os.makedirs(os.path.dirname(self.CONFIG_FILE), exist_ok=True)
            
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    saved_dir = config.get('models_directory')
                    selected_model = config.get('selected_model')
                    
                    if saved_dir and os.path.exists(saved_dir):
                        self.models_dir = saved_dir
                        self.refresh_models_list()
                        self.info_label.setText(f"Dossier chargé: {saved_dir}")
                        
                        # Sélectionner le modèle sauvegardé dans le ComboBox
                        if selected_model:
                            index = self.model_combo.findText(selected_model)
                            if index >= 0:
                                self.model_combo.setCurrentIndex(index)
                                self.current_model_label.setText(f"Modèle sélectionné: {selected_model}")
                            
        except Exception as e:
            print(f"Erreur lors du chargement de la configuration: {e}")

    def save_config(self):
        """Sauvegarde la configuration dans le fichier JSON"""
        try:
            # Créer le dossier config s'il n'existe pas
            os.makedirs(os.path.dirname(self.CONFIG_FILE), exist_ok=True)
            
            config = {'models_directory': self.models_dir}
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de la configuration: {e}")

    def select_models_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Sélectionner le dossier contenant les modèles",
            self.models_dir or "",  # Utilise le dossier actuel comme point de départ
            QFileDialog.ShowDirsOnly
        )
        if directory:
            self.models_dir = directory
            self.refresh_models_list()
            self.save_config()  # Sauvegarde la nouvelle configuration

    def refresh_models_list(self):
        """Met à jour la liste des modèles disponibles dans le ComboBox"""
        if not self.models_dir:
            return

        selected_model = self.model_combo.currentText()  # Sauvegarde du modèle sélectionné

        self.model_combo.clear()
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pt')]

        if model_files:
            self.model_combo.addItems(model_files)
            self.info_label.setText(f"Modèles trouvés: {len(model_files)}")

            # Rétablir la sélection précédente si elle existe encore
            if selected_model in model_files:
                index = self.model_combo.findText(selected_model)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)
                    self.current_model_label.setText(f"Modèle sélectionné: {selected_model}")
        else:
            self.info_label.setText("Aucun modèle (.pt) trouvé dans ce dossier")


    def on_model_selected(self, model_name):
        if model_name:
            config = {
                'models_directory': self.models_dir,
                'selected_model': model_name
            }
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            self.current_model_label.setText(f"Modèle sélectionné: {model_name}")

    def get_selected_model_path(self):
        if self.model_combo.currentText() and self.models_dir:
            return os.path.join(self.models_dir, self.model_combo.currentText())
        return None

    def showEvent(self, event):
        """Rafraîchit la liste des modèles à chaque affichage de la fenêtre"""
        super().showEvent(event)
        self.refresh_models_list()

    def clear_subdirectories(self):
        """Supprime les sous-dossiers dans le répertoire des modèles sauf 'results'"""
        if self.models_dir and os.path.exists(self.models_dir):
            for item in os.listdir(self.models_dir):
                item_path = os.path.join(self.models_dir, item)
                if os.path.isdir(item_path) and item != "results":
                    try:
                        shutil.rmtree(item_path)  # Supprime le dossier et son contenu
                    except OSError:
                        print(f"Impossible de supprimer le dossier: {item_path}")
