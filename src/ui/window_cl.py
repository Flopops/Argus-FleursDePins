from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QFormLayout, QSpinBox, QLineEdit, QHBoxLayout, QMessageBox, QProgressBar, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from utils.utils_continous_learning import continual_learning_yolo
import json
import os
from datetime import datetime
from ultralytics import YOLO

class TrainingThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    CONFIG_FILE = "config/model_config.json"

    def __init__(self, model_path, yaml_path, epochs, img_size, batch_size,directory):
        super().__init__()
        self.model_path = model_path
        self.yaml_path = yaml_path
        self.epochs = epochs
        self.img_size = img_size
        self.batch_size = batch_size
        self.directory = directory
    def run(self):
        try:
            self.progress_signal.emit("Démarrage de l'apprentissage...")

            # Charger le modèle depuis la configuration
            with open(self.CONFIG_FILE, 'r') as f:
                config = json.load(f)
                models_directory = config.get('models_directory', '')

            # Générer le chemin de sauvegarde avec la date et l'heure
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(models_directory, f"{model_name}_{timestamp}.pt")

            continual_learning_yolo(self.model_path, self.yaml_path, self.epochs, self.img_size, self.batch_size, save_path,self.directory)

            self.finished_signal.emit(True, f"Apprentissage terminé avec succès! Modèle sauvegardé à {save_path}")
        except Exception as e:
            self.finished_signal.emit(False, f"Erreur: {str(e)}")

class ContinuousLearningUI(QWidget):
    def __init__(self):
        super().__init__()
        self.yaml_path = None
        self.training_thread = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Zone de drop pour le YAML
        self.drop_zone = QWidget()
        self.drop_zone.setAcceptDrops(True)
        self.drop_zone.setMinimumHeight(100)
        self.drop_zone.setStyleSheet("""
            QWidget {
                border: 2px dashed #aaa;
                border-radius: 5px;
                background-color: #f8f8f8;
            }
            QWidget:hover {
                border-color: #666;
            }
        """)

        drop_layout = QVBoxLayout(self.drop_zone)
        self.drop_label = QLabel("Glissez votre fichier YAML ici\nou cliquez pour sélectionner")
        self.drop_label.setAlignment(Qt.AlignCenter)
        drop_layout.addWidget(self.drop_label)

        main_layout.addWidget(self.drop_zone)
        self.drop_zone.installEventFilter(self)
        self.drop_zone.mousePressEvent = self.select_yaml

        # Formulaire des paramètres
        params_group = QGroupBox("Paramètres d'apprentissage")
        form_layout = QFormLayout()

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(100)
        form_layout.addRow("Nombre d'époques:", self.epochs_input)

        self.img_size_input = QSpinBox()
        self.img_size_input.setRange(32, 1280)
        self.img_size_input.setValue(640)
        self.img_size_input.setSingleStep(32)
        form_layout.addRow("Taille des images:", self.img_size_input)

        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 128)
        self.batch_size_input.setValue(16)
        form_layout.addRow("Batch size:", self.batch_size_input)

        params_group.setLayout(form_layout)
        main_layout.addWidget(params_group)

        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        main_layout.addWidget(self.progress_bar)

        # Barre de progression et information
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.info_label)

        # Bouton pour démarrer l'apprentissage
        self.start_button = QPushButton("Démarrer l'apprentissage")
        self.start_button.clicked.connect(self.start_learning)
        main_layout.addWidget(self.start_button)

    def eventFilter(self, obj, event):
        if obj == self.drop_zone:
            if event.type() == event.DragEnter:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    return True
            elif event.type() == event.Drop:
                url = event.mimeData().urls()[0]
                path = url.toLocalFile()
                if path.lower().endswith(('.yaml', '.yml')):
                    self.yaml_path = path
                    self.drop_label.setText(f"Fichier chargé: {os.path.basename(path)}")
                return True
        return super().eventFilter(obj, event)

    def select_yaml(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner un fichier YAML",
            "",
            "YAML files (*.yaml *.yml)"
        )
        if file_path:
            self.yaml_path = file_path
            self.drop_label.setText(f"Fichier chargé: {os.path.basename(file_path)}")

    def start_learning(self):
        if not self.yaml_path:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord charger un fichier YAML")
            return

        try:
            # Charger le modèle depuis la configuration
            with open("config/model_config.json", 'r') as f:
                config = json.load(f)
                model_path = os.path.join(config.get('models_directory', ''), config.get('selected_model', ''))
                directory = os.path.join(config.get('models_directory', ''))
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Erreur", "Aucun modèle valide n'a été configuré")
                return

            # Désactiver les contrôles pendant l'apprentissage
            self.start_button.setEnabled(False)
            self.drop_zone.setEnabled(False)
            self.epochs_input.setEnabled(False)
            self.img_size_input.setEnabled(False)
            self.batch_size_input.setEnabled(False)

            # Lancer l'apprentissage dans un thread séparé
            self.training_thread = TrainingThread(
                model_path=model_path,
                yaml_path=self.yaml_path,
                epochs=self.epochs_input.value(),
                img_size=self.img_size_input.value(),
                batch_size=self.batch_size_input.value(),
                directory=directory
            )
            self.training_thread.progress_signal.connect(self.update_progress)
            self.training_thread.finished_signal.connect(self.training_finished)
            self.training_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors du démarrage: {str(e)}")
            self.enable_controls()

    def update_progress(self, message):
        self.info_label.setText(message)
        # Update progress bar here if you have one

    def training_finished(self, success, message):
        self.enable_controls()
        if success:
            QMessageBox.information(self, "Succès", message)
        else:
            QMessageBox.critical(self, "Erreur", message)

    def enable_controls(self):
        self.start_button.setEnabled(True)
        self.drop_zone.setEnabled(True)
        self.epochs_input.setEnabled(True)
        self.img_size_input.setEnabled(True)
        self.batch_size_input.setEnabled(True)
