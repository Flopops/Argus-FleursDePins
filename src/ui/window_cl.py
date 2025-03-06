from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, 
    QFormLayout, QSpinBox, QLineEdit, QHBoxLayout
)
from PyQt5.QtCore import Qt, pyqtSignal

class DropYamlWidget(QWidget):
    yaml_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)
        self.yaml_path = None
        
        # Style pour la zone de drop
        self.setStyleSheet("""
            DropYamlWidget {
                border: 2px dashed #aaa;
                border-radius: 5px;
                background-color: #f8f8f8;
            }
            DropYamlWidget:hover {
                border-color: #666;
            }
        """)
        
        # Layout pour le texte
        layout = QVBoxLayout(self)
        self.label = QLabel("Glissez votre fichier YAML ici\nou cliquez pour sélectionner", self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner un fichier YAML", "", "YAML files (*.yaml *.yml)"
        )
        if file_path:
            self.yaml_dropped.emit(file_path)
            self.yaml_path = file_path
            self.label.setText(f"Fichier chargé: {file_path.split('/')[-1]}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.yaml', '.yml')):
                self.yaml_dropped.emit(path)
                self.yaml_path = path
                self.label.setText(f"Fichier chargé: {path.split('/')[-1]}")
                break

class ContinuousLearningUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Layout principal
        main_layout = QVBoxLayout(self)

        # Zone de drop pour le YAML
        self.yaml_drop = DropYamlWidget()
        main_layout.addWidget(self.yaml_drop)

        # Formulaire pour les paramètres
        form_layout = QFormLayout()

        # Nombre d'époques
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(100)
        form_layout.addRow("Nombre d'époques:", self.epochs_input)

        # Taille des images
        self.img_size_input = QSpinBox()
        self.img_size_input.setRange(32, 1280)
        self.img_size_input.setValue(640)
        self.img_size_input.setSingleStep(32)
        form_layout.addRow("Taille des images:", self.img_size_input)

        # Batch size
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 128)
        self.batch_size_input.setValue(16)
        form_layout.addRow("Batch size:", self.batch_size_input)

        main_layout.addLayout(form_layout)# Label pour les informations
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.info_label)
        
        # Label pour les informations
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.info_label)

        # Bouton pour démarrer l'apprentissage
        self.start_button = QPushButton("Démarrer l'apprentissage")
        self.start_button.clicked.connect(self.start_learning)
        main_layout.addWidget(self.start_button)

        

    def start_learning(self):
        if not self.yaml_drop.yaml_path:
            self.info_label.setText("Erreur: Veuillez d'abord charger un fichier YAML")
            return

        config = {
            'yaml_path': self.yaml_drop.yaml_path,
            'epochs': self.epochs_input.value(),
            'img_size': self.img_size_input.value(),
            'batch_size': self.batch_size_input.value()
        }

        self.info_label.setText(
            f"Configuration:\n"
            f"YAML: {config['yaml_path'].split('/')[-1]}\n"
            f"Époques: {config['epochs']}\n"
            f"Taille images: {config['img_size']}\n"
            f"Batch size: {config['batch_size']}"
        )
        
        # Ici, vous pouvez ajouter la logique pour démarrer l'apprentissage
        # avec les paramètres configurés