from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, 
    QHBoxLayout, QMessageBox, QProgressBar, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal
import pandas as pd
import os

class DatasetUI(QWidget):
    def __init__(self):
        super().__init__()
        self.images = []
        self.csv_file = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Création du groupe pour le dataset
        dataset_group = QGroupBox("Chargement du Dataset")
        dataset_layout = QVBoxLayout(dataset_group)

        # Zone de drop commune
        self.drop_zone = QWidget()
        self.drop_zone.setAcceptDrops(True)
        self.drop_zone.setMinimumHeight(200)
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
        
        # Label pour les instructions
        self.drop_label = QLabel("Glissez vos images (.jpg, .png, .bmp)\net votre fichier CSV\nou cliquez pour sélectionner")
        self.drop_label.setAlignment(Qt.AlignCenter)
        drop_layout.addWidget(self.drop_label)

        # Bouton pour sélectionner les fichiers
        select_button = QPushButton("Sélectionner les fichiers")
        select_button.clicked.connect(self.select_files)
        drop_layout.addWidget(select_button)

        dataset_layout.addWidget(self.drop_zone)

        # Label d'information
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        dataset_layout.addWidget(self.info_label)

        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        dataset_layout.addWidget(self.progress_bar)

        # Bouton de validation
        self.validate_button = QPushButton("Valider le dataset")
        self.validate_button.clicked.connect(self.validate_dataset)
        dataset_layout.addWidget(self.validate_button)

        main_layout.addWidget(dataset_group)

        # Installer l'event filter pour le drag & drop
        self.drop_zone.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self.drop_zone:
            if event.type() == event.DragEnter:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    return True
            elif event.type() == event.Drop:
                self.handle_drop(event.mimeData().urls())
                return True
        return super().eventFilter(obj, event)

    def handle_drop(self, urls):
        for url in urls:
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.images.append(file_path)
            elif file_path.lower().endswith('.csv'):
                self.csv_file = file_path
                self.validate_csv(file_path)

        self.update_info_label()

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Sélectionner les fichiers du dataset",
            "",
            "Dataset Files (*.jpg *.jpeg *.png *.bmp *.csv);;All Files (*)"
        )
        
        for file_path in files:
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.images.append(file_path)
            elif file_path.lower().endswith('.csv'):
                self.csv_file = file_path
                self.validate_csv(file_path)

        self.update_info_label()

    def update_info_label(self):
        info_text = []
        if self.images:
            info_text.append(f"Images chargées: {len(self.images)}")
        if self.csv_file:
            info_text.append(f"CSV chargé: {os.path.basename(self.csv_file)}")
        
        if info_text:
            self.info_label.setText("\n".join(info_text))
        else:
            self.info_label.setText("Aucun fichier chargé")

    def validate_csv(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['Label', 'X1', 'Y1', 'X2', 'Y2']
            
            if all(col in df.columns for col in required_columns):
                self.info_label.setText("Format CSV valide")
                return True
            
            if self.try_convert_csv(df, csv_path):
                self.info_label.setText("CSV converti avec succès")
                return True
            
            self.info_label.setText("Format CSV incompatible")
            return False
                
        except Exception as e:
            self.info_label.setText(f"Erreur lors de la lecture du CSV: {str(e)}")
            return False

    def try_convert_csv(self, df, original_path):
        """Tente de convertir différents formats de CSV connus"""
        try:
            if len(df.columns) == 5 and df.iloc[:, 1:].apply(lambda x: 0 <= x <= 1).all().all():
                converted_df = self.convert_yolo_format(df)
                new_path = original_path.replace('.csv', '_converted.csv')
                converted_df.to_csv(new_path, index=False)
                self.csv_file = new_path  # Mettre à jour le chemin du CSV
                return True
            return False
        except Exception as e:
            print(f"Erreur de conversion: {str(e)}")
            return False

    def convert_yolo_format(self, df):
        """Convertit du format YOLO vers notre format"""
        img_width = 1  # À ajuster selon vos besoins
        img_height = 1
        
        converted = pd.DataFrame(columns=['Label', 'X1', 'Y1', 'X2', 'Y2'])
        converted['Label'] = df.iloc[:, 0]
        
        x_center = df.iloc[:, 1]
        y_center = df.iloc[:, 2]
        width = df.iloc[:, 3]
        height = df.iloc[:, 4]
        
        converted['X1'] = (x_center - width/2) * img_width
        converted['Y1'] = (y_center - height/2) * img_height
        converted['X2'] = (x_center + width/2) * img_width
        converted['Y2'] = (y_center + height/2) * img_height
        
        return converted

    def validate_dataset(self):
        if not self.images or not self.csv_file:
            QMessageBox.warning(self, "Erreur", "Veuillez charger des images et un fichier CSV")
            return

        try:
            df = pd.read_csv(self.csv_file)
            image_files = set(os.path.splitext(os.path.basename(f))[0] 
                            for f in self.images)
            
            QMessageBox.information(self, "Succès", "Dataset validé avec succès!")
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la validation: {str(e)}") 