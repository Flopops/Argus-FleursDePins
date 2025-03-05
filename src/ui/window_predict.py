import os
import csv
from PyQt5.QtWidgets import (
    QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QScrollArea,
    QHBoxLayout, QMessageBox, QProgressBar, QStackedWidget, QCheckBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import time
from utils.utils_predict import predict_image, model  # Assurez-vous que ce module est correctement importé

class PredictUI(QWidget):
    def __init__(self):
        super().__init__()

        # Layout principal
        layout = QVBoxLayout(self)

        # Zone pour afficher les images
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.images_widget = DropZoneWidget()  # Nouveau widget personnalisé
        self.images_layout = QHBoxLayout(self.images_widget)
        self.scroll_area.setWidget(self.images_widget)
        layout.addWidget(self.scroll_area)

        # Connecter le signal de drop du widget
        self.images_widget.files_dropped.connect(self.handle_dropped_files)

        # Bouton pour charger des images
        self.load_button = QPushButton("Charger des images", self)
        self.load_button.clicked.connect(self.load_images)
        layout.addWidget(self.load_button)

        # Barre de progression
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        # Label pour afficher les résultats
        self.result_label = QLabel("", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        # Thread pour le chargement des images
        self.loader_thread = None

        # Ajouter un bouton pour lancer la prédiction
        self.predict_button = QPushButton("Lancer la prédiction", self)
        self.predict_button.clicked.connect(self.process_predictions)
        layout.addWidget(self.predict_button)

        # Ajouter l'attribut pour le thread de prédiction
        self.prediction_thread = None

        # Stocker les chemins des images
        self.image_paths = []
        self.save_annotations_checkbox = QCheckBox("Sauvegarder les images avec annotations", self)
        layout.addWidget(self.save_annotations_checkbox)
        self.output_directory = None

    def load_images(self):
        options = QFileDialog.Options()
        file_names, _ = QFileDialog.getOpenFileNames(self, "Charger des images", "",
                                                     "Images (*.png *.xpm *.jpg *.jpeg *.tiff *.bmp);;All Files (*)", options=options)
        if file_names:
            # Ajouter les nouveaux chemins à la liste existante
            self.image_paths.extend(file_names)
            self.progress_bar.setMaximum(len(self.image_paths))
            self.progress_bar.setValue(0)
            self.loader_thread = ImageLoaderThread(file_names)
            self.loader_thread.progress_updated.connect(self.update_progress)
            self.loader_thread.image_loaded.connect(self.add_image)
            self.loader_thread.error_occurred.connect(self.show_error_message)
            self.loader_thread.finished.connect(self.reset_progress_bar)
            self.loader_thread.start()

    def process_predictions(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord charger des images.")
            return

        # Si la checkbox est cochée, demander le dossier de sauvegarde
        if self.save_annotations_checkbox.isChecked():
            self.output_directory = QFileDialog.getExistingDirectory(
                self,
                "Sélectionner le dossier de sauvegarde des images annotées",
                "",
                QFileDialog.ShowDirsOnly
            )
            if not self.output_directory:  # Si l'utilisateur annule
                return

        # Désactiver les boutons pendant le traitement
        self.predict_button.setEnabled(False)
        self.load_button.setEnabled(False)

        # Initialiser la barre de progression
        self.progress_bar.setMaximum(len(self.image_paths))
        self.progress_bar.setValue(0)

        # Créer et démarrer le thread de prédiction
        self.prediction_thread = PredictionThread(
            self.image_paths,
            model,
            self.save_annotations_checkbox.isChecked(),
            self.output_directory
        )
        self.prediction_thread.progress_updated.connect(self.update_progress)
        self.prediction_thread.prediction_complete.connect(self.save_predictions)
        self.prediction_thread.error_occurred.connect(self.handle_prediction_error)
        self.prediction_thread.finished.connect(self.prediction_finished)
        self.prediction_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        if value == 0:
            self.result_label.setText("Démarrage du traitement...")
        elif value < self.progress_bar.maximum():
            self.result_label.setText(f"Traitement en cours... ({value}/{self.progress_bar.maximum()})")
        else:
            self.result_label.setText("Traitement terminé. Enregistrement des résultats...")

    def save_predictions(self, results):
        try:
            # Demander à l'utilisateur où sauvegarder le CSV
            default_filename = "resultats_predictions.csv"
            csv_path, _ = QFileDialog.getSaveFileName(
                self,
                "Sauvegarder les résultats",
                default_filename,
                "CSV Files (*.csv);;All Files (*)"
            )

            if csv_path:
                with open(csv_path, 'w', newline='') as csvfile:
                    fieldnames = ['image', 'pine_flowers']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for result in results:
                        writer.writerow(result)

                self.result_label.setText(f"Prédictions terminées ! Résultats sauvegardés dans {csv_path}")

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de l'export CSV : {str(e)}")

    def handle_prediction_error(self, error_message):
        QMessageBox.critical(self, "Erreur", f"Erreur lors de la prédiction : {error_message}")

    def prediction_finished(self):
        # Réactiver les boutons
        self.predict_button.setEnabled(True)
        self.load_button.setEnabled(True)

        # Réinitialiser la barre de progression
        self.progress_bar.setValue(0)
        self.result_label.setText("Prêt pour un nouveau traitement.")

    def add_image(self, pixmap):
        image_label = QLabel()
        image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.images_layout.addWidget(image_label)

    def show_error_message(self, file_name):
        QMessageBox.warning(self, "Erreur de Chargement", f"Le fichier '{file_name}' n'est pas une image valide.")

    def reset_progress_bar(self):
        self.progress_bar.reset()
        self.result_label.setText("Images chargées. Prêt pour la prédiction.")

    def handle_dropped_files(self, file_paths):
        # Ajouter les nouveaux chemins à la liste existante
        self.image_paths.extend(file_paths)
        self.progress_bar.setMaximum(len(self.image_paths))
        self.progress_bar.setValue(0)
        self.loader_thread = ImageLoaderThread(file_paths)
        self.loader_thread.progress_updated.connect(self.update_progress)
        self.loader_thread.image_loaded.connect(self.add_image)
        self.loader_thread.error_occurred.connect(self.show_error_message)
        self.loader_thread.finished.connect(self.reset_progress_bar)
        self.loader_thread.start()

# Nouveau widget pour gérer le drag & drop
class DropZoneWidget(QWidget):
    files_dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        # Style pour montrer que c'est une zone de drop
        self.setStyleSheet("""
            DropZoneWidget {
                border: 2px dashed #aaa;
                border-radius: 5px;
                background-color: #f8f8f8;
            }
            DropZoneWidget:hover {
                border-color: #666;
            }
        """)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_paths = []
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.xpm')):
                    file_paths.append(file_path)
            if file_paths:
                self.files_dropped.emit(file_paths)
            event.acceptProposedAction()

class ImageLoaderThread(QThread):
    progress_updated = pyqtSignal(int)
    image_loaded = pyqtSignal(QPixmap)
    error_occurred = pyqtSignal(str)

    def __init__(self, file_names):
        super().__init__()
        self.file_names = file_names

    def run(self):
        for index, file_name in enumerate(self.file_names):
            if not QPixmap(file_name).isNull():
                pixmap = QPixmap(file_name)
                self.image_loaded.emit(pixmap)
            else:
                self.error_occurred.emit(file_name)
            self.progress_updated.emit(index + 1)
            time.sleep(0.02)  # Simule un délai de chargement

# Nouvelle classe pour le thread de prédiction
class PredictionThread(QThread):
    progress_updated = pyqtSignal(int)
    prediction_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_paths, model, save_annotations=False, output_directory=None):
        super().__init__()
        self.image_paths = image_paths
        self.model = model
        self.save_annotations = save_annotations
        self.output_directory = output_directory


    def run(self):
        try:
            results = []
            for i, image_path in enumerate(self.image_paths):
                # Faire la prédiction
                counts = predict_image(
                    image_path,
                    self.model,
                    save_annotations=self.save_annotations,
                    output_directory=self.output_directory
                )

                # Stocker les résultats
                results.append({
                    'image': os.path.basename(image_path),
                    'pine_flowers': counts['pine_flowers']
                })

                # Émettre le progrès
                self.progress_updated.emit(i + 1)

            # Émettre les résultats complets
            self.prediction_complete.emit(results)

        except Exception as e:
            self.error_occurred.emit(str(e))
