from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, 
    QHBoxLayout, QMessageBox, QProgressBar, QGroupBox, QDialog,
    QSpinBox, QDialogButtonBox, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
import pandas as pd
import os
from utils.utils_dataset import validate_and_merge_csv, process_all_images_in_folder, validate_and_merge_csv_parallel,organize_dataset_with_txt, apply_gamma_correction
from utils.utils_dataset import DatasetProcessor
import shutil
import cv2
from utils.utils_augmentation_data import flip_image, rotate_image, apply_gamma_correction

# Ajouter la classe de dialogue
class DatasetConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration du Dataset Final")
        self.setMinimumWidth(400)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Groupe pour les ratios d'images
        images_group = QGroupBox("Sélection des images")
        images_layout = QVBoxLayout()
        images_layout.setSpacing(10)

        # Pourcentage d'images annotées
        annotated_layout = QHBoxLayout()
        annotated_label = QLabel("Pourcentage d'images annotées à utiliser:")
        self.annotated_spin = QSpinBox()
        self.annotated_spin.setRange(0, 100)
        self.annotated_spin.setValue(80)
        self.annotated_spin.setSuffix("%")
        self.annotated_spin.setMinimumWidth(80)
        annotated_layout.addWidget(annotated_label)
        annotated_layout.addWidget(self.annotated_spin)
        images_layout.addLayout(annotated_layout)

        # Pourcentage d'images non annotées
        non_annotated_layout = QHBoxLayout()
        non_annotated_label = QLabel("Pourcentage d'images non annotées à utiliser:")
        self.non_annotated_spin = QSpinBox()
        self.non_annotated_spin.setRange(0, 100)
        self.non_annotated_spin.setValue(20)
        self.non_annotated_spin.setSuffix("%")
        self.non_annotated_spin.setMinimumWidth(80)
        non_annotated_layout.addWidget(non_annotated_label)
        non_annotated_layout.addWidget(self.non_annotated_spin)
        images_layout.addLayout(non_annotated_layout)

        # Note explicative
        note_label = QLabel(
            "Note: Les pourcentages sont appliqués indépendamment\n"
            "sur les images annotées et non annotées."
        )
        note_label.setStyleSheet("color: gray;")
        images_layout.addWidget(note_label)

        images_group.setLayout(images_layout)
        layout.addWidget(images_group)

        # Groupe pour les ratios train/val/test
        split_group = QGroupBox("Ratios de séparation du dataset")
        split_layout = QVBoxLayout()
        split_layout.setSpacing(10)

        # Train ratio
        train_layout = QHBoxLayout()
        train_label = QLabel("Train:")
        self.train_spin = QSpinBox()
        self.train_spin.setRange(0, 100)
        self.train_spin.setValue(80)
        self.train_spin.setSuffix("%")
        self.train_spin.setMinimumWidth(80)
        train_layout.addWidget(train_label)
        train_layout.addWidget(self.train_spin)
        split_layout.addLayout(train_layout)

        # Validation ratio
        val_layout = QHBoxLayout()
        val_label = QLabel("Validation:")
        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 100)
        self.val_spin.setValue(15)
        self.val_spin.setSuffix("%")
        self.val_spin.setMinimumWidth(80)
        val_layout.addWidget(val_label)
        val_layout.addWidget(self.val_spin)
        split_layout.addLayout(val_layout)

        # Test ratio (calculé automatiquement)
        test_layout = QHBoxLayout()
        test_label = QLabel("Test (automatique):")
        self.test_label = QLabel("5%")
        test_layout.addWidget(test_label)
        test_layout.addWidget(self.test_label)
        split_layout.addLayout(test_layout)

        split_group.setLayout(split_layout)
        layout.addWidget(split_group)

        # Connecter les signaux pour mettre à jour le ratio de test
        self.train_spin.valueChanged.connect(self.update_test_ratio)
        self.val_spin.valueChanged.connect(self.update_test_ratio)

        # Boutons OK/Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def update_test_ratio(self):
        test_ratio = 100 - self.train_spin.value() - self.val_spin.value()
        self.test_label.setText(f"{test_ratio}%")
        
    def validate_and_accept(self):
        # Vérifier que les ratios sont valides individuellement
        if self.annotated_spin.value() > 100:
            QMessageBox.warning(self, "Erreur", 
                "Le pourcentage d'images annotées ne peut pas dépasser 100%")
            return
            
        if self.non_annotated_spin.value() > 100:
            QMessageBox.warning(self, "Erreur", 
                "Le pourcentage d'images non annotées ne peut pas dépasser 100%")
            return

        # Vérifier les ratios train/val/test
        total_split = self.train_spin.value() + self.val_spin.value()
        if total_split >= 100:
            QMessageBox.warning(self, "Erreur", 
                "La somme des ratios Train et Validation ne doit pas dépasser 100%")
            return
            
        self.accept()

    def get_values(self):
        return {
            'annotated_ratio': self.annotated_spin.value() / 100,
            'non_annotated_ratio': self.non_annotated_spin.value() / 100,
            'train_ratio': self.train_spin.value() / 100,
            'val_ratio': self.val_spin.value() / 100
        }

class DatasetWorker(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, task, **kwargs):
        super().__init__()
        self.task = task
        self.kwargs = kwargs

    def run(self):
        try:
            if self.task == "validate":
                merged_df, message = validate_and_merge_csv_parallel(
                    self.kwargs['csv_files'],
                    progress_callback=self.progress_updated.emit
                )
                if merged_df is not None:
                    self.finished.emit({"df": merged_df, "message": "Success"})
                else:
                    self.error.emit(message)

            elif self.task == "process":
                processor = DatasetProcessor()
                processor.progress_updated.connect(self.progress_updated.emit)
                processor.status_updated.connect(self.status_updated.emit)
                
                processor.process_all_images_in_folder(
                    self.kwargs['folder_path'],
                    self.kwargs['df'],
                    self.kwargs['output_dir'],
                    self.kwargs['crop_size'],
                    self.kwargs['overlap']
                )

                # Apply gamma correction if augmentation is enabled
                if self.kwargs.get('augmentation', False):
                    self.apply_augmentation(self.kwargs['output_dir'])

                self.finished.emit({"message": "Success"})

            elif self.task == "organize":
                # Émettre un statut initial
                self.status_updated.emit("Organisation du dataset en cours...")
                self.progress_updated.emit(0)

                # Créer les répertoires
                output_dir = self.kwargs['output_dir']
                for split in ['train', 'val', 'test']:
                    os.makedirs(os.path.join(output_dir, f'images/{split}'), exist_ok=True)
                    os.makedirs(os.path.join(output_dir, f'labels/{split}'), exist_ok=True)
                self.progress_updated.emit(10)

                # Organiser le dataset
                stats = organize_dataset_with_txt(
                    self.kwargs['input_dir'],
                    self.kwargs['output_dir'],
                    self.kwargs['train_ratio'],
                    self.kwargs['val_ratio'],
                    self.kwargs['annotated_ratio'],
                    self.kwargs['non_annotated_ratio']
                )
                self.progress_updated.emit(80)

                # Créer le fichier dataset.yaml
                yaml_content = """train: images/train
val: images/val
test: images/test

nc: 1  # nombre de classes
names: ['pins']  # noms des classes"""

                with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
                    f.write(yaml_content)
                self.progress_updated.emit(100)

                self.finished.emit({"stats": stats})

        except Exception as e:
            self.error.emit(str(e))

    def apply_augmentation(self, output_dir):
        # Compter le nombre total de fichiers à traiter
        total_files = sum(1 for _, _, files in os.walk(output_dir) 
                         for file in files 
                         if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')))
        
        processed_files = 0
        self.progress_updated.emit(0)
        self.status_updated.emit("Début de l'augmentation des données...")

        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    image_path = os.path.join(root, file)
                    image = cv2.imread(image_path)
                    if image is not None:
                        base_name = os.path.splitext(file)[0]
                        # Définir annotation_path ici
                        annotation_path = os.path.join(root, f"{base_name}.txt")
                        
                        # Appliquer les augmentations sélectionnées
                        if self.kwargs['augmentation_options']['gamma']:
                            self.status_updated.emit(f"Application gamma correction sur {file}")
                            gamma_image = apply_gamma_correction(image)
                            gamma_path = os.path.join(root, f"gamma_{file}")
                            cv2.imwrite(gamma_path, gamma_image)
                            if os.path.exists(annotation_path):
                                shutil.copy(annotation_path, os.path.join(root, f"gamma_{base_name}.txt"))
                        
                        if self.kwargs['augmentation_options']['flip']:
                            self.status_updated.emit(f"Application flip horizontal sur {file}")
                            flip_image_h = flip_image(image, 1)
                            flip_path = os.path.join(root, f"flip_{file}")
                            cv2.imwrite(flip_path, flip_image_h)
                            if os.path.exists(annotation_path):
                                self.update_annotation_for_flip(annotation_path, root, f"flip_{base_name}.txt", image.shape[1])
                        
                        if self.kwargs['augmentation_options']['rotate']:
                            self.status_updated.emit(f"Application rotation sur {file}")
                            rotated_image = rotate_image(image, 90)
                            rotate_path = os.path.join(root, f"rotate_{file}")
                            cv2.imwrite(rotate_path, rotated_image)
                            if os.path.exists(annotation_path):
                                self.update_annotation_for_rotation(annotation_path, root, f"rotate_{base_name}.txt")

                    processed_files += 1
                    progress = int((processed_files / total_files) * 100)
                    self.progress_updated.emit(progress)

        self.status_updated.emit("Augmentation des données terminée")

    def update_annotation_for_flip(self, annotation_path, output_dir, new_annotation_name, image_width):
        with open(annotation_path, 'r') as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Assuming YOLO format: class x_center y_center width height
                class_id, x_center, y_center, width, height = map(float, parts)
                # Flip horizontally: x_center = 1 - x_center
                x_center = 1 - x_center
                new_lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

        new_annotation_path = os.path.join(output_dir, new_annotation_name)
        with open(new_annotation_path, 'w') as file:
            file.writelines(new_lines)

    def update_annotation_for_rotation(self, annotation_path, output_dir, new_annotation_name):
        """Met à jour les annotations pour une rotation de 90 degrés"""
        with open(annotation_path, 'r') as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                # Pour une rotation de 90°:
                # - Le nouveau x est l'ancien y
                # - Le nouveau y est 1 - l'ancien x
                # - Échanger width et height
                new_x = y_center
                new_y = 1 - x_center
                new_width = height
                new_height = width
                new_lines.append(f"{class_id} {new_x} {new_y} {new_width} {new_height}\n")

        new_annotation_path = os.path.join(output_dir, new_annotation_name)
        with open(new_annotation_path, 'w') as file:
            file.writelines(new_lines)

class DatasetUI(QWidget):
    def __init__(self):
        super().__init__()
        self.images = []
        self.csv_files = []  # Liste pour stocker plusieurs fichiers CSV
        self.merged_df = None  # Pour stocker le DataFrame fusionné
        self.dataset_processor = DatasetProcessor()
        self.dataset_processor.progress_updated.connect(self.update_progress)
        self.dataset_processor.status_updated.connect(self.update_status)
        self.worker = None
        self.data_augmentation_enabled = False  # New attribute to track augmentation state
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
        self.drop_label = QLabel("Glissez vos images (.jpg,.JPG, .png, .PNG, .bmp,.BMP,.tif,.tiff,.TIFF,)\net votre fichier CSV\nou cliquez pour sélectionner")
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

        # Ajouter un bouton pour le traitement des images
        self.process_button = QPushButton("Traiter les images")
        self.process_button.clicked.connect(self.process_images)
        self.process_button.setEnabled(False)  # Désactivé par défaut
        dataset_layout.addWidget(self.process_button)

        # Groupe pour les options d'augmentation
        augmentation_group = QGroupBox("Options d'augmentation")
        augmentation_layout = QVBoxLayout()

        self.augmentation_checkbox = QCheckBox("Activer l'augmentation de données")
        self.gamma_checkbox = QCheckBox("Correction gamma")
        self.flip_checkbox = QCheckBox("Flip horizontal")
        self.rotate_checkbox = QCheckBox("Rotation 90°")

        # Désactiver les options par défaut
        self.gamma_checkbox.setEnabled(False)
        self.flip_checkbox.setEnabled(False)
        self.rotate_checkbox.setEnabled(False)

        # Connecter les signaux
        self.augmentation_checkbox.stateChanged.connect(self.toggle_augmentation_options)

        augmentation_layout.addWidget(self.augmentation_checkbox)
        augmentation_layout.addWidget(self.gamma_checkbox)
        augmentation_layout.addWidget(self.flip_checkbox)
        augmentation_layout.addWidget(self.rotate_checkbox)
        augmentation_group.setLayout(augmentation_layout)

        dataset_layout.addWidget(augmentation_group)

        # Ajouter le groupe principal au layout principal
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
            if file_path.lower().endswith(('.jpg','.JPG', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                if file_path not in self.images:  # Éviter les doublons
                    self.images.append(file_path)
            elif file_path.lower().endswith('.csv'):
                if file_path not in self.csv_files:  # Éviter les doublons
                    self.csv_files.append(file_path)

        self.update_info_label()

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Sélectionner les fichiers du dataset",
            "",
            "Dataset Files (*.jpg *.jpeg *.png *.bmp *.JPG *.PNG *.BMP *.TIFF *.tif *.tiff *.csv);;All Files (*)"
        )
        
        for file_path in files:
            if file_path.lower().endswith(('.jpg','.JPG', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                if file_path not in self.images:
                    self.images.append(file_path)
            elif file_path.lower().endswith('.csv'):
                if file_path not in self.csv_files:
                    self.csv_files.append(file_path)

        self.update_info_label()

    def update_info_label(self):
        info_text = []
        if self.images:
            info_text.append(f"Images chargées: {len(self.images)}")
        if self.csv_files:
            info_text.append(f"CSV chargés: {len(self.csv_files)}")
        if self.merged_df is not None:
            info_text.append("CSV fusionnés avec succès")
        
        if info_text:
            self.info_label.setText("\n".join(info_text))
        else:
            self.info_label.setText("Aucun fichier chargé")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.info_label.setText(message)

    def validate_dataset(self):
        if not self.images or not self.csv_files:
            QMessageBox.warning(self, "Erreur", "Veuillez charger des images et au moins un fichier CSV")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.validate_button.setEnabled(False)

        self.worker = DatasetWorker(task="validate", csv_files=self.csv_files)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.finished.connect(self.on_validation_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_validation_finished(self, result):
        if "df" in result:
            self.merged_df = result["df"]
            self.info_label.setText("Dataset validé et CSV fusionnés avec succès")
            self.process_button.setEnabled(True)
            QMessageBox.information(self, "Succès", "Dataset validé avec succès!")
        
        self.validate_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def toggle_augmentation_options(self, state):
        """Active/désactive les options d'augmentation"""
        enabled = state == Qt.Checked
        self.gamma_checkbox.setEnabled(enabled)
        self.flip_checkbox.setEnabled(enabled)
        self.rotate_checkbox.setEnabled(enabled)
        self.data_augmentation_enabled = enabled

    def process_images(self):
        if self.merged_df is None or not self.images:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord valider le dataset")
            return

        try:
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            os.makedirs(temp_dir, exist_ok=True)

            # Désactiver les deux boutons pendant le traitement
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.process_button.setEnabled(False)
            self.validate_button.setEnabled(False)  # Désactiver le bouton de validation

            self.worker = DatasetWorker(
                task="process",
                folder_path=os.path.dirname(self.images[0]),
                df=self.merged_df,
                output_dir=temp_dir,
                crop_size=(640, 640),
                overlap=100,
                augmentation=self.data_augmentation_enabled,
                augmentation_options={
                    'gamma': self.gamma_checkbox.isChecked(),
                    'flip': self.flip_checkbox.isChecked(),
                    'rotate': self.rotate_checkbox.isChecked()
                }
            )
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.status_updated.connect(self.update_status)
            self.worker.finished.connect(lambda: self.show_config_dialog(temp_dir))
            self.worker.error.connect(self.on_error)
            self.worker.start()

        except Exception as e:
            self.cleanup_and_reset(temp_dir)
            QMessageBox.critical(self, "Erreur", f"Erreur lors du traitement: {str(e)}")

    def show_config_dialog(self, temp_dir):
        config_dialog = DatasetConfigDialog(self)
        if config_dialog.exec_():
            config = config_dialog.get_values()
            
            final_output_dir = QFileDialog.getExistingDirectory(
                self, "Sélectionner le dossier de sortie final", "",
                QFileDialog.ShowDirsOnly
            )

            if final_output_dir:
                # Afficher la barre de progression
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                
                self.worker = DatasetWorker(
                    task="organize",
                    input_dir=temp_dir,
                    output_dir=final_output_dir,
                    **config
                )
                self.worker.progress_updated.connect(self.update_progress)
                self.worker.status_updated.connect(self.update_status)
                self.worker.finished.connect(lambda result: self.on_organization_finished(result, temp_dir))
                self.worker.error.connect(self.on_error)
                self.worker.start()
            else:
                self.cleanup_and_reset(temp_dir)
        else:
            self.cleanup_and_reset(temp_dir)

    def on_organization_finished(self, result, temp_dir):
        if "stats" in result:
            stats = result["stats"]
            stats_message = (
                f"Dataset créé avec succès!\n\n"
                f"Images annotées: {stats['selected_annotated']}/{stats['total_annotated']}\n"
                f"Images non annotées: {stats['selected_non_annotated']}/{stats['total_non_annotated']}\n\n"
                f"Distribution:\n"
                f"- Train: {stats['train']} images\n"
                f"- Validation: {stats['val']} images\n"
                f"- Test: {stats['test']} images"
            )
            QMessageBox.information(self, "Succès", stats_message)
        
        self.cleanup_and_reset(temp_dir)

    def cleanup_and_reset(self, temp_dir):
        # Nettoyer le dossier temporaire
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Erreur lors du nettoyage du dossier temporaire: {str(e)}")

        # Réinitialiser l'interface
        self.progress_bar.setVisible(False)
        self.process_button.setEnabled(False)
        self.validate_button.setEnabled(True)  # Réactiver le bouton de validation
        self.images = []
        self.csv_files = []
        self.merged_df = None
        self.update_info_label()

    def on_error(self, error_message):
        QMessageBox.critical(self, "Erreur", error_message)
        self.progress_bar.setVisible(False)
        self.validate_button.setEnabled(True)  # Réactiver le bouton en cas d'erreur
        self.process_button.setEnabled(bool(self.merged_df)) 