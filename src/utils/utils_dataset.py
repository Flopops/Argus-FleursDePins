import pandas as pd
import os
import logging
from typing import List, Tuple, Optional
import cv2
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtCore import QObject, pyqtSignal
import shutil
from sklearn.model_selection import train_test_split

class DatasetProcessor(QObject):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.max_workers = max(1, os.cpu_count() - 1)  # Garder un cœur libre

    def process_image_parallel(self, args):
        image_path, df, output_dir, crop_size, overlap = args
        try:
            # Utiliser la version parallélisée du découpage d'image
            crops_count = crop_image_and_annotations_parallel(image_path, df, output_dir, crop_size, overlap)
            self.status_updated.emit(f"Image {os.path.basename(image_path)} traitée : {crops_count} découpes")
            return True
        except Exception as e:
            self.status_updated.emit(f"Erreur sur {image_path}: {str(e)}")
            return False

    def process_all_images_in_folder(self, folder_path, df, output_dir, crop_size, overlap=0):
        """Version parallélisée du traitement des images"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Collecter toutes les images
        image_paths = []
        for image_name in os.listdir(folder_path):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.tif')):
                image_paths.append(os.path.join(folder_path, image_name))

        total_images = len(image_paths)
        if total_images == 0:
            self.status_updated.emit("Aucune image trouvée dans le dossier")
            return

        self.status_updated.emit(f"Traitement de {total_images} images...")
        processed_images = 0

        # Préparer les arguments pour chaque image
        args_list = [(path, df, output_dir, crop_size, overlap) for path in image_paths]

        # Traitement parallèle des images
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_image_parallel, args) for args in args_list]
            
            for future in concurrent.futures.as_completed(futures):
                processed_images += 1
                progress = int((processed_images / total_images) * 100)
                self.progress_updated.emit(progress)

        self.status_updated.emit(f"Traitement terminé : {processed_images} images traitées")

    def process_csv_parallel(self, file):
        """Traite un fichier CSV en parallèle"""
        try:
            # Essayer différents encodages
            df = None
            encodings = ['utf-8', 'latin1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                self.status_updated.emit(f"Impossible de lire le fichier {file}")
                return None

            # Mots-clés pour identifier les colonnes
            keywords = {
                'Label': ['label', 'class', 'category', 'type', 'name', 'object'],
                'X1': ['x1', 'xmin', 'left', 'bbox_left', 'x_min', 'left_x', 'BX'],
                'Y1': ['y1', 'ymin', 'top', 'bbox_top', 'y_min', 'top_y', 'BY'],
                'Width': ['width', 'w', 'bbox_width', 'Width'],
                'Height': ['height', 'h', 'bbox_height', 'Height']
            }

            # Colonnes requises
            required_columns = {'Label', 'X1', 'Y1', 'X2', 'Y2'}

            # Vérifier si déjà au bon format
            if required_columns.issubset(df.columns):
                return df[list(required_columns)]

            # Vérifier si format YOLO
            if len(df.columns) == 5 and df.iloc[:, 1:].apply(lambda x: 0 <= x <= 1).all().all():
                return convert_yolo_format(df)

            # Identifier les colonnes
            identified_columns = {}
            used_columns = set()
            
            for key, keywords_list in keywords.items():
                for col in df.columns:
                    if col not in used_columns and any(keyword.lower() in col.lower() for keyword in keywords_list):
                        identified_columns[key] = col
                        used_columns.add(col)
                        break

            # Transformer si toutes les colonnes nécessaires sont trouvées
            if len(identified_columns) >= 4:  # Label + X1/Y1/Width/Height
                df_transformed = df[list(identified_columns.values())].copy()
                df_transformed.rename(columns={v: k for k, v in identified_columns.items()}, inplace=True)
                
                # Calculer X2 et Y2
                if 'X2' not in df_transformed.columns:
                    df_transformed['X2'] = df_transformed['X1'] + df_transformed['Width']
                if 'Y2' not in df_transformed.columns:
                    df_transformed['Y2'] = df_transformed['Y1'] + df_transformed['Height']
                
                # Nettoyer les colonnes temporaires
                for col in ['Width', 'Height']:
                    if col in df_transformed.columns:
                        df_transformed.drop(columns=[col], inplace=True)
                
                return df_transformed[list(required_columns)]
            
            self.status_updated.emit(f"Format non reconnu dans {file}")
            return None

        except Exception as e:
            self.status_updated.emit(f"Erreur lors du traitement de {file}: {str(e)}")
            return None

def validate_and_merge_csv(csv_files: List[str]) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Valide et fusionne les fichiers CSV sans sauvegarder.
    
    Args:
        csv_files: Liste des chemins vers les fichiers CSV
        
    Returns:
        Tuple[Optional[pd.DataFrame], str]: (DataFrame fusionné, message d'état)
    """
    # Configuration du logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Mots-clés pour identifier les colonnes
    keywords = {
        'Label': ['label', 'class', 'category', 'type', 'name', 'object'],
        'X1': ['x1', 'xmin', 'left', 'bbox_left', 'x_min', 'left_x', 'BX'],
        'Y1': ['y1', 'ymin', 'top', 'bbox_top', 'y_min', 'top_y', 'BY'],
        'Width': ['width', 'w', 'bbox_width', 'Width'],
        'Height': ['height', 'h', 'bbox_height', 'Height']
    }

    # Colonnes finales requises
    required_final_columns = {'Label', 'X1', 'Y1', 'X2', 'Y2'}
    dataframes = []

    for file in csv_files:
        try:
            # Essayer différents encodages
            df = None
            encodings = ['utf-8', 'latin1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                logging.error(f"Impossible de lire le fichier {file} avec les encodages disponibles")
                continue

            # Afficher les colonnes pour le debug
            logging.info(f"Colonnes trouvées dans {file}: {df.columns.tolist()}")

            # Vérifier si le fichier est déjà au bon format
            if required_final_columns.issubset(df.columns):
                logging.info(f"Le fichier {file} est déjà au bon format.")
                dataframes.append(df[list(required_final_columns)])
                continue

            # Vérifier si c'est au format YOLO
            if len(df.columns) == 5 and df.iloc[:, 1:].apply(lambda x: 0 <= x <= 1).all().all():
                converted_df = convert_yolo_format(df)
                dataframes.append(converted_df)
                logging.info(f"Le fichier {file} a été converti depuis le format YOLO.")
                continue

            # Identifier les colonnes basées sur les mots-clés
            identified_columns = {}
            used_columns = set()  # Pour éviter d'utiliser la même colonne plusieurs fois
            
            for key, keywords_list in keywords.items():
                for col in df.columns:
                    if col not in used_columns and any(keyword.lower() in col.lower() for keyword in keywords_list):
                        identified_columns[key] = col
                        used_columns.add(col)  # Marquer la colonne comme utilisée
                        break

            # Afficher les colonnes identifiées pour le debug
            logging.info(f"Colonnes identifiées dans {file}: {identified_columns}")

            # Vérifier si toutes les colonnes nécessaires ont été identifiées
            if len(identified_columns) >= 4:  # Label + X1/Y1/Width/Height
                df_transformed = df[list(identified_columns.values())].copy()
                df_transformed.rename(columns={v: k for k, v in identified_columns.items()}, inplace=True)
                
                # Calculer X2 et Y2
                if 'X2' not in df_transformed.columns:
                    df_transformed['X2'] = df_transformed['X1'] + df_transformed['Width']
                if 'Y2' not in df_transformed.columns:
                    df_transformed['Y2'] = df_transformed['Y1'] + df_transformed['Height']
                
                # Supprimer les colonnes temporaires
                if 'Width' in df_transformed.columns:
                    df_transformed.drop(columns=['Width'], inplace=True)
                if 'Height' in df_transformed.columns:
                    df_transformed.drop(columns=['Height'], inplace=True)
                
                dataframes.append(df_transformed)
                logging.info(f"Le fichier {file} a été converti avec succès.")
            else:
                logging.warning(f"Format non reconnu dans {file}, colonnes manquantes: {set(keywords.keys()) - set(identified_columns.keys())}")

        except Exception as e:
            logging.error(f"Erreur lors du traitement de {file}: {str(e)}")
            return None, f"Erreur lors du traitement de {file}: {str(e)}"

    if not dataframes:
        return None, "Aucun fichier CSV valide n'a été trouvé."

    try:
        # Fusionner tous les DataFrames
        merged_df = pd.concat(dataframes, ignore_index=True)
        return merged_df, "Les fichiers CSV ont été fusionnés avec succès"

    except Exception as e:
        return None, f"Erreur lors de la fusion des fichiers: {str(e)}"

def convert_yolo_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit un DataFrame du format YOLO vers le format requis.
    
    Args:
        df: DataFrame au format YOLO (class, x_center, y_center, width, height)
        
    Returns:
        DataFrame au format requis (Label, X1, Y1, X2, Y2)
    """
    converted = pd.DataFrame(columns=['Label', 'X1', 'Y1', 'X2', 'Y2'])
    converted['Label'] = df.iloc[:, 0]
    
    # Les coordonnées YOLO sont normalisées (0-1)
    x_center = df.iloc[:, 1]
    y_center = df.iloc[:, 2]
    width = df.iloc[:, 3]
    height = df.iloc[:, 4]
    
    # Conversion en coordonnées absolues
    converted['X1'] = x_center - width/2
    converted['Y1'] = y_center - height/2
    converted['X2'] = x_center + width/2
    converted['Y2'] = y_center + height/2
    
    return converted

def convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height):
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height

def crop_image_and_annotations(image_path, df, output_dir, crop_size, overlap=0):
    # Lire l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Erreur : Impossible de charger l'image {image_path}")
        return  # Sortir de la fonction pour éviter l'erreur
    height, width = image.shape[:2]

    # Filtrer les annotations pour l'image spécifiée
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0] 
    print(image_name)
    filtered_df = df[df['Label'] == image_name]

    annotations = []
    for index, row in filtered_df.iterrows():
        class_id = '0'  # Assurez-vous que cela correspond à votre cas d'utilisation
        x1, y1, x2, y2 = row['X1'], row['Y1'], row['X2'], row['Y2']
        x_center, y_center, bbox_width, bbox_height = convert_to_yolo_format(x1, y1, x2, y2, width, height)
        annotations.append([class_id, x_center, y_center, bbox_width, bbox_height])

    # Découper l'image
    crop_width, crop_height = crop_size
    stride_x = crop_width - overlap
    stride_y = crop_height - overlap

    for y in range(0, height, stride_y):
        for x in range(0, width, stride_x):
            # Calculer les coordonnées de la sous-image
            x_end = min(x + crop_width, width)
            y_end = min(y + crop_height, height)
            cropped_image = image[y:y_end, x:x_end]

            if cropped_image.shape[0] == crop_height and cropped_image.shape[1] == crop_width:
                new_annotations = []
                for ann in annotations:
                    class_id, x_center, y_center, bbox_width, bbox_height = ann
                    x_center_new = (x_center * width - x) / crop_width
                    y_center_new = (y_center * height - y) / crop_height
                    bbox_width_new = bbox_width * width / crop_width
                    bbox_height_new = bbox_height * height / crop_height

                    # Vérifier si l'annotation est dans la sous-image
                    if 0 <= x_center_new <= 1 and 0 <= y_center_new <= 1:
                        new_annotations.append(f"{class_id} {x_center_new} {y_center_new} {bbox_width_new} {bbox_height_new}\n")

                # Enregistrer la sous-image et les nouvelles annotations
                cropped_image_path = os.path.join(output_dir, f'{base_name}_cropped_{x}_{y}.jpg')
                cropped_label_path = os.path.join(output_dir, f'{base_name}_cropped_{x}_{y}.txt')
                cv2.imwrite(cropped_image_path, cropped_image)
                with open(cropped_label_path, 'w') as file:
                    file.writelines(new_annotations)

def process_all_images_in_folder(folder_path, df, output_dir, crop_size, overlap=0):
    """
    Traite toutes les images d'un dossier avec les annotations du DataFrame.
    
    Args:
        folder_path: Chemin du dossier contenant les images
        df: DataFrame contenant les annotations
        output_dir: Dossier de sortie
        crop_size: Tuple (width, height) pour la taille de découpage
        overlap: Chevauchement entre les découpes
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Parcourir chaque image dans le dossier
    for image_name in os.listdir(folder_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg','.JPG','.tif')):
            image_path = os.path.join(folder_path, image_name)
            crop_image_and_annotations(image_path, df, output_dir, crop_size, overlap)

def crop_image_and_annotations_parallel(image_path, df, output_dir, crop_size, overlap=0):
    """Version parallélisée du découpage d'image"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Erreur : Impossible de charger l'image {image_path}")
        return
    height, width = image.shape[:2]

    # Filtrer les annotations
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    filtered_df = df[df['Label'] == image_name]

    # Préparer les annotations YOLO
    annotations = []
    has_annotations = False
    if not filtered_df.empty:
        has_annotations = True
        for index, row in filtered_df.iterrows():
            class_id = '0'
            x1, y1, x2, y2 = row['X1'], row['Y1'], row['X2'], row['Y2']
            x_center, y_center, bbox_width, bbox_height = convert_to_yolo_format(x1, y1, x2, y2, width, height)
            annotations.append([class_id, x_center, y_center, bbox_width, bbox_height])

    # Préparer les coordonnées de découpage
    crop_width, crop_height = crop_size
    stride_x = crop_width - overlap
    stride_y = crop_height - overlap
    
    crop_coordinates = []
    for y in range(0, height, stride_y):
        for x in range(0, width, stride_x):
            x_end = min(x + crop_width, width)
            y_end = min(y + crop_height, height)
            if (x_end - x) == crop_width and (y_end - y) == crop_height:
                crop_coordinates.append((x, y, x_end, y_end))

    def process_crop(coords):
        x, y, x_end, y_end = coords
        cropped_image = image[y:y_end, x:x_end]
        
        # Sauvegarder l'image découpée
        cropped_image_path = os.path.join(output_dir, f'{base_name}_cropped_{x}_{y}.jpg')
        cv2.imwrite(cropped_image_path, cropped_image)

        # Si l'image a des annotations, les traiter et les sauvegarder
        if has_annotations:
            new_annotations = []
            for ann in annotations:
                class_id, x_center, y_center, bbox_width, bbox_height = ann
                x_center_new = (x_center * width - x) / crop_width
                y_center_new = (y_center * height - y) / crop_height
                bbox_width_new = bbox_width * width / crop_width
                bbox_height_new = bbox_height * height / crop_height

                if 0 <= x_center_new <= 1 and 0 <= y_center_new <= 1:
                    new_annotations.append(f"{class_id} {x_center_new} {y_center_new} {bbox_width_new} {bbox_height_new}\n")

            # Sauvegarder les annotations si elles existent
            if new_annotations:
                cropped_label_path = os.path.join(output_dir, f'{base_name}_cropped_{x}_{y}.txt')
                with open(cropped_label_path, 'w') as file:
                    file.writelines(new_annotations)

        return True

    # Traitement parallèle des découpes
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_crop, crop_coordinates))
    
    return len(results)  # Retourne le nombre total de découpes

def validate_and_merge_csv_parallel(csv_files: List[str], progress_callback=None) -> Tuple[Optional[pd.DataFrame], str]:
    processor = DatasetProcessor()
    dataframes = []
    total_files = len(csv_files)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(processor.process_csv_parallel, file) for file in csv_files]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if progress_callback:
                progress = int((i + 1) / total_files * 100)
                progress_callback(progress)
            
            result = future.result()
            if result is not None:
                dataframes.append(result)

    if not dataframes:
        return None, "Aucun fichier CSV valide n'a été trouvé."

    try:
        merged_df = pd.concat(dataframes, ignore_index=True)
        return merged_df, "Les fichiers CSV ont été fusionnés avec succès"
    except Exception as e:
        return None, f"Erreur lors de la fusion des fichiers: {str(e)}"

def organize_dataset_with_txt(image_folder, output_folder, train_ratio=0.8, val_ratio=0.15, annotated_ratio=0.8, non_annotated_ratio=0.2, min_annotation_length=1):
    """
    Organise le dataset en séparant les images annotées et non annotées selon les ratios spécifiés.
    
    Args:
        image_folder: Dossier contenant les images et annotations
        output_folder: Dossier de sortie pour le dataset organisé
        train_ratio: Ratio pour l'ensemble d'entraînement
        val_ratio: Ratio pour l'ensemble de validation
        annotated_ratio: Ratio d'images annotées à inclure
        non_annotated_ratio: Ratio d'images non annotées à inclure
        min_annotation_length: Longueur minimale d'annotation pour considérer une image comme annotée
    """
    # Créer les répertoires de sortie
    image_train_dir = os.path.join(output_folder, 'images/train')
    image_val_dir = os.path.join(output_folder, 'images/val')
    image_test_dir = os.path.join(output_folder, 'images/test')
    label_train_dir = os.path.join(output_folder, 'labels/train')
    label_val_dir = os.path.join(output_folder, 'labels/val')
    label_test_dir = os.path.join(output_folder, 'labels/test')

    for dir_path in [image_train_dir, image_val_dir, image_test_dir,
                    label_train_dir, label_val_dir, label_test_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Lister les images et vérifier les annotations
    annotated_images = []
    non_annotated_images = []
    
    for image_name in os.listdir(image_folder):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            base_name = os.path.splitext(image_name)[0]
            label_path = os.path.join(image_folder, f'{base_name}.txt')
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as file:
                    content = file.read().strip()
                    if len(content) >= min_annotation_length:
                        annotated_images.append(image_name)
                    else:
                        non_annotated_images.append(image_name)
            else:
                non_annotated_images.append(image_name)

    print(f'Total images annotées: {len(annotated_images)}, non annotées: {len(non_annotated_images)}')

    # Sélectionner les images selon les ratios spécifiés
    num_annotated = int(len(annotated_images) * annotated_ratio)
    num_non_annotated = int(len(non_annotated_images) * non_annotated_ratio)

    selected_annotated = annotated_images[:num_annotated]
    selected_non_annotated = non_annotated_images[:num_non_annotated]

    print(f'Images sélectionnées - annotées: {len(selected_annotated)}, non annotées: {len(selected_non_annotated)}')

    # Combiner et mélanger les images sélectionnées
    all_selected_images = selected_annotated + selected_non_annotated

    # Séparer en train/val/test
    train_images, temp_images = train_test_split(
        all_selected_images, 
        train_size=train_ratio,
        random_state=42
    )
    
    val_images, test_images = train_test_split(
        temp_images,
        train_size=val_ratio/(1-train_ratio),
        random_state=42
    )

    print(f'Distribution - Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}')

    # Copier les fichiers dans leurs dossiers respectifs
    for image_name in all_selected_images:
        base_name = os.path.splitext(image_name)[0]
        src_image_path = os.path.join(image_folder, image_name)
        src_label_path = os.path.join(image_folder, f'{base_name}.txt')

        # Déterminer les chemins de destination
        if image_name in train_images:
            dest_image_path = os.path.join(image_train_dir, image_name)
            dest_label_path = os.path.join(label_train_dir, f'{base_name}.txt')
        elif image_name in val_images:
            dest_image_path = os.path.join(image_val_dir, image_name)
            dest_label_path = os.path.join(label_val_dir, f'{base_name}.txt')
        else:  # test_images
            dest_image_path = os.path.join(image_test_dir, image_name)
            dest_label_path = os.path.join(label_test_dir, f'{base_name}.txt')

        # Copier l'image
        shutil.copy(src_image_path, dest_image_path)
        
        # Copier l'annotation si elle existe
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dest_label_path)

    return {
        'total_annotated': len(annotated_images),
        'total_non_annotated': len(non_annotated_images),
        'selected_annotated': len(selected_annotated),
        'selected_non_annotated': len(selected_non_annotated),
        'train': len(train_images),
        'val': len(val_images),
        'test': len(test_images)
    }
