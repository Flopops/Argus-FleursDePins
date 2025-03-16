from ultralytics import YOLO
import torch
import shutil
import os

def continual_learning_yolo(model_path, data_config_path, epochs=10, img_size=640, batch_size=16, save_path='backup_model.pt',directory='', progress_callback=None):
    """
    Effectue l'apprentissage continu sur un modèle YOLOv8 pré-entraîné avec de nouvelles données.

    :param model_path: Chemin vers le modèle YOLOv8 pré-entraîné.
    :param data_config_path: Chemin vers le fichier de configuration des nouvelles données.
    :param epochs: Nombre d'époques d'entraînement.
    :param img_size: Taille des images d'entrée.
    :param batch_size: Taille du batch pour l'entraînement.
    :param save_path: Chemin où sauvegarder le modèle avant réentraînement.
    """
    # Charger le modèle Yolo pré-entraîné
    model = YOLO(model_path)

    # Vérifier si CUDA est disponible et définir le device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Entraîner le modèle avec les nouvelles données
    for epoch in range(epochs):
        # Simulate training process
        # Update progress
        if progress_callback:
            progress_callback(f"Époque {epoch + 1}/{epochs} en cours...")
        model.train(
            data=data_config_path,
            epochs=1,
            imgsz=img_size,
            batch=batch_size,
            device=device, 
            project=save_path
        )

    # Déplacer le fichier best.pt
    best_model_path = os.path.join(os.path.dirname(save_path), 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        shutil.move(best_model_path, directory)

    # Conserver results.png
    results_png_path = os.path.join(os.path.dirname(save_path), 'results.png')
    if os.path.exists(results_png_path):
        shutil.copy(results_png_path, os.path.dirname(directory))

    '''# Supprimer le dossier save_path
    if os.path.exists(os.path.dirname(save_path)):
        shutil.rmtree(os.path.dirname(save_path))'''
    
   

