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
    model = YOLO(os.path.join(os.path.dirname(save_path), 'yolo12n.pt'))
    model = YOLO(model_path)
    
    # Vérifier si CUDA est disponible et définir le device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    file_name_without_extension = os.path.splitext(os.path.basename(save_path))[0]
    
    for epoch in range(epochs):
        # Simulate training process
        # Update progress
        if progress_callback:
            progress_callback(f"Époque {epoch + 1}/{epochs} en cours...")
    directory_save_path=os.path.join(os.path.dirname(save_path),file_name_without_extension)
    model.train(
                data=data_config_path,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=device, 
                project=directory_save_path
            )
    # Déplacer et renommer le fichier best.pt
    
    best_model_path = os.path.join(directory_save_path, 'train', 'weights', 'best.pt')
    print(best_model_path)
    if os.path.exists(best_model_path):
        os.rename(best_model_path, save_path)

    # Conserver results.png
    results_png_path = os.path.join(directory_save_path, 'train', 'results.png')
    print(results_png_path)
    if os.path.exists(results_png_path):
        shutil.move(results_png_path, directory)


    # Vérifiez si le dossier existe
    if os.path.exists(directory_save_path):
        # Supprimez le dossier et tout son contenu
        shutil.rmtree(directory_save_path)
        print(f"Le dossier {directory_save_path} a été supprimé avec succès.")
    else:
        print(f"Le dossier {directory_save_path} n'existe pas.")

