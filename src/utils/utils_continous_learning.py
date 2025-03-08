from ultralytics import YOLO
import torch
def continual_learning_yolo(model_path, data_config_path, epochs=10, img_size=640, batch_size=16, save_path='backup_model.pt'):
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
    model.train(
        data=data_config_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device, 
        save=save_path
    )
    
   

