from ultralytics import YOLO

def continual_learning_yolov8(model_path, data_config_path, epochs=10, img_size=640, batch_size=16):
    """
    Effectue l'apprentissage continu sur un modèle YOLOv8 pré-entraîné avec de nouvelles données.

    :param model_path: Chemin vers le modèle YOLOv8 pré-entraîné.
    :param data_config_path: Chemin vers le fichier de configuration des nouvelles données.
    :param epochs: Nombre d'époques d'entraînement.
    :param img_size: Taille des images d'entrée.
    :param batch_size: Taille du batch pour l'entraînement.
    """
    # Charger le modèle YOLOv8 pré-entraîné
    model = YOLO(model_path)

    # Entraîner le modèle avec les nouvelles données
    model.train(
        data=data_config_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        cache=True  # Utiliser le cache pour accélérer l'entraînement
    )
