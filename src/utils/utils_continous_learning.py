from ultralytics import YOLO
import torch
import shutil
import os
import csv
def continual_learning_yolo(model_path, data_config_path, epochs=10, img_size=640, batch_size=16, save_path='backup_model.pt',directory='', progress_callback=None):
    """
    Effectue l'apprentissage continu sur un modèle YOLOv12n pré-entraîné avec de nouvelles données.

    :param model_path: Chemin vers le modèle YOLOv12n pré-entraîné.
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
        # Validate the model on the test dataset
    metrics = model.val(data=data_config_path, imgsz=img_size, conf=0.25, iou=0.45, split='test', device=device)
    # Conserver results.png
    results_png_path = os.path.join(directory_save_path, 'train', 'results.png')
    print(results_png_path)
    if os.path.exists(results_png_path):
        os.makedirs(os.path.join(directory, 'results'), exist_ok=True)
        os.rename(results_png_path, os.path.join(directory, 'results',f'results_{file_name_without_extension}.png'))
    # Enregistrer les métriques dans un fichier CSV
    results_csv_path = os.path.join(directory, 'results', f'results_test_{file_name_without_extension}.csv')
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)

    with open(results_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Metric", "Value"])  # En-tête

        # Ajouter les métriques principales
        writer.writerow(["mAP50-95", metrics.box.map])
        writer.writerow(["mAP50", metrics.box.map50])
        writer.writerow(["mAP75", metrics.box.map75])

        # Ajouter les mAP par catégorie
        for idx, map_value in enumerate(metrics.box.maps):
            writer.writerow([f"mAP50-95 catégorie {idx}", map_value])

    print(f"Métriques enregistrées dans {results_csv_path}")