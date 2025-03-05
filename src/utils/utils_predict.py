from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torchvision
from sklearn.cluster import DBSCAN
# Charger le modèle YOLO avec les poids personnalisés
model = YOLO('best.pt')
def predict_image(original_image_path,model):
    


    original_image = cv2.imread(original_image_path)

    # Dimensions de l'image originale
    original_height, original_width = original_image.shape[:2]

    # Dimensions des parties découpées
    tile_size = 640

    # Liste pour stocker les résultats des prédictions en coordonnées globales
    all_detections = []

    # Parcourir chaque partie de l'image
    for y in range(0, original_height, tile_size):
        for x in range(0, original_width, tile_size):
            # Découper la partie de l'image
            cropped_image = original_image[y:y+tile_size, x:x+tile_size]

            # Vérifier que la découpe est valide
            if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                continue  

            # Effectuer la prédiction sur la partie découpée
            results = model(cropped_image)

            # Récupérer les boîtes englobantes
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convertir en liste de valeurs
                    score = box.conf[0].item()    # Confiance
                    class_id = int(box.cls[0].item())  # Classe détectée

                    # Convertir en coordonnées globales
                    x1_global, y1_global, x2_global, y2_global = x1 + x, y1 + y, x2 + x, y2 + y

                    # Ajouter aux détections globales
                    all_detections.append([x1_global, y1_global, x2_global, y2_global, score, class_id])

    # ---------- SUPPRESSION DES DOUBLONS AVANCÉE ----------
    if len(all_detections) > 0:
        boxes = np.array([[det[0], det[1], det[2], det[3]] for det in all_detections])
        scores = np.array([det[4] for det in all_detections])
        class_ids = np.array([det[5] for det in all_detections])

        # Convertir en (x_centre, y_centre, largeur, hauteur)
        centers = np.array([[ (b[0] + b[2]) / 2, (b[1] + b[3]) / 2 ] for b in boxes])


        # Appliquer DBSCAN pour grouper les boîtes proches
        clustering = DBSCAN(eps=30, min_samples=1).fit(centers)

        # Fusionner les boîtes du même groupe
        unique_clusters = np.unique(clustering.labels_)
        final_detections = []

        for cluster in unique_clusters:
            indices = np.where(clustering.labels_ == cluster)[0]
            cluster_boxes = boxes[indices]

            # Trouver les coordonnées min/max pour fusionner
            x1_min, y1_min = np.min(cluster_boxes[:, 0:2], axis=0)
            x2_max, y2_max = np.max(cluster_boxes[:, 2:4], axis=0)
            best_score = np.max(scores[indices])
            class_id = class_ids[indices][np.argmax(scores[indices])]

            final_detections.append([x1_min, y1_min, x2_max, y2_max, best_score, class_id])

    else:
        final_detections = []

    # ---------- DESSINER LES BOÎTES FINALES ----------
    annotated_image = np.copy(original_image)

    for x1, y1, x2, y2, score, class_id in final_detections:
        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    annotation_counts = {
        'pine_flowers': len(final_detections)
    }
    # Sauvegarder l'image complète avec les annotations
    #cv2.imwrite('output_image_with_annotations_merged.jpg', annotated_image)
    return annotation_counts

