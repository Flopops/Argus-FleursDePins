import cv2
import numpy as np

def apply_gamma_correction(image, gamma=0.4):
    """Applies gamma correction to an image."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def flip_image(image, flip_code):
    """Retourne l'image horizontalement ou verticalement"""
    return cv2.flip(image, flip_code)

def rotate_image(image, angle):
    """Fait pivoter l'image d'un angle donné"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))
