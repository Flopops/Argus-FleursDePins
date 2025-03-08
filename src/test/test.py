import unittest
import os
import json
import pandas as pd
from PyQt5.QtWidgets import QApplication
import sys

# Ajouter le chemin parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration de l'environnement Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Utiliser le backend offscreen pour les tests

# Créer une seule instance de QApplication pour tous les tests
app = QApplication(sys.argv)

# Imports absolus au lieu d'imports relatifs
from ui.window_dataset import DatasetUI
from ui.window_cl import ContinuousLearningUI
from utils.utils_continous_learning import continual_learning_yolo
from utils.utils_predict import predict_image

class TestDatasetUI(unittest.TestCase):
    def setUp(self):
        # Initialiser l'interface pour chaque test
        self.dataset_ui = DatasetUI()
        
        # Créer des fichiers temporaires pour les tests
        self.test_csv = "src/test/data/test.csv"
        self.test_image = "src/test/data/test.JPG"
        os.makedirs("src/test/data", exist_ok=True)
        
        # Créer un CSV de test
        df = pd.DataFrame({
            'Label': [0],
            'X1': [0.1],
            'Y1': [0.2],
            'X2': [0.3],
            'Y2': [0.4]
        })
        df.to_csv(self.test_csv, index=False)

    def test_csv_validation(self):
        """Test la validation du format CSV"""
        self.assertTrue(self.dataset_ui.validate_csv(self.test_csv))

    def test_yolo_conversion(self):
        """Test la conversion du format YOLO"""
        df = pd.DataFrame({
            'class': [0],
            'x_center': [0.5],
            'y_center': [0.5],
            'width': [0.2],
            'height': [0.2]
        })
        converted = self.dataset_ui.convert_yolo_format(df)
        self.assertEqual(list(converted.columns), ['Label', 'X1', 'Y1', 'X2', 'Y2'])

    def tearDown(self):
        # Nettoyer les fichiers de test
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
        if os.path.exists(self.test_image):
            os.remove(self.test_image)
        if os.path.exists("src/test/data"):
            os.rmdir("src/test/data")

class TestContinuousLearning(unittest.TestCase):
    def setUp(self):
        # Initialiser l'interface pour chaque test
        self.cl_ui = ContinuousLearningUI()
        
        # Créer des fichiers de configuration temporaires
        self.config_dir = "src/test/config"
        self.config_file = os.path.join(self.config_dir, "model_config.json")
        os.makedirs(self.config_dir, exist_ok=True)
        
        config = {
            "models_directory": "src/test/models",
            "selected_model": "test_model.pt"
        }
        os.makedirs("src/test/models", exist_ok=True)
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f)

    def test_config_loading(self):
        """Test le chargement de la configuration"""
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        self.assertEqual(config["models_directory"], "src/test/models")
        self.assertEqual(config["selected_model"], "test_model.pt")

    def test_training_parameters(self):
        """Test la validation des paramètres d'entraînement"""
        self.assertEqual(self.cl_ui.epochs_input.value(), 100)  # Valeur par défaut
        self.assertEqual(self.cl_ui.img_size_input.value(), 640)
        self.assertEqual(self.cl_ui.batch_size_input.value(), 16)

    def tearDown(self):
        pass

class TestPrediction(unittest.TestCase):
    def setUp(self):
        # Créer un fichier de configuration temporaire et le dossier data
        os.makedirs("src/test/config", exist_ok=True)
        os.makedirs("src/test/data", exist_ok=True)
        
        # Créer une image de test vide
        with open("src/test/data/test.JPG", "wb") as f:
            f.write(b"")
            
        self.config = {
            "models_directory": "src/test/models",
            "selected_model": "test_model.pt"
        }
        with open("src/test/config/model_config.json", 'w') as f:
            json.dump(self.config, f)

    def test_prediction_parameters(self):
        """Test la validation des paramètres de prédiction"""
        # Test avec un fichier image inexistant
        with self.assertRaises(FileNotFoundError):
            predict_image("nonexistent.jpg")
            
        # Test avec un modèle invalide (devrait retourner None)
        result = predict_image("src/test/data/test.JPG")
        self.assertIsNone(result)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main() 