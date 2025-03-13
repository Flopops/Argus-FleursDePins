import unittest
import os
import json
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest

# Ajouter le chemin parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration de l'environnement Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Créer une instance de QApplication
app = QApplication(sys.argv)

# Imports des modules à tester
from ui.window_dataset import DatasetUI, DatasetProcessor
from ui.window_cl import ContinuousLearningUI
from ui.window_predict import PredictUI
from ui.window_model import ModelSelectorUI
from utils.utils_continous_learning import continual_learning_yolo
from utils.utils_predict import predict_image, calculate_adaptive_eps, update_model
from utils.utils_dataset import validate_and_merge_csv, convert_to_yolo_format, organize_dataset_with_txt

class TestDatasetProcessing(unittest.TestCase):
    def setUp(self):
        self.processor = DatasetProcessor()
        self.test_dir = "src/test/test_data"
        os.makedirs(self.test_dir, exist_ok=True)

    def test_csv_processing(self):
        # Créer un CSV de test
        test_csv = os.path.join(self.test_dir, "test.csv")
        df = pd.DataFrame({
            'Label': ['image1.jpg'],
            'X1': [100],
            'Y1': [100],
            'X2': [200],
            'Y2': [200]
        })
        df.to_csv(test_csv, index=False)

        # Tester le traitement du CSV
        result = self.processor.process_csv_parallel(test_csv)
        self.assertIsNotNone(result)
        self.assertEqual(list(result.columns), ['Label', 'X1', 'Y1', 'X2', 'Y2'])

    def test_yolo_format_conversion(self):
        # Test de conversion des coordonnées
        x1, y1, x2, y2 = 100, 100, 200, 200
        img_width, img_height = 640, 640
        x_center, y_center, width, height = convert_to_yolo_format(
            x1, y1, x2, y2, img_width, img_height
        )
        
        self.assertTrue(0 <= x_center <= 1)
        self.assertTrue(0 <= y_center <= 1)
        self.assertTrue(0 <= width <= 1)
        self.assertTrue(0 <= height <= 1)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)

class TestPredictionSystem(unittest.TestCase):
    def setUp(self):
        self.test_dir = "src/test/test_data"
        os.makedirs(self.test_dir, exist_ok=True)

    def test_adaptive_eps_calculation(self):
        # Test avec différentes configurations de centres
        centers = np.array([[0, 0], [10, 10], [100, 100]])
        eps = calculate_adaptive_eps(centers)
        self.assertIsInstance(eps, (int, float))
        self.assertTrue(eps > 0)

    def test_model_update(self):
        # Créer un faux modèle pour le test
        test_model_path = os.path.join(self.test_dir, "test_model.pt")
        with open(test_model_path, 'w') as f:
            f.write("dummy model")

        # Test de la mise à jour du modèle
        result = update_model(test_model_path)
        self.assertFalse(result)  # Devrait échouer car ce n'est pas un vrai modèle

    def tearDown(self):
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)

class TestContinuousLearning(unittest.TestCase):
    def setUp(self):
        self.cl_ui = ContinuousLearningUI()
        self.test_dir = "src/test/test_data"
        os.makedirs(self.test_dir, exist_ok=True)

    def test_parameter_validation(self):
        # Test des valeurs par défaut
        self.assertEqual(self.cl_ui.epochs_input.value(), 100)
        self.assertEqual(self.cl_ui.img_size_input.value(), 640)
        self.assertEqual(self.cl_ui.batch_size_input.value(), 16)

        # Test des limites
        self.cl_ui.epochs_input.setValue(0)
        self.assertTrue(self.cl_ui.epochs_input.value() >= 1)
        
        self.cl_ui.img_size_input.setValue(1500)
        self.assertTrue(self.cl_ui.img_size_input.value() <= 1280)

    def test_yaml_loading(self):
        # Créer un fichier YAML de test
        test_yaml = os.path.join(self.test_dir, "test.yaml")
        with open(test_yaml, 'w') as f:
            f.write("test: value")

        # Simuler le chargement du fichier
        self.cl_ui.yaml_path = test_yaml
        self.assertTrue(os.path.exists(self.cl_ui.yaml_path))

    def tearDown(self):
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)

class TestModelSelector(unittest.TestCase):
    def setUp(self):
        self.model_ui = ModelSelectorUI()
        self.test_dir = "src/test/test_data"
        os.makedirs(self.test_dir, exist_ok=True)

    def test_config_handling(self):
        # Test de sauvegarde de configuration
        test_config = {
            'models_directory': self.test_dir,
            'selected_model': 'test_model.pt'
        }
        
        os.makedirs("config", exist_ok=True)
        with open(self.model_ui.CONFIG_FILE, 'w') as f:
            json.dump(test_config, f)

        # Recharger la configuration
        self.model_ui.load_config()
        self.assertEqual(self.model_ui.models_dir, self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)
        if os.path.exists("config"):
            import shutil
            shutil.rmtree("config")

class TestPredictUI(unittest.TestCase):
    def setUp(self):
        self.predict_ui = PredictUI()
        self.test_dir = "src/test/test_data"
        os.makedirs(self.test_dir, exist_ok=True)

    def test_ui_state(self):
        # Test de l'état initial
        self.assertFalse(self.predict_ui.predict_button.isEnabled())
        
        # Simuler le chargement d'une image
        test_image = os.path.join(self.test_dir, "test.jpg")
        with open(test_image, 'w') as f:
            f.write("dummy image")
        
        self.predict_ui.image_paths.append(test_image)
        self.predict_ui.on_loading_finished()
        
        # Le bouton devrait être activé
        self.assertTrue(self.predict_ui.predict_button.isEnabled())

    def test_checkbox_state(self):
        # Test de l'état de la checkbox
        self.assertFalse(self.predict_ui.save_annotations_checkbox.isChecked())
        
        # Simuler un clic
        QTest.mouseClick(self.predict_ui.save_annotations_checkbox, Qt.LeftButton)
        self.assertTrue(self.predict_ui.save_annotations_checkbox.isChecked())

    def tearDown(self):
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main() 