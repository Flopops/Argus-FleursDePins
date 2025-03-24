# Argus-FleursDePins 🌲

## Description

Argus-FleursDePins est une application de détection et d'analyse des fleurs de pins utilisant l'intelligence artificielle. Cette application permet de détecter automatiquement les fleurs de pins dans des images, de les annoter et d'effectuer des analyses quantitatives.

## Fonctionnalités

- 🔍 Détection automatique des fleurs de pins
- 📊 Analyse quantitative des détections
- 🖼️ Interface graphique intuitive
- 🔄 Apprentissage continu du modèle
- 📈 Visualisation des résultats
- 🎯 Prédiction en temps réel

## Prérequis

- Python 3.9 ou supérieur
- CUDA compatible GPU (recommandé pour de meilleures performances)
- Docker (optionnel, pour l'utilisation en conteneur)

## Installation

### Option 1 : Installation locale

1. Cloner le repository :

```bash
git clone https://github.com/Flopops/Argus-FleursDePins.git
cd Argus-FleursDePins
```

2. Créer un environnement virtuel (recommandé) :

```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate  # Sur Windows
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

### Option 2 : Installation via Docker

1. Construire l'image Docker :

```bash
docker-compose build
```

2. Lancer l'application :

```bash
docker-compose up
```

## Structure du projet

```
Argus-FleursDePins/
├── config/
│   └── model_config.json
├── pretrained_models/
│   ├── yolo12n.pt
│   ├── best_petite_fleurs_de_pins.pt
│   └── best_grosse_moyenne_fleurs_de_pins.pt
├── src/
│   ├── utils/
│   │   ├── utils_predict.py
│   │   ├── utils_augmentation_data.py
│   │   ├── utils_continous_learning.py
│   │   └── utils_dataset.py
│   ├── ui/
│   │   ├── window_cl.py
│   │   ├── window_dataset.py
│   │   └── window_predict.py
│   └── main.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Utilisation

### Lancement de l'application

```bash
python src/main.py
```
- Si vous êtes sur Windows dans src/main.py supprimer ces deux lignes
    ```python
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/qt5/plugins"
    ```
### Fonctionnalités principales

#### 1. Détection des fleurs de pins

- Chargez une image ou un dossier d'images
- L'application détectera automatiquement les fleurs de pins
- Les résultats seront affichés avec des annotations visuelles

#### 2. Apprentissage continu

- Accédez à l'interface d'apprentissage continu
- Chargez de nouvelles données d'entraînement
- Configurez les paramètres d'apprentissage (époques, taille du batch, etc.)
- Lancez l'apprentissage du modèle

#### 3. Gestion des données

- Préparez vos données d'entraînement
- Utilisez les outils d'augmentation de données
- Organisez votre dataset en ensembles d'entraînement, validation et test

## Configuration du modèle

Le fichier `config/model_config.json` permet de configurer :

- Le répertoire des modèles pré-entraînés
- Le modèle sélectionné pour la détection

## Modèles pré-entraînés

Trois modèles sont disponibles dans le dossier `pretrained_models/` :

- `yolo12n.pt` : Modèle de base YOLO
- `best_petite_fleurs_de_pins.pt` : Modèle optimisé pour les petites fleurs
- `best_grosse_moyenne_fleurs_de_pins.pt` : Modèle optimisé pour les grandes et moyennes fleurs

## Dépendances principales

- ultralytics : Pour l'utilisation de YOLO
- opencv-python-headless : Pour le traitement d'images
- torch et torchvision : Pour le deep learning
- scikit-learn : Pour les outils de machine learning
- pandas : Pour la manipulation des données
- PyQt5 : Pour l'interface graphique

## Auteurs

Lucien Lachaud
Anatole Garnier
Florian Grenier
Alex Dumerc

