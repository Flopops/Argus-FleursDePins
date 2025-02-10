# Projet ARGUS - LaBRI

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation de l'application

### Windows

**Pré-requis :** Installation de python 3.10 et vérification des ExecutionPolicy.  
Vérifier sur le [site](https://www.python.org/downloads/windows/) de python, mais normalement ce [lien](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)
lance le téléchargement de l'installeur python 3.10.  
Faire ` Get-ExecutionPolicy -List` pour vérifier l'ExecutionPolicy de CurrentUser.  
Si celle-ci est à Undefined faire la commande suivante (pas nécessaire d'être administrateur) : `Set-ExecutionPolicy  -Scope CurrentUser RemoteSigned -Force`. Cette commande permet d'autoriser l'exécution des scripts locaux sur la machine.

Une fois python installé, exécuter le fichier `run.ps1` en faisant clic droit puis "Exécuter en tant que script power shell".

Le script va installer les dépendances dont l'application a besoin pour fonctionner et lancera l'application tout seul.

Pour relancer l'application, il suffit de re-exécuter le script avec le clic droit.

### Linux

**Pré-requis :** Installation de python 3.10.  
```shell
sudo apt install python3.10  # Par exemple
```  

Une fois python installé si vous ne l'aviez pas déjà, exécuter le script `run.sh`.

Le script va installer les dépendances dont l'application a besoin pour fonctionner et lancera l'application tout seul.

Pour relancer l'application, il suffit de re-exécuter le script ou bien de faire 
```shell
python src/app.py
```

## Utilisation de l'application

Faire glisser un dossier d'images et cliquer sur "Lancer la détection".

Un modèle d'intelligence artificielle va compter les fleurs sur les images contenues dans le dossier.

Une fois la détection terminée, cliquer sur "Exporter dans un excel".
Un fichier excel sera ainsi créé à côté du dossier glissé.

## Dev

### Installation de l'environnement de développement

Installer tout d'abord conda et lancer la commande suivante :

```shell
conda env create -f environment.yml
conda activate argus
```

### Obfuscate

Pour lancer la génération du dossier `prod` :
- se mettre sur linux et lancer `obfuscate_to_prod.sh`
- se mettre sur windows et lancer `obfuscate_to_prod.ps1`

La commande `pyarmor étant dans les deux requirements.txt`