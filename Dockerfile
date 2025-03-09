# Image de base avec Python
FROM python:3.9-slim

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5core5a \
    libx11-xcb1 \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/ ./src/
COPY config/ ./config/

# Variables d'environnement pour PyQt
ENV QT_QPA_PLATFORM=xcb
ENV PYTHONPATH=/app
ENV QT_DEBUG_PLUGINS=1

# Port si nécessaire
EXPOSE 5000

# Commande par défaut
CMD ["python", "src/main.py"] 