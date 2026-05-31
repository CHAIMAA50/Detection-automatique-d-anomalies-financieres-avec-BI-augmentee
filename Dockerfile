# Dockerfile pour le projet de détection de fraude
FROM python:3.10-slim

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copie des requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Variables d'environnement
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Exposition des ports
EXPOSE 8000 8501

# Commande par défaut (API)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
