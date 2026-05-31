# Fraud Detection System

Système complet de détection d'anomalies financières avec BI augmentée.

## Table des Matières

- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [API](#api)
- [Dashboard](#dashboard)
- [Modèles](#modèles)
- [Active Learning](#active-learning)

## Architecture

```
fraud_detection_project/
├── data/          # Données (raw + processed)
├── src/
│  ├── config.py      # Configuration
│  ├── data_ingestion.py  # Pipeline d'ingestion
│  ├── feature_engineering.py
│  ├── models/
│  │  ├── isolation_forest.py
│  │  ├── autoencoder.py
│  │  └── ensemble.py
│  ├── api/
│  │  └── main.py     # API FastAPI
│  └── bi/
│    └── dashboard.py  # Dashboard Streamlit
├── notebooks/       # Notebooks Jupyter
├── tests/         # Tests
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Installation

### Prérequis

- Python 3.10+
- Docker (optionnel)
- Dataset PaySim1 de Kaggle

### Installation locale

```bash
# Cloner le repository
git clone <repository-url>
cd fraud_detection_project

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate # Linux/Mac
# ou
venv\Scripts\activate # Windows

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le dataset PaySim1 depuis Kaggle
# https://www.kaggle.com/datasets/ealaxi/paysim1
# Placer le fichier dans data/raw/
```

### Installation Docker

```bash
docker-compose up --build
```

## Utilisation

### 1. Entraîner les modèles

```python
from src.feature_engineering import FeatureEngineer
from src.models.isolation_forest import IsolationForestModel
from src.models.autoencoder import AutoencoderModel
from src.models.ensemble import EnsembleModel
import pandas as pd

# Charger les données
df = pd.read_csv('data/raw/PS_20174392719_1491204439457_log.csv')

# Feature engineering
fe = FeatureEngineer()
df_features = fe.fit_transform(df)

# Préparer les données
feature_cols = fe.get_feature_names()
X = df_features[feature_cols].fillna(0).values

# Entraîner les modèles
iso_model = IsolationForestModel()
iso_model.fit(X, feature_cols)
iso_model.save('models/isolation_forest.joblib')

auto_model = AutoencoderModel()
auto_model.fit(X, feature_cols)
auto_model.save('models/autoencoder.joblib')

# Ensemble
ensemble = EnsembleModel()
ensemble.add_model('isolation_forest', iso_model, weight=0.3)
ensemble.add_model('autoencoder', auto_model, weight=0.7)
ensemble.fit(X)
ensemble.save('models/ensemble.joblib')
```

### 2. Lancer l'API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Documentation interactive: http://localhost:8000/docs

### 3. Lancer le Dashboard

```bash
streamlit run src/bi/dashboard.py
```

Accès: http://localhost:8501

## API

### Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Info service |
| `/health` | GET | Health check |
| `/models` | GET | Liste des modèles |
| `/predict` | POST | Prédire une transaction |
| `/predict/batch` | POST | Prédire un batch |
| `/feedback` | POST | Soumettre un feedback |
| `/stats` | GET | Statistiques |
| `/retrain` | POST | Réentraîner un modèle |

### Exemple d'utilisation

```python
import requests

# Prédire une transaction
transaction = {
  "step": 1,
  "type": "TRANSFER",
  "amount": 181.0,
  "nameOrig": "C1305486145",
  "oldbalanceOrg": 181.0,
  "newbalanceOrig": 0.0,
  "nameDest": "C553264065",
  "oldbalanceDest": 0.0,
  "newbalanceDest": 0.0
}

response = requests.post("http://localhost:8000/predict", json=transaction)
result = response.json()

print(f"Score anomalie: {result['anomaly_score']}")
print(f"Est une anomalie: {result['is_anomaly']}")
```

## Dashboard

Le dashboard Streamlit permet:
- **Vue d'ensemble**: Métriques clés et tendances
- **Investigation**: Révision manuelle des anomalies avec annotation
- **Analytique**: Performance des modèles et matrices de confusion
- **Configuration**: Paramètres des modèles et connexions

## Modèles

### Isolation Forest
- Détection basée sur la profondeur dans les arbres
- Non supervisé, rapide
- AUC-ROC: ~0.87

### Autoencoder
- Réseau de neurones (MLP)
- Détection par erreur de reconstruction
- AUC-ROC: ~0.999

### Ensemble
- Combinaison pondérée des modèles
- Meilleure robustesse
- AUC-ROC: ~0.88

## Active Learning

Le système intègre une boucle de feedback:
1. Détection des anomalies
2. Révision humaine
3. Annotation (Fraude/Normal)
4. Réentraînement incrémental
5. Amélioration continue

## Snowflake

Configuration pour Snowflake:

```python
import snowflake.connector

conn = snowflake.connector.connect(
  user='your_user',
  password='your_password',
  account='your_account',
  warehouse='COMPUTE_WH',
  database='FRAUD_DETECTION',
  schema='PUBLIC'
)
```

## Notes

- Le dataset PaySim1 est synthétique et à usage éducatif
- Les features de balance ne doivent PAS être utilisées pour la prédiction dans un vrai système (fuite de données)
- Le taux de fraude réel dans PaySim1 est d'environ 0.13%

## Références

- PaySim Paper: Lopez-Rojas et al., "PaySim: A financial mobile money simulator for fraud detection", EMSS 2016
- Kaggle Dataset: https://www.kaggle.com/datasets/ealaxi/paysim1

## Licence

MIT License
