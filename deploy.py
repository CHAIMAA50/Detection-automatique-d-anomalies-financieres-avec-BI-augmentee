#!/usr/bin/env python3
"""
Script de déploiement complet du système de détection de fraude
Usage: python deploy.py [--train] [--api] [--dashboard] [--docker]
"""
import argparse
import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudDetectionDeployer:
  def __init__(self, project_dir='.'):
    self.project_dir = Path(project_dir)
    self.data_dir = self.project_dir / 'data'
    self.models_dir = self.project_dir / 'models'
    self.src_dir = self.project_dir / 'src'

  def check_environment(self):
    """Vérifie que l'environnement est correctement configuré"""
    logger.info("Vérification de l'environnement...")

    # Vérifier Python
    if sys.version_info < (3, 10):
      logger.error("Python 3.10+ requis")
      return False

    # Vérifier les dépendances
    try:
      import pandas, numpy, sklearn, fastapi, streamlit
      logger.info(" Dépendances Python OK")
    except ImportError as e:
      logger.error(f" Dépendance manquante: {e}")
      logger.info("Exécutez: pip install -r requirements.txt")
      return False

    # Vérifier le dataset
    dataset_path = self.data_dir / 'raw' / 'PS_20174392719_1491204439457_log.csv'
    if not dataset_path.exists():
      logger.warning(f" Dataset non trouvé: {dataset_path}")
      logger.info("Téléchargez depuis: https://www.kaggle.com/datasets/ealaxi/paysim1")
      return False

    logger.info(" Dataset trouvé")
    return True

  def setup_directories(self):
    """Crée la structure de répertoires"""
    dirs = [
      self.data_dir / 'raw',
      self.data_dir / 'processed',
      self.models_dir,
      self.project_dir / 'logs',
      self.project_dir / 'notebooks',
    ]

    for d in dirs:
      d.mkdir(parents=True, exist_ok=True)
      logger.info(f" Répertoire créé: {d}")

  def train_models(self, sample_size=None):
    """Entraîne tous les modèles"""
    logger.info("Démarrage de l'entraînement...")

    import pandas as pd
    from src.feature_engineering import FeatureEngineer
    from src.models.isolation_forest import IsolationForestModel
    from src.models.autoencoder import AutoencoderModel
    from src.models.ensemble import EnsembleModel
    from sklearn.model_selection import train_test_split

    # Charger les données
    dataset_path = self.data_dir / 'raw' / 'PS_20174392719_1491204439457_log.csv'
    logger.info(f"Chargement du dataset: {dataset_path}")

    if sample_size:
      df = pd.read_csv(dataset_path, nrows=sample_size)
      logger.info(f"Dataset chargé (échantillon): {len(df)} lignes")
    else:
      df = pd.read_csv(dataset_path)
      logger.info(f"Dataset chargé: {len(df)} lignes")

    # Feature engineering
    logger.info("Feature engineering...")
    fe = FeatureEngineer()
    df_features = fe.fit_transform(df)

    feature_cols = fe.get_feature_names()
    print("Colonnes disponibles :")
    print(df_features.columns.tolist())

    print("\nColonnes attendues :")
    print(feature_cols)
    X = df_features[feature_cols].fillna(0).values

    # Split train/test
    if 'isFraud' in df.columns:
      y = df['isFraud'].values
      X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
      )
    else:
      X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 1. Isolation Forest
    logger.info("Entraînement Isolation Forest...")
    iso_model = IsolationForestModel()
    iso_model.fit(X_train, feature_cols)
    iso_model.save(self.models_dir / 'isolation_forest.joblib')
    logger.info(" Isolation Forest sauvegardé")

    # 2. Autoencoder
    logger.info("Entraînement Autoencoder...")
    auto_model = AutoencoderModel()
    auto_model.fit(X_train, feature_cols)
    auto_model.save(self.models_dir / 'autoencoder.joblib')
    logger.info(" Autoencoder sauvegardé")

    # 3. Ensemble
    logger.info("Création de l'ensemble...")
    ensemble = EnsembleModel()
    ensemble.add_model('isolation_forest', iso_model, weight=0.3)
    ensemble.add_model('autoencoder', auto_model, weight=0.7)
    ensemble.fit(X_train)
    ensemble.save(self.models_dir / 'ensemble.joblib')
    logger.info(" Ensemble sauvegardé")

    # Évaluation
    logger.info("Évaluation sur le jeu de test...")
    for name, model in [('Isolation Forest', iso_model),
              ('Autoencoder', auto_model),
              ('Ensemble', ensemble)]:
      result = model.predict(X_test)
      logger.info(f" {name}: {result['anomalies'].sum()} anomalies détectées")

    logger.info(" Entraînement terminé!")

  def start_api(self, host='0.0.0.0', port=8000):
    """Démarre l'API FastAPI"""
    logger.info(f"Démarrage de l'API sur {host}:{port}...")

    env = os.environ.copy()
    env['MODELS_DIR'] = str(self.models_dir)
    env['API_HOST'] = host
    env['API_PORT'] = str(port)

    subprocess.Popen([
      sys.executable, '-m', 'uvicorn',
      'src.api.main:app',
      '--host', host,
      '--port', str(port),
      '--reload'
    ], env=env)

    logger.info(f" API démarrée: http://{host}:{port}")
    logger.info(f" Documentation: http://{host}:{port}/docs")

  def start_dashboard(self, port=8501):
    """Démarre le dashboard Streamlit"""
    logger.info(f"Démarrage du dashboard sur le port {port}...")

    env = os.environ.copy()
    env['API_URL'] = f"http://localhost:8000"

    subprocess.Popen([
      sys.executable, '-m', 'streamlit', 'run',
      'src/bi/dashboard.py',
      '--server.port', str(port),
      '--server.address', '0.0.0.0'
    ], env=env)

    logger.info(f" Dashboard démarré: http://localhost:{port}")

  def start_docker(self):
    """Démarre avec Docker Compose"""
    logger.info("Démarrage avec Docker Compose...")

    result = subprocess.run(
      ['docker-compose', 'up', '--build', '-d'],
      capture_output=True, text=True
    )

    if result.returncode == 0:
      logger.info(" Services Docker démarrés")
      logger.info(" API: http://localhost:8000")
      logger.info(" Dashboard: http://localhost:8501")
    else:
      logger.error(f" Erreur Docker: {result.stderr}")

def main():
  parser = argparse.ArgumentParser(description='Déployer le système de détection de fraude')
  parser.add_argument('--setup', action='store_true', help='Configurer l environnement')
  parser.add_argument('--train', action='store_true', help='Entraîner les modèles')
  parser.add_argument('--sample', type=int, help='Taille d échantillon pour l entraînement')
  parser.add_argument('--api', action='store_true', help='Démarrer l API')
  parser.add_argument('--dashboard', action='store_true', help='Démarrer le dashboard')
  parser.add_argument('--docker', action='store_true', help='Démarrer avec Docker')
  parser.add_argument('--all', action='store_true', help='Tout démarrer')

  args = parser.parse_args()

  deployer = FraudDetectionDeployer()

  if args.setup or args.all:
    deployer.setup_directories()
    if not deployer.check_environment():
      sys.exit(1)

  if args.train or args.all:
    deployer.train_models(sample_size=args.sample)

  if args.api or args.all:
    deployer.start_api()

  if args.dashboard or args.all:
    deployer.start_dashboard()

  if args.docker:
    deployer.start_docker()

  if not any([args.setup, args.train, args.api, args.dashboard, args.docker, args.all]):
    parser.print_help()

if __name__ == '__main__':
  main()
