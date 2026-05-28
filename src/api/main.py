"""
API REST pour la détection de fraude
FastAPI + Uvicorn
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
  title="Fraud Detection API",
  description="API pour la détection d'anomalies financières",
  version="1.0.0"
)

# Modèles en mémoire
models = {}

# ============================================================================
# SCHÉMAS DE DONNÉES
# ============================================================================

class Transaction(BaseModel):
  """Schéma d'une transaction"""
  step: int = Field(..., description="Heure de simulation (1-744)")
  type: str = Field(..., description="Type: PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT")
  amount: float = Field(..., description="Montant de la transaction")
  nameOrig: str = Field(..., description="ID client origine")
  oldbalanceOrg: float = Field(..., description="Balance avant transaction")
  newbalanceOrig: float = Field(..., description="Balance après transaction")
  nameDest: str = Field(..., description="ID client destination")
  oldbalanceDest: float = Field(..., description="Balance destination avant")
  newbalanceDest: float = Field(..., description="Balance destination après")

class TransactionBatch(BaseModel):
  """Batch de transactions"""
  transactions: List[Transaction]

class PredictionResponse(BaseModel):
  """Réponse de prédiction"""
  transaction_id: str
  anomaly_score: float
  is_anomaly: bool
  model_used: str
  timestamp: str
  explanation: Optional[Dict] = None

class BatchPredictionResponse(BaseModel):
  """Réponse de prédiction batch"""
  predictions: List[PredictionResponse]
  total_anomalies: int
  processing_time_ms: float

class FeedbackRequest(BaseModel):
  """Requête de feedback"""
  transaction_id: str
  is_fraud: bool
  investigator_notes: Optional[str] = None

class ModelInfo(BaseModel):
  """Information sur un modèle"""
  name: str
  version: str
  status: str
  last_trained: Optional[str] = None

# ============================================================================
# CHARGEMENT DES MODÈLES
# ============================================================================

@app.on_event("startup")
async def load_models():
  """Charge les modèles au démarrage"""
  logger.info("Chargement des modèles...")

  # Chemin des modèles
  #models_dir = os.getenv('MODELS_DIR', './models')
  BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
  models_dir = os.path.join(BASE_DIR, "models")

  try:
    # Charger Isolation Forest
    from src.models.isolation_forest import IsolationForestModel
    iso_model = IsolationForestModel()
    iso_model.load(os.path.join(models_dir, 'isolation_forest.joblib'))
    models['isolation_forest'] = iso_model
    logger.info(" Isolation Forest chargé")
  except Exception as e:
    logger.warning(f" Isolation Forest non chargé: {e}")

  try:
    # Charger Autoencoder
    from src.models.autoencoder import AutoencoderModel
    auto_model = AutoencoderModel()
    auto_model.load(os.path.join(models_dir, 'autoencoder.joblib'))
    models['autoencoder'] = auto_model
    logger.info(" Autoencoder chargé")
  except Exception as e:
    logger.warning(f" Autoencoder non chargé: {e}")

  try:
    # Charger Ensemble
    from src.models.ensemble import EnsembleModel
    ensemble = EnsembleModel()
    ensemble.load(os.path.join(models_dir, 'ensemble.joblib'))
    models['ensemble'] = ensemble
    logger.info(" Ensemble chargé")
  except Exception as e:
    logger.warning(f" Ensemble non chargé: {e}")

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
  """Point d'entrée"""
  return {
    "service": "Fraud Detection API",
    "version": "1.0.0",
    "status": "running",
    "models_loaded": list(models.keys())
  }

@app.get("/health")
async def health_check():
  """Health check"""
  return {
    "status": "healthy",
    "models": len(models),
    "timestamp": datetime.now().isoformat()
  }

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
  """Liste les modèles disponibles"""
  return [
    ModelInfo(
      name=name,
      version="1.0.0",
      status="loaded" if hasattr(model, 'is_fitted') and model.is_fitted else "not_fitted",
      last_trained=datetime.now().isoformat()
    )
    for name, model in models.items()
  ]

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction, model_name: str = "ensemble"):
  """
  Prédit si une transaction est une anomalie
  """
  if model_name not in models:
    raise HTTPException(status_code=404, detail=f"Modèle {model_name} non trouvé")

  model = models[model_name]

  # Convertir en features
  from src.feature_engineering import FeatureEngineer
  fe = FeatureEngineer()

  df = pd.DataFrame([transaction.dict()])
  df_features = fe.transform(df)

  # Sélectionner les features numériques
  feature_cols = fe.get_feature_names()
  X = df_features[feature_cols].fillna(0).values

  # Prédiction
  result = model.predict(X)

  # Explication
  explanation = None
  try:
    explanation = model.explain(X, 0)
  except:
    pass

  return PredictionResponse(
    transaction_id=f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S')}",
    anomaly_score=float(result['scores'][0]),
    is_anomaly=bool(result['anomalies'][0]),
    model_used=model_name,
    timestamp=datetime.now().isoformat(),
    explanation=explanation
  )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: TransactionBatch, model_name: str = "ensemble"):
  """
  Prédit un batch de transactions
  """
  import time
  start_time = time.time()

  if model_name not in models:
    raise HTTPException(status_code=404, detail=f"Modèle {model_name} non trouvé")

  model = models[model_name]

  # Convertir en features
  from src.feature_engineering import FeatureEngineer
  fe = FeatureEngineer()

  df = pd.DataFrame([t.dict() for t in batch.transactions])
  df_features = fe.transform(df)

  feature_cols = fe.get_feature_names()
  X = df_features[feature_cols].fillna(0).values

  # Prédiction
  result = model.predict(X)

  # Construire la réponse
  predictions = []
  for i, txn in enumerate(batch.transactions):
    pred = PredictionResponse(
      transaction_id=f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}",
      anomaly_score=float(result['scores'][i]),
      is_anomaly=bool(result['anomalies'][i]),
      model_used=model_name,
      timestamp=datetime.now().isoformat()
    )
    predictions.append(pred)

  processing_time = (time.time() - start_time) * 1000

  return BatchPredictionResponse(
    predictions=predictions,
    total_anomalies=int(result['anomalies'].sum()),
    processing_time_ms=processing_time
  )

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
  """
  Soumet un feedback sur une prediction et le sauvegarde en CSV.
  """
  feedback_record = {
    'transaction_id': feedback.transaction_id,
    'is_fraud': feedback.is_fraud,
    'notes': feedback.investigator_notes,
    'timestamp': datetime.now().isoformat()
  }

  feedback_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "feedback"
  )
  os.makedirs(feedback_dir, exist_ok=True)
  feedback_path = os.path.join(feedback_dir, "feedback.csv")

  feedback_df = pd.DataFrame([feedback_record])
  feedback_df.to_csv(
    feedback_path,
    mode="a",
    header=not os.path.exists(feedback_path),
    index=False,
    encoding="utf-8"
  )

  logger.info(f"Feedback recu: {feedback.transaction_id} -> {feedback.is_fraud}")

  return {
    "status": "feedback_recorded",
    "transaction_id": feedback.transaction_id,
    "saved_to": feedback_path
  }

@app.get("/stats")
async def get_stats():
  """Statistiques du système"""
  return {
    "total_predictions": 0, # TODO: implémenter le compteur
    "total_anomalies_detected": 0,
    "total_feedback_received": len(pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "feedback", "feedback.csv"))) if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "feedback", "feedback.csv")) else 0,
    "models": list(models.keys()),
    "uptime": "TODO"
  }

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks, model_name: str = "ensemble"):
  """
  Déclenche le réentraînement d'un modèle
  """
  # TODO: Implémenter le réentraînement
  return {"status": "retraining_scheduled", "model": model_name}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
