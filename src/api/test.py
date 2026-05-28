"""
Tests unitaires pour l'API de détection de fraude
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import numpy as np

client = TestClient(app)

class TestAPI:
  def test_root(self):
    """Test du point d'entrée"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Fraud Detection API" in response.json()["service"]

  def test_health(self):
    """Test du health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

  def test_models_list(self):
    """Test de la liste des modèles"""
    response = client.get("/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

  def test_predict_single(self):
    """Test de prédiction unitaire"""
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

    response = client.post("/predict?model_name=ensemble", json=transaction)
    assert response.status_code == 200

    result = response.json()
    assert "anomaly_score" in result
    assert "is_anomaly" in result
    assert 0 <= result["anomaly_score"] <= 1
    assert isinstance(result["is_anomaly"], bool)

  def test_predict_batch(self):
    """Test de prédiction batch"""
    transactions = {
      "transactions": [
        {
          "step": 1,
          "type": "TRANSFER",
          "amount": 181.0,
          "nameOrig": "C1305486145",
          "oldbalanceOrg": 181.0,
          "newbalanceOrig": 0.0,
          "nameDest": "C553264065",
          "oldbalanceDest": 0.0,
          "newbalanceDest": 0.0
        },
        {
          "step": 1,
          "type": "PAYMENT",
          "amount": 9839.64,
          "nameOrig": "C153514495",
          "oldbalanceOrg": 170136.0,
          "newbalanceOrig": 160296.36,
          "nameDest": "M1979787155",
          "oldbalanceDest": 0.0,
          "newbalanceDest": 0.0
        }
      ]
    }

    response = client.post("/predict/batch?model_name=ensemble", json=transactions)
    assert response.status_code == 200

    result = response.json()
    assert "predictions" in result
    assert "total_anomalies" in result
    assert len(result["predictions"]) == 2

  def test_feedback(self):
    """Test de soumission de feedback"""
    feedback = {
      "transaction_id": "TXN_20240101120000",
      "is_fraud": True,
      "investigator_notes": "Fraude confirmée - pattern suspect"
    }

    response = client.post("/feedback", json=feedback)
    assert response.status_code == 200
    assert response.json()["status"] == "feedback_recorded"

  def test_stats(self):
    """Test des statistiques"""
    response = client.get("/stats")
    assert response.status_code == 200
    assert "models" in response.json()

class TestModels:
  def test_isolation_forest(self):
    """Test du modèle Isolation Forest"""
    from src.models.isolation_forest import IsolationForestModel

    model = IsolationForestModel()
    X = np.random.randn(100, 10)

    model.fit(X)
    result = model.predict(X[:5])

    assert "scores" in result
    assert "anomalies" in result
    assert len(result["scores"]) == 5

  def test_autoencoder(self):
    """Test du modèle Autoencoder"""
    from src.models.autoencoder import AutoencoderModel

    model = AutoencoderModel()
    X = np.random.randn(100, 10)

    model.fit(X)
    result = model.predict(X[:5])

    assert "scores" in result
    assert "anomalies" in result
    assert len(result["scores"]) == 5

  def test_ensemble(self):
    """Test du modèle Ensemble"""
    from src.models.ensemble import EnsembleModel
    from src.models.isolation_forest import IsolationForestModel
    from src.models.autoencoder import AutoencoderModel

    X = np.random.randn(100, 10)

    iso = IsolationForestModel()
    iso.fit(X)

    auto = AutoencoderModel()
    auto.fit(X)

    ensemble = EnsembleModel()
    ensemble.add_model("iso", iso, 0.5)
    ensemble.add_model("auto", auto, 0.5)
    ensemble.fit(X)

    result = ensemble.predict(X[:5])
    assert "scores" in result
    assert len(result["scores"]) == 5

class TestFeatureEngineering:
  def test_feature_engineer(self):
    """Test du feature engineering"""
    import pandas as pd
    from src.feature_engineering import FeatureEngineer

    df = pd.DataFrame({
      'step': [1, 2],
      'type': ['TRANSFER', 'PAYMENT'],
      'amount': [100.0, 200.0],
      'nameOrig': ['C123', 'C456'],
      'oldbalanceOrg': [100.0, 200.0],
      'newbalanceOrig': [0.0, 150.0],
      'nameDest': ['C789', 'M123'],
      'oldbalanceDest': [0.0, 0.0],
      'newbalanceDest': [100.0, 0.0]
    })

    fe = FeatureEngineer()
    result = fe.fit_transform(df)

    assert len(result.columns) > len(df.columns)
    assert 'amount_log' in result.columns
    assert 'is_cash_out' in result.columns

if __name__ == '__main__':
  pytest.main([__file__, '-v'])
