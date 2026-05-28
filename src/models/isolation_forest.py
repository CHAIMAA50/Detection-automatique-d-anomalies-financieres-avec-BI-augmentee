"""
Modèle Isolation Forest pour la détection d'anomalies
"""
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import pandas as pd

class IsolationForestModel:
    """
    Wrapper pour Isolation Forest avec gestion des features
    """
    def __init__(self, params: Dict = None):
        self.params = params or {
            'n_estimators': 200,
            'contamination': 0.002,
            'random_state': 42,
            'n_jobs': -1
        }
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, feature_names: List[str] = None):
        """
        Entraîne le modèle Isolation Forest
        """
        # Standardisation
        X_scaled = self.scaler.fit_transform(X)

        # Entraînement
        self.model = IsolationForest(**self.params)
        self.model.fit(X_scaled)

        self.feature_names = feature_names
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> Dict:
        """
        Prédit les anomalies
        """
        if not self.is_fitted:
            raise ValueError("Modèle non entraîné. Appelez fit() d'abord.")

        X_scaled = self.scaler.transform(X)

        # Scores d'anomalie (négatif = plus anormal)
        scores = self.model.decision_function(X_scaled)

        # Prédictions (-1 = anomalie, 1 = normal)
        predictions = self.model.predict(X_scaled)
        anomalies = np.where(predictions == -1, 1, 0)

        return {
            'scores': -scores,  # Inverser pour que plus grand = plus anormal
            'anomalies': anomalies,
            'threshold': np.percentile(-scores, 99)
        }

    def explain(self, X: np.ndarray, sample_idx: int = 0) -> Dict:
        """
        Explique une prédiction (feature importance simplifiée)
        """
        if not self.is_fitted:
            raise ValueError("Modèle non entraîné")

        X_scaled = self.scaler.transform(X)

        # Importance basée sur la profondeur dans les arbres
        # Approximation: features avec valeurs extrêmes sont plus importantes
        sample = X_scaled[sample_idx]
        importance = np.abs(sample)

        if self.feature_names:
            feature_importance = dict(zip(self.feature_names, importance))
        else:
            feature_importance = {f'feature_{i}': v for i, v in enumerate(importance)}

        return {
            'sample_index': sample_idx,
            'feature_importance': feature_importance,
            'top_features': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        }

    def save(self, filepath: str):
        """Sauvegarde le modèle"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'params': self.params,
            'feature_names': getattr(self, 'feature_names', None)
        }, filepath)

    def load(self, filepath: str):
        """Charge le modèle"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.params = data['params']
        self.feature_names = data.get('feature_names')
        self.is_fitted = True
        return self
