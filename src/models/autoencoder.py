"""
Autoencoder pour la détection d'anomalies
"""
import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List

class AutoencoderModel:
    """
    Autoencoder basé sur MLPRegressor pour la détection d'anomalies
    """
    def __init__(self, params: Dict = None):
        self.params = params or {
            'hidden_layer_sizes': (32, 16, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 256,
            'learning_rate': 'adaptive',
            'max_iter': 200,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
            'random_state': 42
        }
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.reconstruction_error_threshold = None

    def fit(self, X: np.ndarray, feature_names: List[str] = None):
        """
        Entraîne l'autoencoder sur les données normales
        """
        # Standardisation
        X_scaled = self.scaler.fit_transform(X)

        # Entraînement (apprentissage de l'identité)
        self.model = MLPRegressor(**self.params)
        self.model.fit(X_scaled, X_scaled)

        # Calcul du seuil d'erreur sur les données d'entraînement
        X_pred = self.model.predict(X_scaled)
        errors = np.mean(np.square(X_scaled - X_pred), axis=1)
        self.reconstruction_error_threshold = np.percentile(errors, 99.5)

        self.feature_names = feature_names
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> Dict:
        """
        Prédit les anomalies basées sur l'erreur de reconstruction
        """
        if not self.is_fitted:
            raise ValueError("Modèle non entraîné. Appelez fit() d'abord.")

        X_scaled = self.scaler.transform(X)

        # Reconstruction
        X_pred = self.model.predict(X_scaled)

        # Erreur de reconstruction
        reconstruction_error = np.mean(np.square(X_scaled - X_pred), axis=1)

        # Classification
        anomalies = (reconstruction_error > self.reconstruction_error_threshold).astype(int)

        return {
            'scores': reconstruction_error,
            'anomalies': anomalies,
            'threshold': self.reconstruction_error_threshold,
            'reconstruction': X_pred
        }

    def explain(self, X: np.ndarray, sample_idx: int = 0) -> Dict:
        """
        Explique une anomalie par l'erreur de reconstruction par feature
        """
        if not self.is_fitted:
            raise ValueError("Modèle non entraîné")

        X_scaled = self.scaler.transform(X)
        X_pred = self.model.predict(X_scaled)

        # Erreur par feature
        per_feature_error = np.square(X_scaled[sample_idx] - X_pred[sample_idx])

        if self.feature_names:
            feature_error = dict(zip(self.feature_names, per_feature_error))
        else:
            feature_error = {f'feature_{i}': v for i, v in enumerate(per_feature_error)}

        return {
            'sample_index': sample_idx,
            'total_error': np.sum(per_feature_error),
            'feature_errors': feature_error,
            'top_features': sorted(feature_error.items(), key=lambda x: x[1], reverse=True)[:10]
        }

    def save(self, filepath: str):
        """Sauvegarde le modèle"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'params': self.params,
            'threshold': self.reconstruction_error_threshold,
            'feature_names': getattr(self, 'feature_names', None)
        }, filepath)

    def load(self, filepath: str):
        """Charge le modèle"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.params = data['params']
        self.reconstruction_error_threshold = data['threshold']
        self.feature_names = data.get('feature_names')
        self.is_fitted = True
        return self
