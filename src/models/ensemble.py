"""
Ensemble de modèles pour la détection d'anomalies
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List
import joblib

class EnsembleModel:
    """
    Combinaison de plusieurs modèles de détection d'anomalies
    """
    def __init__(self, models: Dict = None, weights: Dict = None):
        """
        Args:
            models: Dict {nom: modèle}
            weights: Dict {nom: poids}
        """
        self.models = models or {}
        self.weights = weights or {}
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.threshold = None

    def add_model(self, name: str, model, weight: float = 1.0):
        """Ajoute un modèle à l'ensemble"""
        self.models[name] = model
        self.weights[name] = weight

    def fit(self, X: np.ndarray):
        """
        Entraîne tous les modèles
        """
        for name, model in self.models.items():
            print(f"Entraînement {name}...")
            model.fit(X)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> Dict:
        """
        Prédit avec l'ensemble
        """
        if not self.is_fitted:
            raise ValueError("Ensemble non entraîné")

        all_scores = []
        model_results = {}

        for name, model in self.models.items():
            result = model.predict(X)
            scores = result['scores']

            # Normaliser les scores.
            # Si on pr?dit une seule transaction, min == max et la normalisation
            # min/max donne toujours 0. Dans ce cas, on utilise la d?cision du
            # mod?le de base: 1 = anomalie, 0 = normal.
            score_range = scores.max() - scores.min()
            if len(scores) > 1 and score_range > 1e-10:
                scores_norm = (scores - scores.min()) / (score_range + 1e-10)
            else:
                scores_norm = np.asarray(result.get('anomalies', np.zeros_like(scores)), dtype=float)

            all_scores.append(scores_norm * self.weights.get(name, 1.0))
            model_results[name] = result

        # Score ensemble
        ensemble_scores = np.mean(all_scores, axis=0)

        # Seuil
        if self.threshold is None:
            self.threshold = np.percentile(ensemble_scores, 99.5)

        anomalies = (ensemble_scores > self.threshold).astype(int)

        return {
            'scores': ensemble_scores,
            'anomalies': anomalies,
            'threshold': self.threshold,
            'model_results': model_results,
            'weights': self.weights
        }

    def explain(self, X: np.ndarray, sample_idx: int = 0) -> Dict:
        """
        Agrège les explications de tous les modèles
        """
        explanations = {}

        for name, model in self.models.items():
            try:
                explanations[name] = model.explain(X, sample_idx)
            except:
                explanations[name] = {'error': 'Explication non disponible'}

        return {
            'sample_index': sample_idx,
            'model_explanations': explanations,
            'ensemble_weights': self.weights
        }

    def save(self, filepath: str):
        """Sauvegarde l'ensemble"""
        joblib.dump({
            'models': self.models,
            'weights': self.weights,
            'threshold': self.threshold
        }, filepath)

    def load(self, filepath: str):
        """Charge l'ensemble"""
        data = joblib.load(filepath)
        self.models = data['models']
        self.weights = data['weights']
        self.threshold = data['threshold']
        self.is_fitted = True
        return self
