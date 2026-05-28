"""
Feature Engineering Avancé pour la Détection de Fraude
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List

class FeatureEngineer:
  """
  Ingénierie des features pour la détection d'anomalies
  """
  def __init__(self):
    self.label_encoders = {}
    self.fitted = False

  def fit(self, df: pd.DataFrame):
    """
    Ajuste les encodeurs sur les données d'entraînement
    """
    # Encoder le type de transaction
    le_type = LabelEncoder()
    le_type.fit(df['type'])
    self.label_encoders['type'] = le_type

    self.fitted = True

  def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme un DataFrame en features pour le modèle
    """
    df_feat = df.copy()

    # 1. Features de balance
    df_feat['balance_diff_orig'] = df_feat['oldbalanceOrg'] - df_feat['newbalanceOrig']
    df_feat['balance_diff_dest'] = df_feat['oldbalanceDest'] - df_feat['newbalanceDest']
    df_feat['balance_ratio_orig'] = df_feat['newbalanceOrig'] / (df_feat['oldbalanceOrg'] + 1e-10)
    df_feat['balance_ratio_dest'] = df_feat['newbalanceDest'] / (df_feat['oldbalanceDest'] + 1e-10)

    # 2. Features de montant
    df_feat['amount_to_balance_ratio'] = df_feat['amount'] / (df_feat['oldbalanceOrg'] + 1e-10)
    df_feat['amount_to_oldbalance_ratio'] = df_feat['amount'] / (df_feat['oldbalanceOrg'] + 1e-10)
    df_feat['amount_log'] = np.log1p(df_feat['amount'])

    # 3. Features de type
    df_feat['is_cash_out'] = (df_feat['type'] == 'CASH_OUT').astype(int)
    df_feat['is_transfer'] = (df_feat['type'] == 'TRANSFER').astype(int)
    df_feat['is_payment'] = (df_feat['type'] == 'PAYMENT').astype(int)
    df_feat['is_cash_in'] = (df_feat['type'] == 'CASH_IN').astype(int)
    df_feat['is_debit'] = (df_feat['type'] == 'DEBIT').astype(int)

    # 4. Features de risque
    # 4. Features de risque
    df_feat['is_merchant'] = df_feat['nameDest'].str.startswith('M').astype(int)
    df_feat['is_new_account'] = (df_feat['oldbalanceOrg'] == 0).astype(int)
    df_feat['is_empty_after'] = (df_feat['newbalanceOrig'] == 0).astype(int)

    # AJOUTER CETTE LIGNE
    df_feat['is_full_before'] = (
      df_feat['amount'] >= df_feat['oldbalanceOrg'] * 0.95
    ).astype(int)

    # 5. Features temporelles
    df_feat['hour'] = df_feat['step'] % 24
    df_feat['day'] = df_feat['step'] // 24
    df_feat['is_weekend'] = ((df_feat['day'] % 7) >= 5).astype(int)
    df_feat['is_night'] = ((df_feat['hour'] >= 0) & (df_feat['hour'] <= 6)).astype(int)

    # 6. Features de comportement
    df_feat['balance_change_orig'] = df_feat['newbalanceOrig'] - df_feat['oldbalanceOrg']
    df_feat['balance_change_dest'] = df_feat['newbalanceDest'] - df_feat['oldbalanceDest']
    df_feat['is_balance_consistent_orig'] = (
      np.abs(df_feat['balance_change_orig'] + df_feat['amount']) < 1
    ).astype(int)
    df_feat['is_balance_consistent_dest'] = (
      np.abs(df_feat['balance_change_dest'] - df_feat['amount']) < 1
    ).astype(int)

    # 7. Patterns de fraude connus
    df_feat['transfer_then_cashout_pattern'] = (
      (df_feat['type'] == 'TRANSFER') &
      (df_feat['amount'] == df_feat['oldbalanceOrg']) &
      (df_feat['newbalanceOrig'] == 0)
    ).astype(int)

    df_feat['high_amount_low_balance'] = (
      (df_feat['amount'] > df_feat['oldbalanceOrg'] * 0.9) &
      (df_feat['oldbalanceOrg'] > 1000)
    ).astype(int)

    # 8. Features statistiques par client
    orig_counts = df_feat['nameOrig'].value_counts()
    df_feat['orig_transaction_count'] = df_feat['nameOrig'].map(orig_counts)

    dest_counts = df_feat['nameDest'].value_counts()
    df_feat['dest_transaction_count'] = df_feat['nameDest'].map(dest_counts)

    orig_mean = df_feat.groupby('nameOrig')['amount'].mean()
    df_feat['orig_mean_amount'] = df_feat['nameOrig'].map(orig_mean)

    orig_max = df_feat.groupby('nameOrig')['amount'].max()
    df_feat['orig_max_amount'] = df_feat['nameOrig'].map(orig_max)

    # 9. Ratios comportementaux
    df_feat['amount_vs_mean_ratio'] = df_feat['amount'] / (df_feat['orig_mean_amount'] + 1e-10)
    df_feat['amount_vs_max_ratio'] = df_feat['amount'] / (df_feat['orig_max_amount'] + 1e-10)

    # 10. Encodage
    if self.fitted and 'type' in self.label_encoders:
      df_feat['type_encoded'] = self.label_encoders['type'].transform(df_feat['type'])
    else:
      le = LabelEncoder()
      df_feat['type_encoded'] = le.fit_transform(df_feat['type'])

    return df_feat

  def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit puis transforme
    """
    self.fit(df)
    return self.transform(df)

  def get_feature_names(self) -> List[str]:
    """
    Retourne la liste des features créées
    """
    return [
      'step', 'amount', 'amount_log', 'oldbalanceOrg', 'newbalanceOrig',
      'oldbalanceDest', 'newbalanceDest', 'balance_diff_orig', 'balance_diff_dest',
      'balance_ratio_orig', 'balance_ratio_dest', 'amount_to_balance_ratio',
      'amount_to_oldbalance_ratio', 'is_cash_out', 'is_transfer', 'is_payment',
      'is_cash_in', 'is_debit', 'is_merchant', 'is_new_account', 'is_empty_after',
      'is_full_before', 'hour', 'day', 'is_weekend', 'is_night',
      'balance_change_orig', 'balance_change_dest', 'is_balance_consistent_orig',
      'is_balance_consistent_dest', 'transfer_then_cashout_pattern',
      'high_amount_low_balance', 'orig_transaction_count', 'dest_transaction_count',
      'orig_mean_amount', 'orig_max_amount', 'amount_vs_mean_ratio',
      'amount_vs_max_ratio', 'type_encoded'
    ]
