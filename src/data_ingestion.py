"""
Pipeline d'ingestion de transactions (simulation haute fréquence)
"""
import pandas as pd
import numpy as np
from typing import Iterator, Dict, List
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionIngestionPipeline:
  """
  Pipeline d'ingestion simulant un flux haute fréquence de transactions
  """
  def __init__(self, batch_size: int = 1000, frequency_hz: float = 10.0):
    self.batch_size = batch_size
    self.frequency_hz = frequency_hz
    self.processed_count = 0
    self.anomaly_count = 0

  def stream_from_csv(self, filepath: str) -> Iterator[pd.DataFrame]:
    """
    Simule un flux de données depuis un fichier CSV
    """
    logger.info(f"Démarrage du streaming depuis {filepath}")

    # Lecture par chunks pour simuler le streaming
    for chunk in pd.read_csv(filepath, chunksize=self.batch_size):
      yield chunk

      # Simuler la fréquence d'arrivée
      time.sleep(1.0 / self.frequency_hz)

  def stream_from_kafka_simulation(self, n_transactions: int = 10000) -> Iterator[Dict]:
    """
    Simule un flux Kafka de transactions
    """
    for i in range(n_transactions):
      transaction = {
        'transaction_id': f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}",
        'timestamp': datetime.now().isoformat(),
        'step': np.random.randint(1, 745),
        'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT']),
        'amount': np.random.exponential(150000),
        'nameOrig': f"C{np.random.randint(1000000000, 9999999999)}",
        'oldbalanceOrg': np.random.exponential(50000),
        'nameDest': f"C{np.random.randint(1000000000, 9999999999)}",
        'oldbalanceDest': np.random.exponential(30000)
      }

      yield transaction

      if (i + 1) % self.batch_size == 0:
        logger.info(f"Transactions traitées: {i + 1}")

  def preprocess_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
    """
    Prétraitement d'un batch de transactions
    """
    # Nettoyage
    batch = batch.dropna()

    # Conversion des types
    batch['amount'] = pd.to_numeric(batch['amount'], errors='coerce')
    batch['step'] = pd.to_numeric(batch['step'], errors='coerce')

    # Filtrage des valeurs aberrantes évidentes
    batch = batch[batch['amount'] >= 0]
    batch = batch[batch['amount'] <= 1e8]

    self.processed_count += len(batch)
    return batch

  def save_to_snowflake(self, batch: pd.DataFrame, connection):
    """
    Sauvegarde dans Snowflake
    """
    from snowflake.connector.pandas_tools import write_pandas

    success, num_chunks, num_rows, _ = write_pandas(
      conn=connection,
      df=batch,
      table_name='TRANSACTIONS_STREAM',
      database='FRAUD_DETECTION',
      schema='PUBLIC'
    )

    logger.info(f"Sauvegardé {num_rows} lignes dans Snowflake")
    return success

class DataValidator:
  """
  Validation des données entrantes
  """
  def __init__(self):
    self.validation_rules = {
      'amount': {'min': 0, 'max': 1e8},
      'step': {'min': 1, 'max': 1000},
      'type': {'allowed': ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT']}
    }

  def validate(self, transaction: Dict) -> Dict:
    """
    Valide une transaction selon les règles
    """
    errors = []

    # Validation du montant
    if 'amount' in transaction:
      amount = transaction['amount']
      if amount < self.validation_rules['amount']['min']:
        errors.append(f"Amount {amount} below minimum")
      if amount > self.validation_rules['amount']['max']:
        errors.append(f"Amount {amount} above maximum")

    # Validation du type
    if 'type' in transaction:
      if transaction['type'] not in self.validation_rules['type']['allowed']:
        errors.append(f"Invalid type: {transaction['type']}")

    return {
      'valid': len(errors) == 0,
      'errors': errors,
      'transaction': transaction
    }
