"""
Ingestion temps réel simulant un flux de transactions
"""
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeTransactionProducer:
  """
  Producteur de transactions temps réel
  """
  def __init__(self, kafka_bootstrap='localhost:9092', topic='transactions'):
    self.topic = topic
    try:
      self.producer = KafkaProducer(
        bootstrap_servers=kafka_bootstrap,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
      )
      self.use_kafka = True
    except:
      logger.warning("Kafka non disponible, mode simulation")
      self.use_kafka = False
      self.buffer = []

  def generate_transaction(self):
    """Génère une transaction aléatoire"""
    types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT']
    type_weights = [0.35, 0.10, 0.30, 0.22, 0.03]

    txn_type = np.random.choice(types, p=type_weights)

    transaction = {
      'timestamp': datetime.now().isoformat(),
      'step': np.random.randint(1, 745),
      'type': txn_type,
      'amount': np.random.exponential(150000),
      'nameOrig': f"C{np.random.randint(1000000000, 9999999999, dtype=np.int64)}",
      'oldbalanceOrg': np.random.exponential(50000),
      'nameDest': f"C{np.random.randint(1000000000, 9999999999, dtype=np.int64)}",
      'oldbalanceDest': np.random.exponential(30000)
    }

    # Calculer les balances après
    if txn_type in ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT']:
      transaction['newbalanceOrig'] = max(0, transaction['oldbalanceOrg'] - transaction['amount'])
    else:
      transaction['newbalanceOrig'] = transaction['oldbalanceOrg'] + transaction['amount']

    if txn_type in ['CASH_IN', 'TRANSFER']:
      transaction['newbalanceDest'] = transaction['oldbalanceDest'] + transaction['amount']
    else:
      transaction['newbalanceDest'] = max(0, transaction['oldbalanceDest'] - transaction['amount'])

    return transaction

  def send_transaction(self, transaction):
    """Envoie une transaction"""
    if self.use_kafka:
      self.producer.send(self.topic, transaction)
    else:
      self.buffer.append(transaction)
      if len(self.buffer) >= 1000:
        logger.info(f"Buffer: {len(self.buffer)} transactions")

  def run(self, duration_seconds=60, frequency_hz=10):
    """Génère un flux de transactions"""
    logger.info(f"Démarrage du flux: {frequency_hz} Hz pendant {duration_seconds}s")

    start_time = time.time()
    count = 0

    while time.time() - start_time < duration_seconds:
      transaction = self.generate_transaction()
      self.send_transaction(transaction)
      count += 1

      if count % 1000 == 0:
        logger.info(f"Transactions générées: {count}")

      time.sleep(1.0 / frequency_hz)

    logger.info(f"Flux terminé: {count} transactions générées")

class RealtimeTransactionConsumer:
  """
  Consommateur de transactions temps réel avec détection
  """
  def __init__(self, kafka_bootstrap='localhost:9092', topic='transactions'):
    self.topic = topic
    try:
      self.consumer = KafkaConsumer(
        topic,
        bootstrap_servers=kafka_bootstrap,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset='latest'
      )
      self.use_kafka = True
    except:
      logger.warning("Kafka non disponible, mode simulation")
      self.use_kafka = False

  def process_with_api(self, transaction, api_url='http://localhost:8000'):
    """Envoie la transaction à l'API pour détection"""
    import requests

    try:
      response = requests.post(
        f"{api_url}/predict",
        json=transaction,
        timeout=1
      )
      return response.json()
    except:
      return None

  def run(self, max_transactions=1000):
    """Consomme et traite les transactions"""
    logger.info("Démarrage du consommateur...")

    anomalies_detected = 0

    if self.use_kafka:
      for message in self.consumer:
        transaction = message.value
        result = self.process_with_api(transaction)

        if result and result.get('is_anomaly'):
          anomalies_detected += 1
          logger.warning(f" Anomalie détectée: {result}")

        if anomalies_detected >= max_transactions:
          break
    else:
      logger.info("Mode simulation - génération de transactions de test")
      producer = RealtimeTransactionProducer()

      for _ in range(max_transactions):
        transaction = producer.generate_transaction()
        result = self.process_with_api(transaction)

        if result and result.get('is_anomaly'):
          anomalies_detected += 1

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--produce', action='store_true', help='Mode producteur')
  parser.add_argument('--consume', action='store_true', help='Mode consommateur')
  parser.add_argument('--duration', type=int, default=60, help='Durée en secondes')
  parser.add_argument('--frequency', type=float, default=10.0, help='Fréquence Hz')

  args = parser.parse_args()

  if args.produce:
    producer = RealtimeTransactionProducer()
    producer.run(duration_seconds=args.duration, frequency_hz=args.frequency)
  elif args.consume:
    consumer = RealtimeTransactionConsumer()
    consumer.run()
  else:
    print("Usage: python realtime_ingestion.py --produce|--consume")
