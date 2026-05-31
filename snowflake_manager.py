"""
Configuration et utilitaires Snowflake (version corrigée)
"""
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import pandas as pd
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv() # IMPORTANT

class SnowflakeManager:
  """
  Gestionnaire de connexion et opérations Snowflake
  """

  def __init__(self):
    self.connection = None

    self.config = {
      'user': os.getenv('SNOWFLAKE_USER'),
      'password': os.getenv('SNOWFLAKE_PASSWORD'),
      'account': os.getenv('SNOWFLAKE_ACCOUNT'),
      'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
      'database': os.getenv('SNOWFLAKE_DATABASE', 'FRAUD_DETECTION'),
      'schema': os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC')
    }

    print("=== CONFIG CHECK ===")
    for k, v in self.config.items():
      print(f"{k}: {'OK' if v else 'MISSING'}")
    print("====================")

  def connect(self):
    """Établit la connexion à Snowflake"""
    try:
      if not self.config['user'] or not self.config['password'] or not self.config['account']:
        raise ValueError(" Variables Snowflake manquantes dans .env")

      self.connection = snowflake.connector.connect(**self.config)

      cur = self.connection.cursor()

      # IMPORTANT : définir le contexte
      cur.execute(f"USE WAREHOUSE {self.config['warehouse']}")
      cur.execute(f"USE DATABASE {self.config['database']}")
      cur.execute(f"USE SCHEMA {self.config['schema']}")

      print(" Connecté à Snowflake + contexte initialisé")
      return self.connection

    except Exception as e:
      print(f" Erreur de connexion: {e}")
      return None

  def create_tables(self):
    """Crée les tables nécessaires"""
    if not self.connection:
      self.connect()

    if not self.connection:
      raise Exception(" Connexion Snowflake impossible")

    queries = [
      """
      CREATE TABLE IF NOT EXISTS TRANSACTIONS (
        TRANSACTION_ID VARCHAR(50) PRIMARY KEY,
        STEP INTEGER,
        TYPE VARCHAR(20),
        AMOUNT FLOAT,
        NAME_ORIG VARCHAR(20),
        OLDBALANCE_ORG FLOAT,
        NEWBALANCE_ORIG FLOAT,
        NAME_DEST VARCHAR(20),
        OLDBALANCE_DEST FLOAT,
        NEWBALANCE_DEST FLOAT,
        IS_FRAUD INTEGER,
        IS_FLAGGED_FRAUD INTEGER,
        CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
      """,
      """
      CREATE TABLE IF NOT EXISTS ANOMALIES (
        ANOMALY_ID VARCHAR(50) PRIMARY KEY,
        TRANSACTION_ID VARCHAR(50),
        ANOMALY_SCORE FLOAT,
        MODEL_USED VARCHAR(50),
        DETECTED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        IS_REVIEWED BOOLEAN DEFAULT FALSE,
        IS_CONFIRMED_FRAUD BOOLEAN,
        INVESTIGATOR_NOTES VARCHAR(500)
      )
      """,
      """
      CREATE TABLE IF NOT EXISTS MODEL_PERFORMANCE (
        MODEL_NAME VARCHAR(50),
        VERSION VARCHAR(20),
        AUC_ROC FLOAT,
        PRECISION FLOAT,
        RECALL FLOAT,
        F1_SCORE FLOAT,
        TRAINED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
      """
    ]

    cur = self.connection.cursor()

    for query in queries:
      cur.execute(query)

    print(" Tables créées avec succès")

  def insert_transactions(self, df: pd.DataFrame):
    """Insère des transactions"""
    if not self.connection:
      self.connect()

    success, num_chunks, num_rows, _ = write_pandas(
      conn=self.connection,
      df=df,
      table_name='TRANSACTIONS',
      database=self.config['database'],
      schema=self.config['schema']
    )

    print(f" {num_rows} transactions insérées")
    return success

  def get_anomalies_for_review(self, limit: int = 100) -> pd.DataFrame:
    """Récupère les anomalies à réviser"""
    if not self.connection:
      self.connect()

    query = f"""
    SELECT * FROM ANOMALIES
    WHERE IS_REVIEWED = FALSE
    ORDER BY ANOMALY_SCORE DESC
    LIMIT {limit}
    """

    return pd.read_sql(query, self.connection)

  def update_annotation(self, anomaly_id: str, is_fraud: bool, notes: Optional[str] = None):
    """Met à jour l'annotation d'une anomalie"""
    if not self.connection:
      self.connect()

    query = """
    UPDATE ANOMALIES
    SET IS_REVIEWED = TRUE,
      IS_CONFIRMED_FRAUD = %s,
      INVESTIGATOR_NOTES = %s
    WHERE ANOMALY_ID = %s
    """

    self.connection.cursor().execute(query, (is_fraud, notes, anomaly_id))
    print(f" Anomalie {anomaly_id} mise à jour")

  def close(self):
    """Ferme la connexion"""
    if self.connection:
      self.connection.close()
      print(" Connexion Snowflake fermée")


# ===================== CLI =====================

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--setup', action='store_true', help='Créer les tables')
  parser.add_argument('--test', action='store_true', help='Tester la connexion')

  args = parser.parse_args()

  sf = SnowflakeManager()

  if args.setup:
    sf.connect()
    sf.create_tables()
    sf.close()

  elif args.test:
    sf.connect()

    if sf.connection:
      result = pd.read_sql("SELECT CURRENT_VERSION()", sf.connection)
      print(f"Version Snowflake: {result.iloc[0, 0]}")
      sf.close()
    else:
      print(" Connexion échouée")

  else:
    print("Usage: python snowflake_manager.py --setup | --test")