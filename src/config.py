"""
Configuration du projet de détection de fraude
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Chemins
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Dataset
DATASET_PATH = os.path.join(RAW_DATA_DIR, 'PS_20174392719_1491204439457_log.csv')

# Features
FEATURE_COLS = [
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

# Modèles
MODEL_PARAMS = {
  'isolation_forest': {
    'n_estimators': 200,
    'contamination': 0.002,
    'random_state': 42,
    'n_jobs': -1
  },
  'autoencoder': {
    'hidden_layers': [32, 16, 32],
    'activation': 'relu',
    'max_iter': 200,
    'early_stopping': True
  }
}

# Seuils
ANOMALY_THRESHOLD = 0.99 # Percentile pour classification

# API
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))

# Snowflake
SNOWFLAKE_CONFIG = {
  'user': os.getenv('SNOWFLAKE_USER'),
  'password': os.getenv('SNOWFLAKE_PASSWORD'),
  'account': os.getenv('SNOWFLAKE_ACCOUNT'),
  'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
  'database': os.getenv('SNOWFLAKE_DATABASE', 'FRAUD_DETECTION'),
  'schema': os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC')
}

# Active Learning
ACTIVE_LEARNING = {
  'n_queries_per_cycle': 20,
  'n_cycles': 5,
  'query_strategy': 'uncertainty' # 'uncertainty', 'diversity', 'random'
}
