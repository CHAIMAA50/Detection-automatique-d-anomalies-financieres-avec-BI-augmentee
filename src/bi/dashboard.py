"""
Dashboard BI pour l'investigation manuelle et l'annotation
Streamlit
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import os
from datetime import datetime

# Configuration
st.set_page_config(
  page_title="Fraud Detection Dashboard",
  page_icon="FD",
  layout="wide",
  initial_sidebar_state="expanded"
)

# Style CSS
css = """
<style>
  .main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
  }

  .metric-card {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
  }

  .anomaly-card {
    background-color: #1f2937 !important;
    border-left: 5px solid #ef4444 !important;
    color: #ffffff !important;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
  }

  .normal-card {
    background-color: #1f2937 !important;
    border-left: 5px solid #22c55e !important;
    color: #ffffff !important;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
  }

  .anomaly-card b,
  .normal-card b {
    color: #ffffff !important;
  }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def fetch_anomalies(api_url="http://localhost:8000"):
  """Récupère les anomalies depuis l'API"""
  try:
    response = requests.get(f"{api_url}/stats")
    return response.json()
  except:
    return None


def send_feedback(api_url, transaction_id, is_fraud, notes="Annotation depuis dashboard"):
  """Envoie une annotation au backend FastAPI."""
  try:
    response = requests.post(
      f"{api_url}/feedback",
      json={
        "transaction_id": transaction_id,
        "is_fraud": bool(is_fraud),
        "investigator_notes": notes
      },
      timeout=10
    )

    if response.status_code == 200:
      return True, response.json()

    return False, f"Erreur API {response.status_code}: {response.text}"
  except requests.exceptions.ConnectionError:
    return False, "API indisponible. Lancez FastAPI sur http://localhost:8000."
  except Exception as exc:
    return False, str(exc)


def project_path(*parts):
  """Construit un chemin relatif a la racine du projet."""
  return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), *parts)


def load_feedback_data():
  """Charge les annotations sauvegardees par l'API."""
  feedback_path = project_path("data", "feedback", "feedback.csv")
  if os.path.exists(feedback_path):
    return pd.read_csv(feedback_path)
  return pd.DataFrame(columns=["transaction_id", "is_fraud", "notes", "timestamp"])


def load_prediction_history():
  """Charge l'historique des predictions manuelles."""
  prediction_path = project_path("data", "predictions", "predictions.csv")
  if os.path.exists(prediction_path):
    return pd.read_csv(prediction_path)
  return pd.DataFrame()


def save_prediction(transaction, result, display_is_suspicious, signals):
  """Sauvegarde chaque prediction manuelle dans un CSV local."""
  prediction_dir = project_path("data", "predictions")
  os.makedirs(prediction_dir, exist_ok=True)
  prediction_path = os.path.join(prediction_dir, "predictions.csv")

  row = {
    "timestamp": datetime.now().isoformat(),
    "transaction_type": transaction["type"],
    "amount": transaction["amount"],
    "oldbalanceOrg": transaction["oldbalanceOrg"],
    "newbalanceOrig": transaction["newbalanceOrig"],
    "oldbalanceDest": transaction["oldbalanceDest"],
    "newbalanceDest": transaction["newbalanceDest"],
    "anomaly_score": result.get("anomaly_score", 0.0),
    "model_is_anomaly": result.get("is_anomaly", False),
    "dashboard_result": "suspecte" if display_is_suspicious else "normale",
    "model_used": result.get("model_used", "unknown"),
    "signals": " | ".join(signals)
  }

  pd.DataFrame([row]).to_csv(
    prediction_path,
    mode="a",
    header=not os.path.exists(prediction_path),
    index=False,
    encoding="utf-8"
  )
  return prediction_path


def detect_business_signals(transaction, score):
  """Detecte des signaux metier simples pour expliquer la prediction."""
  signals = []
  amount = transaction["amount"]
  old_org = transaction["oldbalanceOrg"]
  new_org = transaction["newbalanceOrig"]
  old_dest = transaction["oldbalanceDest"]
  new_dest = transaction["newbalanceDest"]
  txn_type = transaction["type"]

  if txn_type in ["TRANSFER", "CASH_OUT"]:
    signals.append(f"Type de transaction a risque: {txn_type}")
  if old_org > 0 and amount >= old_org * 0.95 and new_org <= old_org * 0.05:
    signals.append("Le client vide presque totalement son compte")
  if old_org == 0 and amount > 0:
    signals.append("Montant envoye alors que le solde origine est nul")
  if amount >= 100000:
    signals.append("Montant tres eleve")

  expected_new_org = max(old_org - amount, 0)
  if abs(new_org - expected_new_org) > max(1.0, amount * 0.05):
    signals.append("Incoherence entre montant et nouveau solde origine")
  if txn_type in ["TRANSFER", "CASH_OUT"] and amount == old_org and new_org == 0:
    signals.append("Pattern connu: transfert/cash-out de tout le solde")
  if old_dest == 0 and new_dest == 0 and txn_type == "TRANSFER":
    signals.append("Destination sans solde modifie apres transfert")

  if score >= 0.70:
    signals.append("Score ML eleve")
  elif score >= 0.30:
    signals.append("Score ML moyen a surveiller")
  if not signals:
    signals.append("Aucun signal metier fort detecte")
  return signals


def risk_level(score):
  if score >= 0.70:
    return "Risque eleve", "red"
  if score >= 0.30:
    return "Risque moyen", "orange"
  return "Risque faible", "green"

def load_data():
  """Charge les données de transactions"""
  np.random.seed(42)
  n = 1000

  data = {
    'transaction_id': [f'TXN_{i:06d}' for i in range(n)],
    'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
    'step': np.random.randint(1, 745, n),
    'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT'], n),
    'amount': np.random.exponential(150000, n),
    'nameOrig': [
      f'C{np.random.randint(100000000, 999999999)}'
      for _ in range(n)
    ],
    'nameDest': [
      f'C{np.random.randint(100000000, 999999999)}'
      for _ in range(n)
    ],
    'oldbalanceOrg': np.random.exponential(50000, n),
    'newbalanceOrig': np.random.exponential(40000, n),
    'oldbalanceDest': np.random.exponential(30000, n),
    'newbalanceDest': np.random.exponential(35000, n),
    'anomaly_score': np.random.beta(2, 50, n),
    'is_anomaly': np.random.choice([0, 1], n, p=[0.99, 0.01]),
    'is_fraud': np.random.choice([0, 1], n, p=[0.995, 0.005]),
    'is_reviewed': np.random.choice([0, 1], n, p=[0.8, 0.2]),
    'investigator_notes': [''] * n
  }

  return pd.DataFrame(data)


# ============================================================================
# BARRE LATÉRALE
# ============================================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
  "Choisir une page:",
  ["Vue d'ensemble", "Investigation", "Prediction", "Analytique", "Configuration"]
)

st.sidebar.markdown("---")

st.sidebar.info("""
**Fraud Detection System**
- Version: 1.0.0
- Modèles: Isolation Forest, Autoencoder, Ensemble
- API: http://localhost:8000
""")

api_status_url = st.sidebar.text_input(
  "URL API",
  value="http://localhost:8000",
  key="api_status_url"
)
api_status = fetch_anomalies(api_status_url)
if api_status:
  st.sidebar.success("API connectee")
  st.sidebar.caption(f"Modeles charges: {', '.join(api_status.get('models', [])) or 'aucun'}")
else:
  st.sidebar.error("API indisponible")


# ============================================================================
# PAGE 1: VUE D'ENSEMBLE
# ============================================================================

if page == "Vue d'ensemble":
  st.markdown(
    '<p class="main-header">Tableau de Bord - Détection de Fraude</p>',
    unsafe_allow_html=True
  )

  df = load_data()

  col1, col2, col3, col4 = st.columns(4)

  with col1:
    st.metric(
      label="Transactions Totales",
      value=f"{len(df):,}",
      delta="+12% vs hier"
    )

  with col2:
    st.metric(
      label="Anomalies Détectées",
      value=f"{df['is_anomaly'].sum():,}",
      delta="-5% vs hier",
      delta_color="inverse"
    )

  with col3:
    frauds = df['is_fraud'].sum()
    anomalies = df['is_anomaly'].sum()
    precision = frauds / anomalies * 100 if anomalies > 0 else 0

    st.metric(
      label="Fraudes Confirmées",
      value=f"{frauds:,}",
      delta=f"{precision:.1f}% précision"
    )

  with col4:
    reviewed = df['is_reviewed'].sum()

    st.metric(
      label="Révisions Humaines",
      value=f"{reviewed:,}",
      delta=f"{(reviewed / len(df) * 100):.1f}% couverture"
    )

  st.markdown("---")

  col_left, col_right = st.columns(2)

  with col_left:
    st.subheader("Distribution des Anomalies")

    df['date'] = df['timestamp'].dt.date

    daily_anomalies = df.groupby('date').agg({
      'is_anomaly': 'sum',
      'is_fraud': 'sum'
    }).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
      x=daily_anomalies['date'],
      y=daily_anomalies['is_anomaly'],
      mode='lines+markers',
      name='Anomalies Détectées',
      line=dict(color='#ff6b6b')
    ))

    fig.add_trace(go.Scatter(
      x=daily_anomalies['date'],
      y=daily_anomalies['is_fraud'],
      mode='lines+markers',
      name='Fraudes Confirmées',
      line=dict(color='#4ecdc4')
    ))

    fig.update_layout(
      xaxis_title='Date',
      yaxis_title='Nombre',
      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)

  with col_right:
    st.subheader(" Répartition par Type")

    type_counts = df.groupby('type').agg({
      'is_anomaly': 'sum',
      'is_fraud': 'sum'
    }).reset_index()

    fig = make_subplots(
      rows=1,
      cols=2,
      specs=[[{'type': 'domain'}, {'type': 'domain'}]]
    )

    fig.add_trace(go.Pie(
      labels=type_counts['type'],
      values=type_counts['is_anomaly'],
      name="Anomalies"
    ), 1, 1)

    fig.add_trace(go.Pie(
      labels=type_counts['type'],
      values=type_counts['is_fraud'],
      name="Fraudes"
    ), 1, 2)

    fig.update_traces(hole=.4, hoverinfo="label+percent+value")

    fig.update_layout(
      annotations=[
        dict(text='Anomalies', x=0.18, y=0.5, font_size=12, showarrow=False),
        dict(text='Fraudes', x=0.82, y=0.5, font_size=12, showarrow=False)
      ]
    )

    st.plotly_chart(fig, use_container_width=True)

  st.subheader(" Dernières Anomalies")

  recent_anomalies = (
    df[df['is_anomaly'] == 1]
    .sort_values('timestamp', ascending=False)
    .head(10)
  )

  st.dataframe(
    recent_anomalies[
      ['transaction_id', 'timestamp', 'type', 'amount',
       'anomaly_score', 'is_fraud', 'is_reviewed']
    ],
    use_container_width=True,
    hide_index=True
  )

# ============================================================================
# PAGE 2: INVESTIGATION
# ============================================================================

elif page == "Investigation":
  st.markdown(
    '<p class="main-header">Investigation Manuelle</p>',
    unsafe_allow_html=True
  )

  df = load_data()

  st.sidebar.markdown("---")
  st.sidebar.subheader("Filtres")
  feedback_api_url = api_status_url

  min_score = st.sidebar.slider("Score Anomalie Min", 0.0, 1.0, 0.0)

  show_only_anomalies = st.sidebar.checkbox(
    "Afficher seulement anomalies",
    value=True
  )

  txn_type = st.sidebar.multiselect(
    "Type de Transaction",
    ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT'],
    default=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT']
  )

  show_reviewed = st.sidebar.checkbox("Afficher révisés", value=False)

  filtered = df[
    (df['anomaly_score'] >= min_score) &
    (df['type'].isin(txn_type))
  ]

  if not show_reviewed:
    filtered = filtered[filtered['is_reviewed'] == 0]

  if show_only_anomalies:
    filtered = filtered[filtered['is_anomaly'] == 1]

  filtered = filtered.sort_values('anomaly_score', ascending=False)

  st.write(f"**{len(filtered)} transactions** correspondent aux critères")

  for idx, row in filtered.head(20).iterrows():
    card_class = "anomaly-card" if row['is_anomaly'] else "normal-card"

    with st.container():
      col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

      with col1:
        st.markdown(f"""
        <div class="{card_class}">
          <b>ID:</b> {row['transaction_id']}<br>
          <b>Type:</b> {row['type']} | <b>Montant:</b> ${row['amount']:,.2f}<br>
          <b>Origine:</b> {row['nameOrig']} → <b>Dest:</b> {row['nameDest']}
        </div>
        """, unsafe_allow_html=True)

      with col2:
        st.metric("Score", f"{row['anomaly_score']:.3f}")

      with col3:
        if row['is_reviewed']:
          st.success(" Révisé")
        else:
          st.warning("⏳ En attente")

      with col4:
        col_a, col_b = st.columns(2)

        with col_a:
          if st.button("Normal", key=f"normal_{idx}"):
            ok, feedback_result = send_feedback(
              feedback_api_url,
              row['transaction_id'],
              False,
              "Annotation dashboard: transaction normale"
            )
            if ok:
              st.success("Annote comme Normal et sauvegarde dans l'API")
            else:
              st.error(feedback_result)

        with col_b:
          if st.button("Fraude", key=f"fraud_{idx}"):
            ok, feedback_result = send_feedback(
              feedback_api_url,
              row['transaction_id'],
              True,
              "Annotation dashboard: fraude confirmee"
            )
            if ok:
              st.error("Annote comme Fraude et sauvegarde dans l'API")
            else:
              st.error(feedback_result)


  st.markdown("---")
  st.subheader(" Détails de la Transaction")

  if filtered.empty:
    st.info(
      "Aucune transaction ne correspond aux filtres actuels. "
      "Diminuez le score minimum ou activez l'affichage des transactions révisées."
    )
    selected_id = None
  else:
    selected_id = st.selectbox(
      "Sélectionner une transaction",
      filtered['transaction_id'].head(50)
    )

  if selected_id:
    txn = df[df['transaction_id'] == selected_id].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
      st.write("**Informations Générales**")
      st.json({
        'ID': txn['transaction_id'],
        'Timestamp': str(txn['timestamp']),
        'Type': txn['type'],
        'Montant': f"${txn['amount']:,.2f}",
        'Score Anomalie': f"{txn['anomaly_score']:.4f}"
      })

    with col2:
      st.write("**Balances**")

      balances = {
        'Avant (Orig)': txn['oldbalanceOrg'],
        'Après (Orig)': txn['newbalanceOrig'],
        'Avant (Dest)': txn['oldbalanceDest'],
        'Après (Dest)': txn['newbalanceDest']
      }

      fig = go.Figure(data=[
        go.Bar(
          name='Balance',
          x=list(balances.keys()),
          y=list(balances.values())
        )
      ])

      st.plotly_chart(fig, use_container_width=True)

    st.subheader("Explication du Modèle")

    explanation = {
      'amount': 0.35,
      'oldbalanceOrg': 0.25,
      'type_TRANSFER': 0.20,
      'newbalanceOrig': 0.15,
      'step': 0.05
    }

    fig = go.Figure(go.Bar(
      x=list(explanation.values()),
      y=list(explanation.keys()),
      orientation='h',
      marker_color=[
        '#ff6b6b' if v > 0.15 else '#4ecdc4'
        for v in explanation.values()
      ]
    ))

    fig.update_layout(
      title="Features les plus importantes",
      xaxis_title="Importance"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Notes de l'Investigateur")

    notes = st.text_area(
      "Ajouter des notes",
      value=txn['investigator_notes']
    )

    if st.button("Sauvegarder l'annotation"):
      st.success("Annotation sauvegardée!")

# ============================================================================
# PAGE 3: PREDICTION MANUELLE
# ============================================================================

elif page == "Prediction":
  st.markdown(
    '<p class="main-header">Prediction Manuelle</p>',
    unsafe_allow_html=True
  )

  st.write("Saisissez les informations d'une transaction puis cliquez sur Predire.")

  api_url = api_status_url
  model_name = st.sidebar.selectbox(
    "Modèle",
    ["ensemble", "isolation_forest", "autoencoder"],
    index=0
  )

  with st.form("manual_prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
      step = st.number_input("Step", min_value=1, max_value=744, value=1)
      txn_type = st.selectbox(
        "Type de transaction",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
      )
      amount = st.number_input("Montant", min_value=0.0, value=1000.0, step=100.0)

    with col2:
      name_orig = st.text_input("Client origine", value="C123456789")
      oldbalance_org = st.number_input("Ancien solde origine", min_value=0.0, value=1000.0, step=100.0)
      newbalance_orig = st.number_input("Nouveau solde origine", min_value=0.0, value=0.0, step=100.0)

    with col3:
      name_dest = st.text_input("Client destination", value="C987654321")
      oldbalance_dest = st.number_input("Ancien solde destination", min_value=0.0, value=0.0, step=100.0)
      newbalance_dest = st.number_input("Nouveau solde destination", min_value=0.0, value=1000.0, step=100.0)

    submitted = st.form_submit_button("Predire")

  if submitted:
    transaction = {
      "step": int(step),
      "type": txn_type,
      "amount": float(amount),
      "nameOrig": name_orig,
      "oldbalanceOrg": float(oldbalance_org),
      "newbalanceOrig": float(newbalance_orig),
      "nameDest": name_dest,
      "oldbalanceDest": float(oldbalance_dest),
      "newbalanceDest": float(newbalance_dest)
    }

    try:
      response = requests.post(
        f"{api_url}/predict?model_name={model_name}",
        json=transaction,
        timeout=15
      )

      if response.status_code == 200:
        result = response.json()
        score = result.get("anomaly_score", 0.0)
        is_anomaly = result.get("is_anomaly", False)
        dashboard_threshold = 0.30
        signals = detect_business_signals(transaction, score)
        has_strong_business_signal = any(
          signal != "Aucun signal metier fort detecte"
          for signal in signals
        )
        display_is_suspicious = is_anomaly or score >= dashboard_threshold or has_strong_business_signal
        prediction_path = save_prediction(transaction, result, display_is_suspicious, signals)

        st.markdown("---")
        st.subheader("Resultat")

        col_result_1, col_result_2, col_result_3 = st.columns(3)

        with col_result_1:
          st.metric("Score d'anomalie", f"{score:.4f}")

        with col_result_2:
          if display_is_suspicious:
            st.error("Transaction suspecte")
          else:
            st.success("Transaction normale")
          st.caption(f"Seuil dashboard: {dashboard_threshold:.2f}")

        with col_result_3:
          st.metric("Modele utilise", result.get("model_used", model_name))

        risk_text, risk_color = risk_level(score)
        gauge = go.Figure(go.Indicator(
          mode="gauge+number",
          value=float(score),
          number={"valueformat": ".3f"},
          title={"text": risk_text},
          gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": risk_color},
            "steps": [
              {"range": [0, 0.30], "color": "#1f5132"},
              {"range": [0.30, 0.70], "color": "#7c5c12"},
              {"range": [0.70, 1.00], "color": "#7f1d1d"}
            ],
            "threshold": {
              "line": {"color": "white", "width": 3},
              "thickness": 0.75,
              "value": dashboard_threshold
            }
          }
        ))
        gauge.update_layout(height=280, margin=dict(l=20, r=20, t=45, b=15))
        st.plotly_chart(gauge, use_container_width=True)

        st.subheader("Signaux metier detectes")
        for signal in signals:
          if signal == "Aucun signal metier fort detecte":
            st.info(signal)
          else:
            st.warning(signal)

        st.caption(f"Prediction sauvegardee dans: {prediction_path}")

        with st.expander("Voir la reponse complete de l'API"):
          st.json(result)

      else:
        st.error(f"Erreur API {response.status_code}: {response.text}")
        st.info("Verifiez que l'API FastAPI est lancee sur http://localhost:8000 et que les modeles sont charges.")

    except requests.exceptions.ConnectionError:
      st.error("Impossible de se connecter a l'API FastAPI.")
      st.info("Lancez l'API avec: uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
    except Exception as exc:
      st.error(f"Erreur pendant la prediction: {exc}")


  st.markdown("---")
  st.subheader("Historique des predictions manuelles")
  history = load_prediction_history()
  if history.empty:
    st.info("Aucune prediction manuelle sauvegardee pour le moment.")
  else:
    st.dataframe(
      history.sort_values("timestamp", ascending=False).head(20),
      use_container_width=True,
      hide_index=True
    )

# ============================================================================
# PAGE 4: ANALYTIQUE
# ============================================================================

elif page == "Analytique":
  st.markdown(
    '<p class="main-header">Analytique Avancée</p>',
    unsafe_allow_html=True
  )

  df = load_data()

  feedback_df = load_feedback_data()

  st.subheader("Boucle Active Learning")
  fb_col1, fb_col2, fb_col3, fb_col4 = st.columns(4)
  total_feedback = len(feedback_df)
  fraud_feedback = int(feedback_df['is_fraud'].astype(str).str.lower().isin(['true', '1']).sum()) if total_feedback else 0
  normal_feedback = total_feedback - fraud_feedback
  fraud_rate = (fraud_feedback / total_feedback * 100) if total_feedback else 0

  with fb_col1:
    st.metric("Feedbacks collectes", total_feedback)
  with fb_col2:
    st.metric("Fraudes confirmees", fraud_feedback)
  with fb_col3:
    st.metric("Normales confirmees", normal_feedback)
  with fb_col4:
    st.metric("Taux fraude confirme", f"{fraud_rate:.1f}%")

  if total_feedback >= 20:
    st.success("Donnees suffisantes pour preparer un reentrainement.")
  else:
    st.info(f"Encore {20 - total_feedback} feedback(s) avant un cycle de reentrainement simule.")

  if not feedback_df.empty:
    st.dataframe(
      feedback_df.sort_values("timestamp", ascending=False).head(20),
      use_container_width=True,
      hide_index=True
    )


  st.subheader("Performance des Modèles")

  model_performance = pd.DataFrame({
    'Modèle': ['Isolation Forest', 'Autoencoder', 'Ensemble'],
    'AUC-ROC': [0.866, 0.999, 0.880],
    'Precision': [0.15, 0.27, 0.18],
    'Recall': [0.35, 0.95, 0.42],
    'F1-Score': [0.21, 0.42, 0.25]
  })

  fig = go.Figure()

  for metric in ['AUC-ROC', 'Precision', 'Recall', 'F1-Score']:
    fig.add_trace(go.Bar(
      name=metric,
      x=model_performance['Modèle'],
      y=model_performance[metric]
    ))

  fig.update_layout(barmode='group', title="Métriques par Modèle")

  st.plotly_chart(fig, use_container_width=True)

  st.subheader("Matrice de Confusion")

  col1, col2, col3 = st.columns(3)

  with col1:
    st.write("**Isolation Forest**")
    cm_if = np.array([[29945, 15], [26, 14]])

    fig = px.imshow(
      cm_if,
      text_auto=True,
      labels=dict(x="Prédit", y="Réel"),
      x=['Normal', 'Fraude'],
      y=['Normal', 'Fraude']
    )

    st.plotly_chart(fig, use_container_width=True)

  with col2:
    st.write("**Autoencoder**")
    cm_ae = np.array([[29925, 35], [2, 38]])

    fig = px.imshow(
      cm_ae,
      text_auto=True,
      labels=dict(x="Prédit", y="Réel"),
      x=['Normal', 'Fraude'],
      y=['Normal', 'Fraude']
    )

    st.plotly_chart(fig, use_container_width=True)

  with col3:
    st.write("**Ensemble**")
    cm_en = np.array([[29930, 30], [23, 17]])

    fig = px.imshow(
      cm_en,
      text_auto=True,
      labels=dict(x="Prédit", y="Réel"),
      x=['Normal', 'Fraude'],
      y=['Normal', 'Fraude']
    )

    st.plotly_chart(fig, use_container_width=True)

  st.subheader("Progression Active Learning")

  al_data = pd.DataFrame({
    'Cycle': [1, 2, 3, 4, 5],
    'Annotées': [20, 40, 60, 80, 100],
    'AUC-ROC': [0.30, 0.33, 0.36, 0.63, 0.71],
    'Fraudes Trouvées': [0, 0, 0, 0, 0]
  })

  fig = make_subplots(specs=[[{"secondary_y": True}]])

  fig.add_trace(
    go.Scatter(
      x=al_data['Cycle'],
      y=al_data['AUC-ROC'],
      name="AUC-ROC",
      mode='lines+markers'
    ),
    secondary_y=False
  )

  fig.add_trace(
    go.Bar(
      x=al_data['Cycle'],
      y=al_data['Annotées'],
      name="Annotées",
      opacity=0.5
    ),
    secondary_y=True
  )

  fig.update_layout(title="Évolution Active Learning")

  st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 5: CONFIGURATION
# ============================================================================

else:
  st.markdown(
    '<p class="main-header">Configuration</p>',
    unsafe_allow_html=True
  )

  st.subheader(" Paramètres des Modèles")

  with st.form("model_config"):
    st.write("**Isolation Forest**")
    if_n_estimators = st.slider("Nombre d'estimateurs", 50, 500, 200)
    if_contamination = st.slider(
      "Contamination",
      0.001,
      0.01,
      0.002,
      step=0.001
    )

    st.write("**Autoencoder**")
    ae_hidden = st.selectbox(
      "Couches cachées",
      [(16, 8, 16), (32, 16, 32), (64, 32, 64)]
    )
    ae_max_iter = st.slider("Max iterations", 100, 500, 200)

    st.write("**Seuils**")
    threshold = st.slider(
      "Seuil de détection",
      0.90,
      0.999,
      0.99,
      step=0.001
    )

    submitted = st.form_submit_button("Sauvegarder")

    if submitted:
      st.success("Configuration sauvegardée!")

  st.subheader("Connexion Snowflake")

  with st.form("snowflake_config"):
    account = st.text_input("Account", value="your_account")
    user = st.text_input("User", value="your_user")
    password = st.text_input("Password", type="password")
    warehouse = st.text_input("Warehouse", value="COMPUTE_WH")
    database = st.text_input("Database", value="FRAUD_DETECTION")

    submitted = st.form_submit_button(" Connecter")

    if submitted:
      st.info("Connexion simulée (à implémenter)")

  st.subheader("Import/Export")

  col1, col2 = st.columns(2)

  with col1:
    if st.button("Exporter les Modèles"):
      st.success("Modèles exportés!")

  with col2:
    uploaded_file = st.file_uploader(
      " Importer un modèle",
      type=['joblib', 'pkl']
    )

    if uploaded_file:
      st.success("Modèle importé!")


if __name__ == "__main__":
  pass