"""
Risk Ops Training Pipeline
Description: Trains an ensemble of models and exports metadata for the local console.
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. Configuración ---
CONFIG = {
    "RANDOM_SEED": 42,
    "DATA_PATH": "creditcard.csv",  # Ajustado a tu raíz según tu imagen
    "MODELS_DIR": "models",
    "RESULTS_DIR": "results",
    "TOP_FEATURES": ['V14', 'V10', 'V12', 'V17', 'V11']
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def train_pipeline():
    # A. Carga de datos
    if not os.path.exists(CONFIG["DATA_PATH"]):
        logging.error(f"No se encontró el archivo {CONFIG['DATA_PATH']}")
        return

    df = pd.read_csv(CONFIG["DATA_PATH"])
    logging.info(f"Dataset cargado: {df.shape}")

    # B. Cálculo de promedios para la Consola (Crucial)
    logging.info("Calculando perfiles estadísticos...")
    means_normal = df[df['Class'] == 0].mean().to_dict()
    means_fraud = df[df['Class'] == 1].mean().to_dict()

    # C. Preprocesamiento
    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=CONFIG["RANDOM_SEED"]
    )

    # D. Balanceo con SMOTE
    logging.info("Aplicando SMOTE...")
    smote = SMOTE(sampling_strategy=0.1, random_state=CONFIG["RANDOM_SEED"])
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # E. Entrenamiento de Ensamble
    logging.info("Entrenando Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=CONFIG["RANDOM_SEED"])
    rf.fit(X_train_res, y_train_res)

    logging.info("Entrenando XGBoost...")
    xgb = XGBClassifier(scale_pos_weight=100, eval_metric='logloss', random_state=CONFIG["RANDOM_SEED"])
    xgb.fit(X_train_res, y_train_res)

    logging.info("Entrenando Red Neuronal...")
    nn = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    nn.fit(X_train_res, y_train_res, epochs=5, batch_size=2048, verbose=0)

    # F. EXPORTACIÓN PARA RISK OPS
    logging.info("Exportando modelos y metadatos compatibles...")
    os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)

    # Paquete de metadatos (Modelos ML + Scaler + Estadísticas)
    risk_metadata = {
        'rf': rf,
        'xgb': xgb,
        'scaler': scaler,
        'top_features': CONFIG["TOP_FEATURES"],
        'means_normal': means_normal,
        'means_fraud': means_fraud
    }
    
    joblib.dump(risk_metadata, os.path.join(CONFIG["MODELS_DIR"], 'risk_metadata.pkl'))
    nn.save(os.path.join(CONFIG["MODELS_DIR"], 'risk_nn_model.keras'))

    logging.info(f"✅ ÉXITO: Archivos guardados en /{CONFIG['MODELS_DIR']}")

if __name__ == "__main__":
    train_pipeline()