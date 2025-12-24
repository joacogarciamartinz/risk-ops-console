"""
Risk Ops Training Pipeline
============================
Entrena un ensemble de modelos (RF + XGB + Deep Learning) y exporta
metadatos compatibles con console.py.

Dataset: Credit Card Fraud Detection (Kaggle)
Estrategia: SMOTE para balanceo + Ensemble híbrido
Versión: 1.0-Hybrid
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================================
# CONFIGURACIÓN DEL PIPELINE
# ============================================================================

CONFIG = {
    "RANDOM_SEED": 42,
    "DATA_PATH": "creditcard.csv",
    "MODELS_DIR": "models",
    "RESULTS_DIR": "results",
    "TOP_FEATURES": ['V14', 'V10', 'V12', 'V17', 'V11'],
    
    # Hyperparámetros
    "RF_N_ESTIMATORS": 100,
    "XGB_SCALE_POS_WEIGHT": 100,
    "SMOTE_SAMPLING_STRATEGY": 0.1,
    "NN_EPOCHS": 5,
    "NN_BATCH_SIZE": 2048
}

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ============================================================================
# PIPELINE DE ENTRENAMIENTO
# ============================================================================

def train_pipeline():
    """
    Pipeline completo de entrenamiento:
        1. Carga de datos
        2. Análisis estadístico (medias por clase)
        3. Preprocesamiento y escalado
        4. Balanceo con SMOTE
        5. Entrenamiento de ensemble (RF + XGB + DL)
        6. Exportación compatible con console.py
    """
    
    logging.info("="*70)
    logging.info("INICIANDO PIPELINE DE ENTRENAMIENTO")
    logging.info("="*70)
    
    # ========================================================================
    # PASO 1: Carga de Datos
    # ========================================================================
    
    logging.info("\n[PASO 1/6] Cargando dataset...")
    
    if not os.path.exists(CONFIG["DATA_PATH"]):
        logging.error(f"❌ No se encontró el archivo: {CONFIG['DATA_PATH']}")
        logging.error("Asegúrate de descargar el dataset de Kaggle:")
        logging.error("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        return
    
    df = pd.read_csv(CONFIG["DATA_PATH"])
    logging.info(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Análisis de balance de clases
    class_counts = df['Class'].value_counts()
    logging.info(f"   - Transacciones legítimas: {class_counts[0]} ({class_counts[0]/len(df)*100:.2f}%)")
    logging.info(f"   - Transacciones fraudulentas: {class_counts[1]} ({class_counts[1]/len(df)*100:.2f}%)")
    
    # ========================================================================
    # PASO 2: Análisis Estadístico para UI
    # ========================================================================
    
    logging.info("\n[PASO 2/6] Calculando perfiles estadísticos...")
    
    # Separar por clase ANTES de cualquier transformación
    df_normal = df[df['Class'] == 0]
    df_fraud = df[df['Class'] == 1]
    
    # Calcular medias de todas las features (para scaler y UI)
    means_normal_all = df_normal.mean().to_dict()
    means_fraud_all = df_fraud.mean().to_dict()
    
    # Extraer solo las Top Features para la UI
    means_normal_ui = {feat: means_normal_all[feat] for feat in CONFIG["TOP_FEATURES"]}
    means_fraud_ui = {feat: means_fraud_all[feat] for feat in CONFIG["TOP_FEATURES"]}
    
    logging.info(f"✅ Perfiles calculados para {len(CONFIG['TOP_FEATURES'])} features críticas")
    logging.info(f"   Ejemplo (V14): Normal={means_normal_ui['V14']:.4f}, Fraude={means_fraud_ui['V14']:.4f}")
    
    # ========================================================================
    # PASO 3: Preprocesamiento
    # ========================================================================
    
    logging.info("\n[PASO 3/6] Preprocesando datos...")
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Escalar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split train/test con estratificación
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.3, 
        stratify=y, 
        random_state=CONFIG["RANDOM_SEED"]
    )
    
    logging.info(f"✅ Split completado:")
    logging.info(f"   - Train: {X_train.shape[0]} muestras")
    logging.info(f"   - Test: {X_test.shape[0]} muestras")
    
    # ========================================================================
    # PASO 4: Balanceo con SMOTE
    # ========================================================================
    
    logging.info("\n[PASO 4/6] Aplicando SMOTE para balanceo...")
    
    smote = SMOTE(
        sampling_strategy=CONFIG["SMOTE_SAMPLING_STRATEGY"],
        random_state=CONFIG["RANDOM_SEED"]
    )
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    train_fraud_count = (y_train_resampled == 1).sum()
    train_normal_count = (y_train_resampled == 0).sum()
    
    logging.info(f"✅ SMOTE aplicado:")
    logging.info(f"   - Antes: {(y_train == 1).sum()} fraudes")
    logging.info(f"   - Después: {train_fraud_count} fraudes")
    logging.info(f"   - Ratio final: {train_fraud_count/train_normal_count:.2%}")
    
    # ========================================================================
    # PASO 5: Entrenamiento del Ensemble
    # ========================================================================
    
    logging.info("\n[PASO 5/6] Entrenando ensemble de modelos...")
    
    # ------------------------------------------------------------------------
    # Modelo 1: Random Forest
    # ------------------------------------------------------------------------
    logging.info("\n   [1/3] Entrenando Random Forest...")
    
    rf_model = RandomForestClassifier(
        n_estimators=CONFIG["RF_N_ESTIMATORS"],
        class_weight='balanced',
        random_state=CONFIG["RANDOM_SEED"],
        n_jobs=-1
    )
    rf_model.fit(X_train_resampled, y_train_resampled)
    
    rf_pred = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_pred)
    
    logging.info(f"   ✅ Random Forest entrenado - AUC: {rf_auc:.4f}")
    
    # ------------------------------------------------------------------------
    # Modelo 2: XGBoost
    # ------------------------------------------------------------------------
    logging.info("\n   [2/3] Entrenando XGBoost...")
    
    xgb_model = XGBClassifier(
        scale_pos_weight=CONFIG["XGB_SCALE_POS_WEIGHT"],
        eval_metric='logloss',
        random_state=CONFIG["RANDOM_SEED"],
        n_jobs=-1
    )
    xgb_model.fit(X_train_resampled, y_train_resampled)
    
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_pred)
    
    logging.info(f"   ✅ XGBoost entrenado - AUC: {xgb_auc:.4f}")
    
    # ------------------------------------------------------------------------
    # Modelo 3: Red Neuronal Profunda
    # ------------------------------------------------------------------------
    logging.info("\n   [3/3] Entrenando Red Neuronal...")
    
    nn_model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1, activation='sigmoid')
    ])
    
    nn_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    
    history = nn_model.fit(
        X_train_resampled, y_train_resampled,
        epochs=CONFIG["NN_EPOCHS"],
        batch_size=CONFIG["NN_BATCH_SIZE"],
        validation_split=0.2,
        verbose=0
    )
    
    nn_pred = nn_model.predict(X_test, verbose=0).flatten()
    nn_auc = roc_auc_score(y_test, nn_pred)
    
    logging.info(f"   ✅ Red Neuronal entrenada - AUC: {nn_auc:.4f}")
    
    # ------------------------------------------------------------------------
    # Evaluación del Ensemble
    # ------------------------------------------------------------------------
    logging.info("\n   📊 Evaluando ensemble...")
    
    ensemble_pred = (rf_pred * 0.30 + xgb_pred * 0.35 + nn_pred * 0.35)
    ensemble_auc = roc_auc_score(y_test, ensemble_pred)
    
    logging.info(f"   ✅ Ensemble AUC: {ensemble_auc:.4f}")
    logging.info(f"      - Random Forest: {rf_auc:.4f}")
    logging.info(f"      - XGBoost: {xgb_auc:.4f}")
    logging.info(f"      - Deep Learning: {nn_auc:.4f}")
    
    # ========================================================================
    # PASO 6: Exportación Compatible con console.py
    # ========================================================================
    
    logging.info("\n[PASO 6/6] Exportando modelos y metadatos...")
    
    # Crear directorio de modelos
    os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)
    
    # ------------------------------------------------------------------------
    # ARCHIVO 1: risk_ops_backup.pkl
    # ------------------------------------------------------------------------
    # Contiene: RF + XGB + Scaler + Metadatos UI
    # IMPORTANTE: Las llaves deben coincidir con console.py
    
    backup_data = {
        'rf_model': rf_model,        # ✅ Coincide con console.py
        'xgb_model': xgb_model,      # ✅ Coincide con console.py
        'scaler': scaler,            # ✅ Coincide con console.py
        'means_normal': means_normal_ui,  # ✅ Solo Top Features
        'means_fraud': means_fraud_ui,    # ✅ Solo Top Features
        'top_features': CONFIG["TOP_FEATURES"],
        
        # Metadatos adicionales (opcional)
        'training_info': {
            'rf_auc': rf_auc,
            'xgb_auc': xgb_auc,
            'nn_auc': nn_auc,
            'ensemble_auc': ensemble_auc,
            'smote_strategy': CONFIG["SMOTE_SAMPLING_STRATEGY"],
            'random_seed': CONFIG["RANDOM_SEED"]
        }
    }
    
    backup_path = os.path.join(CONFIG["MODELS_DIR"], 'risk_ops_backup.pkl')
    joblib.dump(backup_data, backup_path)
    
    file_size_mb = os.path.getsize(backup_path) / (1024 * 1024)
    logging.info(f"   ✅ risk_ops_backup.pkl guardado ({file_size_mb:.2f} MB)")
    
    # ------------------------------------------------------------------------
    # ARCHIVO 2: risk_ops_nn.keras
    # ------------------------------------------------------------------------
    # Contiene: Red Neuronal (arquitectura + pesos)
    
    nn_path = os.path.join(CONFIG["MODELS_DIR"], 'risk_ops_nn.keras')
    nn_model.save(nn_path)
    
    file_size_mb = os.path.getsize(nn_path) / (1024 * 1024)
    logging.info(f"   ✅ risk_ops_nn.keras guardado ({file_size_mb:.2f} MB)")
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    
    logging.info("\n" + "="*70)
    logging.info("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    logging.info("="*70)
    logging.info("\n📦 Archivos generados en /models:")
    logging.info(f"   - risk_ops_backup.pkl (RF + XGB + Scaler + Metadata)")
    logging.info(f"   - risk_ops_nn.keras (Red Neuronal)")
    logging.info("\n🚀 Siguiente paso:")
    logging.info(f"   Ejecuta: iniciar_consola.bat")
    logging.info("="*70 + "\n")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    train_pipeline()
