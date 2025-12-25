"""
Risk Ops Console - Interfaz de Detección de Fraude
====================================================
Versión: 1.2-ContextAware (Mejorado)
Arquitectura: Ensemble de 3 modelos (RF + XGB + Deep Learning)
Mejora: Selector de Escenario para evitar el problema del "Relleno con Ceros".
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

print("="*70)
print("RISK OPS CONSOLE - Sistema de Detección de Fraude")
print("="*70)

# Archivos sincronizados
BACKUP_FILE = "risk_ops_backup.pkl"
NN_MODEL_FILE = "risk_ops_nn.keras"

# ============================================================================
# VALIDACIÓN DE DEPENDENCIAS
# ============================================================================
print("[INIT] Verificando dependencias...")
try:
    import joblib
    import numpy as np
    import pandas as pd
    from tensorflow import keras
    import tensorflow as tf
    import gradio as gr
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    print("✅ Todas las dependencias están OK.")
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# ============================================================================
# CARGA DE MODELOS
# ============================================================================

def load_hybrid_models():
    models = {}
    backup_path = MODELS_DIR / BACKUP_FILE
    
    if not backup_path.exists():
        raise FileNotFoundError(f"No se encontró {BACKUP_FILE}")
    
    try:
        backup_data = joblib.load(backup_path)
        models['rf_model'] = backup_data.get('rf_model')
        models['xgb_model'] = backup_data.get('xgb_model')
        models['scaler'] = backup_data.get('scaler')
        models['means_normal'] = backup_data.get('means_normal')
        models['means_fraud'] = backup_data.get('means_fraud')
        
        # Valores de seguridad si el backup no traía las medias completas
        if not models['means_fraud'] or len(models['means_fraud']) < 10:
            models['means_fraud'] = {
                'V1': -4.77, 'V2': 3.62, 'V3': -7.03, 'V4': 4.54, 'V5': -3.15,
                'V6': -1.39, 'V7': -5.56, 'V8': 0.57, 'V9': -2.58, 'V10': -5.67,
                'V11': 3.80, 'V12': -6.25, 'V13': -0.10, 'V14': -6.97, 'V15': -0.09,
                'V16': -4.13, 'V17': -6.66, 'V18': -2.24, 'V19': 0.68, 'V20': 0.37,
                'V21': 0.71, 'V22': 0.01, 'V23': -0.04, 'V24': -0.10, 'V25': 0.04,
                'V26': 0.05, 'V27': 0.17, 'V28': 0.07, 'Amount': 122.21, 'Time': 80000
            }
            
    except Exception as e:
        print(f"[ERROR] Error al cargar backup: {e}")
        raise

    nn_path = MODELS_DIR / NN_MODEL_FILE
    if nn_path.exists():
        models['nn_model'] = keras.models.load_model(nn_path)
    
    return models

# Inicialización
MODELS = load_hybrid_models()
rf_model = MODELS['rf_model']
xgb_model = MODELS['xgb_model']
nn_model = MODELS['nn_model']
scaler = MODELS['scaler']
means_normal = MODELS['means_normal']
means_fraud = MODELS['means_fraud']

# ============================================================================
# PREDICCIÓN CON INYECCIÓN DE CONTEXTO
# ============================================================================

def predict_fraud_ensemble(features_dict):
    """ Mantiene tu lógica de diagnóstico pero asegura el orden """
    try:
        print("\n" + "="*40)
        print("🔍 INICIO DIAGNÓSTICO DE PREDICCIÓN")
        print("="*40)

        col_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        input_df = pd.DataFrame([features_dict])
        
        # Rellenar faltantes (ya vienen pre-rellenos por analyze_transaction)
        for col in col_order:
            if col not in input_df.columns:
                input_df[col] = 0.0
        
        input_df = input_df[col_order]

        print(f"📥 Datos de entrada (Shape): {input_df.shape}")
        print(f"🔹 Valor 'V14' (Crudo): {input_df['V14'].values[0]}")
        
        if hasattr(scaler, 'feature_names_in_'):
            input_df = input_df[list(scaler.feature_names_in_)]
            print("✅ Orden de columnas validado.")

        features_scaled = scaler.transform(input_df)
        
        # Predicciones
        rf_prob = float(rf_model.predict_proba(features_scaled)[0][1])
        xgb_prob = float(xgb_model.predict_proba(features_scaled)[0][1])
        nn_prob = float(nn_model.predict(features_scaled, verbose=0)[0][0])
        
        ensemble_score = (0.20 * rf_prob) + (0.40 * xgb_prob) + (0.40 * nn_prob)
        
        print(f"🤖 RF: {rf_prob:.4f} | XGB: {xgb_prob:.4f} | NN: {nn_prob:.4f}")
        print(f"🏁 Score Final: {ensemble_score:.4f}")
        print("="*40 + "\n")

        # Clasificación
        if ensemble_score >= 0.8: level, rec, color = "CRÍTICO", "🚫 BLOQUEAR", "red"
        elif ensemble_score >= 0.5: level, rec, color = "ALTO", "⚠️ REVISIÓN", "orange"
        elif ensemble_score >= 0.3: level, rec, color = "MEDIO", "⚡ MONITOREAR", "yellow"
        else: level, rec, color = "BAJO", "✅ APROBAR", "green"
        
        return {
            'ensemble_score': round(ensemble_score, 4),
            'risk_level': level,
            'recommendation': rec,
            'individual_predictions': {'rf': rf_prob, 'xgb': xgb_prob, 'nn': nn_prob}
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ============================================================================
# INTERFAZ GRADIO
# ============================================================================

def create_gradio_interface():
    
    def analyze_transaction(scenario, v14, v10, v12, v17, v11, amount, time):
        # EL TRUCO: Si el usuario elige simular fraude, cargamos el ADN de fraude de fondo
        if scenario == "Simular Escenario Criminal":
            features = means_fraud.copy()
        else:
            features = {col: 0.0 for col in (['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])}
            # Aplicar medias normales si existen
            if means_normal: features.update(means_normal)
        
        # Sobrescribir con lo que el usuario mueve en los sliders
        features.update({
            'V14': v14, 'V10': v10, 'V12': v12, 'V17': v17, 'V11': v11,
            'Amount': amount, 'Time': time
        })
        
        result = predict_fraud_ensemble(features)
        if not result: return "Error en el sistema"
        
        output = f"## Riesgo: {result['risk_level']}\n"
        output += f"### Probabilidad Total: **{result['ensemble_score']:.2%}**\n"
        output += f"**Acción recomendada:** {result['recommendation']}\n\n---\n"
        output += f"**Desglose:** RF: {result['individual_predictions']['rf']:.2%} | "
        output += f"XGB: {result['individual_predictions']['xgb']:.2%} | "
        output += f"NN: {result['individual_predictions']['nn']:.2%}"
        return output

    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🛡️ Risk Ops Console v1.2")
        
        with gr.Row():
            with gr.Column():
                scenario_input = gr.Radio(
                    ["Cliente Normal", "Simular Escenario Criminal"], 
                    value="Cliente Normal", 
                    label="1. Contexto de la Transacción",
                    info="¿Cómo rellenar las variables ocultas?"
                )
                
                gr.Markdown("### 2. Variables del Slider")
                v14_input = gr.Slider(-15, 5, value=-1.0, label="V14")
                v10_input = gr.Slider(-15, 5, value=-0.5, label="V10")
                v12_input = gr.Slider(-15, 5, value=-0.3, label="V12")
                v17_input = gr.Slider(-15, 5, value=-0.2, label="V17")
                v11_input = gr.Slider(-5, 10, value=0.5, label="V11")
                
                amount_input = gr.Number(value=100.0, label="Monto ($)")
                time_input = gr.Number(value=0, label="Timestamp")
                
                analyze_btn = gr.Button("🔍 ANALIZAR", variant="primary")

            with gr.Column():
                gr.Markdown("### 3. Resultado Forense")
                output_markdown = gr.Markdown("Esperando análisis...")
                
                gr.Examples(
                    examples=[
                        ["Cliente Normal", -1.0, -0.5, -0.3, -0.2, 0.5, 50, 1000],
                        ["Simular Escenario Criminal", -12.0, -8.0, -9.0, -10.0, 5.0, 1500, 5000]
                    ],
                    inputs=[scenario_input, v14_input, v10_input, v12_input, v17_input, v11_input, amount_input, time_input]
                )

        analyze_btn.click(
            fn=analyze_transaction,
            inputs=[scenario_input, v14_input, v10_input, v12_input, v17_input, v11_input, amount_input, time_input],
            outputs=output_markdown
        )
    
    return interface

if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(server_name="127.0.0.1", server_port=7860)
