import gradio as gr
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import os

# --- 1. CARGA DE MODELOS ---
MODELS_DIR = "models"
metadata = joblib.load(os.path.join(MODELS_DIR, 'risk_metadata.pkl'))
nn_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'risk_nn_model.keras'))

scaler = metadata['scaler']
rf_model = metadata['rf']
xgb_model = metadata['xgb']
top_features = metadata['top_features']
means_normal = metadata['means_normal']

# --- 2. LÓGICA DE PREDICCIÓN ---
def predict_fraud(monto, v14, v10, v12, v17, v11):
    # Creamos un vector con los promedios de una transacción normal
    input_data = pd.Series(means_normal).copy()
    
    # Reemplazamos los valores con lo que el usuario puso en la UI
    input_data['Amount'] = monto
    input_data['V14'] = v14
    input_data['V10'] = v10
    input_data['V12'] = v12
    input_data['V17'] = v17
    input_data['V11'] = v11
    
    # Convertir a DataFrame y escalar (manteniendo el orden de columnas original)
    df_input = pd.DataFrame([input_data]).drop('Class', axis=1, errors='ignore')
    X_scaled = scaler.transform(df_input)

    # Predicciones individuales (probabilidades)
    prob_rf = rf_model.predict_proba(X_scaled)[0][1]
    prob_xgb = xgb_model.predict_proba(X_scaled)[0][1]
    prob_nn = float(nn_model.predict(X_scaled, verbose=0)[0][0])

    # Score final (Ensamble Híbrido)
    avg_score = (prob_rf + prob_xgb + prob_nn) / 3
    
    # Veredicto e Impacto
    es_fraude = avg_score > 0.5
    veredicto = "🚨 FRAUDE DETECTADO" if es_fraude else "✅ TRANSACCIÓN LEGÍTIMA"
    color = "red" if es_fraude else "green"
    ahorro = f"${monto:.2f}" if es_fraude else "$0.00"

    return (
        {"Random Forest": prob_rf, "XGBoost": prob_xgb, "Neural Network": prob_nn},
        f"{avg_score:.2%}",
        veredicto,
        ahorro
    )

# --- 3. INTERFAZ VISUAL ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Risk Ops Console: Hybrid Fraud Detection")
    gr.Markdown("Simulá una transacción para evaluar el riesgo en tiempo real usando el ensamble híbrido.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📝 Datos de la Transacción")
            monto = gr.Number(label="Monto (USD)", value=100)
            gr.Markdown("---")
            gr.Markdown("⚠️ *Features de alto impacto detectadas por el modelo:*")
            v14 = gr.Slider(-20, 20, label="V14 (Anomalía Crítica)", value=means_normal['V14'])
            v12 = gr.Slider(-20, 20, label="V12 (Indicador de Riesgo)", value=means_normal['V12'])
            v10 = gr.Slider(-20, 20, label="V10", value=means_normal['V10'])
            v17 = gr.Slider(-20, 20, label="V17", value=means_normal['V17'])
            v11 = gr.Slider(-20, 20, label="V11", value=means_normal['V11'])
            btn = gr.Button("Evaluar Riesgo", variant="primary")

        with gr.Column():
            gr.Markdown("### 📊 Diagnóstico del Ensamble")
            label_score = gr.Label(label="Probabilidades por Modelo")
            with gr.Row():
                risk_pct = gr.Textbox(label="Score de Riesgo Promedio")
                verdict_text = gr.Textbox(label="Veredicto Final")
            
            roi_box = gr.Textbox(label="Ahorro Preventivo (ROI)", placeholder="$0.00")

    # Acción del botón
    btn.click(
        fn=predict_fraud, 
        inputs=[monto, v14, v10, v12, v17, v11], 
        outputs=[label_score, risk_pct, verdict_text, roi_box]
    )

if __name__ == "__main__":
    demo.launch()
