import gradio as gr
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import os

# --- 1. CONFIGURACIÓN DE RUTAS (Sincronizado con tus archivos reales) ---
MODELS_DIR = "models"
METADATA_PATH = os.path.join(MODELS_DIR, 'risk_ops_backup.pkl')
NN_MODEL_PATH = os.path.join(MODELS_DIR, 'risk_ops_nn.keras')

# --- 2. CARGA DE ACTIVOS DE IA ---
try:
    if not os.path.exists(METADATA_PATH) or not os.path.exists(NN_MODEL_PATH):
        raise FileNotFoundError(f"No se encuentran: {METADATA_PATH} o {NN_MODEL_PATH}")
    
    metadata = joblib.load(METADATA_PATH)
    # Cargamos con compile=False para mayor compatibilidad de versiones
    nn_model = tf.keras.models.load_model(NN_MODEL_PATH, compile=False)
    
    scaler = metadata['scaler']
    rf_model = metadata['rf_model']   # Nombre corregido según tu test
    xgb_model = metadata['xgb_model'] # Nombre corregido según tu test
    means_normal = metadata['means_normal']
    print("✅ Sistema Risk Ops: Modelos cargados y listos para operar.")

except Exception as e:
    print(f"❌ ERROR CRÍTICO: {e}")
    # Fallback para que la UI no rompa si falla la carga
    means_normal = {f'V{i}': 0.0 for i in range(1, 29)}
    means_normal.update({'Amount': 100, 'Time': 0})
    scaler = rf_model = xgb_model = nn_model = None

# --- 3. LÓGICA DE PROCESAMIENTO ---
def predict_fraud(monto, time, v14, v10, v12, v17, v11):
    """
    Recibe datos de la interfaz, asegura el orden de columnas y predice.
    """
    try:
        # 1. Crear diccionario base con promedios normales
        full_features = means_normal.copy()
        
        # 2. Inyectar valores de la interfaz
        full_features['Amount'] = monto
        full_features['Time'] = time
        full_features['V14'] = v14
        full_features['V10'] = v10
        full_features['V12'] = v12
        full_features['V17'] = v17
        full_features['V11'] = v11
        
        # 3. Definir el orden EXACTO de columnas (Igual que en el entrenamiento)
        col_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        # 4. Crear DataFrame y reordenar
        df_input = pd.DataFrame([full_features])
        # Rellenar cualquier columna faltante con 0.0
        for col in col_order:
            if col not in df_input.columns:
                df_input[col] = 0.0
        
        df_input = df_input[col_order]
        
        # 5. Escalar datos (Sin warnings de nombres de features)
        X_scaled = scaler.transform(df_input)

        # 6. Obtener predicciones (Probabilidades de clase 1)
        prob_rf = float(rf_model.predict_proba(X_scaled)[0][1])
        prob_xgb = float(xgb_model.predict_proba(X_scaled)[0][1])
        prob_nn = float(nn_model.predict(X_scaled, verbose=0)[0][0])

        # 7. Cálculo del Ensamble (Promedio ponderado)
        avg_score = (prob_rf * 0.2 + prob_xgb * 0.4 + prob_nn * 0.4)
        
        # 8. Formateo de resultados
        es_fraude = avg_score > 0.5
        veredicto = "🚨 ALTA PROBABILIDAD DE FRAUDE" if es_fraude else "✅ TRANSACCIÓN CONFIABLE"
        ahorro = f"${monto:,.2f}" if es_fraude else "$0.00"
        
        scores_dict = {
            "Random Forest": prob_rf,
            "XGBoost": prob_xgb,
            "Neural Network": prob_nn
        }

        return scores_dict, f"{avg_score:.2%}", veredicto, ahorro

    except Exception as e:
        return {"Error": 0}, "0%", f"Error: {str(e)}", "$0.00"

# --- 4. INTERFAZ VISUAL (GRADIO) ---
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🛡️ Risk Ops Console: Dashboard de Detección Híbrida")
    
    with gr.Row():
        # Columna Izquierda: Entradas
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Entrada de Datos")
            monto = gr.Number(label="Monto de la Transacción (USD)", value=100)
            time = gr.Number(label="Time (Segundos desde inicio)", value=0)
            
            with gr.Accordion("Variables de Comportamiento (Features)", open=True):
                v14 = gr.Slider(-15, 15, label="V14 (Anomalía de Red)", value=means_normal.get('V14', 0))
                v12 = gr.Slider(-15, 15, label="V12 (Historial)", value=means_normal.get('V12', 0))
                v10 = gr.Slider(-15, 15, label="V10 (Ubicación)", value=means_normal.get('V10', 0))
                v17 = gr.Slider(-15, 15, label="V17", value=means_normal.get('V17', 0))
                v11 = gr.Slider(-15, 15, label="V11", value=means_normal.get('V11', 0))
            
            btn = gr.Button("EJECUTAR ANÁLISIS", variant="primary")

        # Columna Derecha: Resultados
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Diagnóstico de IA")
            label_plot = gr.Label(label="Confianza por Modelo Individual")
            
            with gr.Row():
                risk_pct = gr.Textbox(label="Probabilidad Agregada")
                verdict_text = gr.Textbox(label="Veredicto Final")
            
            gr.Markdown("### 💰 Impacto Financiero")
            roi_box = gr.Textbox(label="Pérdida Evitada (Ahorro)", placeholder="$0.00")

    btn.click(
        fn=predict_fraud, 
        inputs=[monto, time, v14, v10, v12, v17, v11], 
        outputs=[label_plot, risk_pct, verdict_text, roi_box]
    )

if __name__ == "__main__":
    demo.launch(server_port=7861) # Usamos el 7861 para no chocar con console.py
