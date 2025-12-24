import gradio as gr
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import os

# --- 1. CONFIGURACIÓN DE RUTAS ---
MODELS_DIR = "models"
METADATA_PATH = os.path.join(MODELS_DIR, 'risk_metadata.pkl')
NN_MODEL_PATH = os.path.join(MODELS_DIR, 'risk_nn_model.keras')

# --- 2. CARGA DE ACTIVOS DE IA ---
# Intentamos cargar los modelos pre-entrenados que ya deberían estar en el repo
try:
    if not os.path.exists(METADATA_PATH) or not os.path.exists(NN_MODEL_PATH):
        raise FileNotFoundError("Los archivos de modelo no se encuentran en la carpeta /models")
    
    metadata = joblib.load(METADATA_PATH)
    nn_model = tf.keras.models.load_model(NN_MODEL_PATH)
    
    # Extraemos componentes del metadata
    scaler = metadata['scaler']
    rf_model = metadata['rf']
    xgb_model = metadata['xgb']
    means_normal = metadata['means_normal']
    print("✅ Sistema Risk Ops: Modelos cargados y listos para operar.")

except Exception as e:
    print(f"❌ ERROR CRÍTICO: No se pudieron cargar los modelos. Detalle: {e}")
    print("Asegúrate de haber ejecutado main.py al menos una vez o de haber descargado la carpeta /models.")

# --- 3. LÓGICA DE PROCESAMIENTO ---
def predict_fraud(monto, v14, v10, v12, v17, v11):
    """
    Recibe datos de la interfaz, completa con promedios normales, 
    escala y predice con el ensamble.
    """
    # 1. Crear vector de entrada basado en promedios normales
    input_data = pd.Series(means_normal).copy()
    
    # 2. Inyectar valores de la interfaz
    input_data['Amount'] = monto
    input_data['V14'] = v14
    input_data['V10'] = v10
    input_data['V12'] = v12
    input_data['V17'] = v17
    input_data['V11'] = v11
    
    # 3. Preparar DataFrame para el Scaler (eliminando 'Class' si existe)
    df_input = pd.DataFrame([input_data]).drop('Class', axis=1, errors='ignore')
    
    # 4. Escalar datos
    X_scaled = scaler.transform(df_input)

    # 5. Obtener predicciones de los 3 modelos
    prob_rf = rf_model.predict_proba(X_scaled)[0][1]
    prob_xgb = xgb_model.predict_proba(X_scaled)[0][1]
    prob_nn = float(nn_model.predict(X_scaled, verbose=0)[0][0])

    # 6. Cálculo del Ensamble Híbrido (Promedio ponderado)
    avg_score = (prob_rf + prob_xgb + prob_nn) / 3
    
    # 7. Formateo de salida para Risk Ops
    es_fraude = avg_score > 0.5
    veredicto = "🚨 ALTA PROBABILIDAD DE FRAUDE" if es_fraude else "✅ TRANSACCIÓN CONFIABLE"
    ahorro = f"${monto:,.2f}" if es_fraude else "$0.00"
    
    # Formatear scores para el gráfico de barras
    scores_dict = {
        "Random Forest": prob_rf,
        "XGBoost": prob_xgb,
        "Neural Network": prob_nn
    }

    return (
        scores_dict,
        f"{avg_score:.2%}",
        veredicto,
        ahorro
    )

# --- 4. INTERFAZ VISUAL (GRADIO) ---
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🛡️ Risk Ops Console: Dashboard de Detección Híbrida")
    gr.Markdown("""
    **Operaciones de Riesgo:** Utilice este panel para evaluar transacciones sospechosas. 
    El sistema utiliza un ensamble de **IA Híbrida** para minimizar falsos positivos.
    """)
    
    with gr.Row():
        # Columna Izquierda: Entradas
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Entrada de Datos")
            monto = gr.Number(label="Monto de la Transacción (USD)", value=100)
            
            with gr.Accordion("Variables de Comportamiento (Features)", open=True):
                gr.Markdown("*Ajuste los valores de las variables detectadas como críticas por el modelo:*")
                v14 = gr.Slider(-15, 15, label="V14 (Anomalía de Red)", value=means_normal.get('V14', 0))
                v12 = gr.Slider(-15, 15, label="V12 (Historial de Cuenta)", value=means_normal.get('V12', 0))
                v10 = gr.Slider(-15, 15, label="V10 (Ubicación IP)", value=means_normal.get('V10', 0))
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
            
            gr.Markdown("> **Nota:** El veredicto final se basa en el consenso del ensamble superando el umbral del 50%.")

    # Definir la interacción
    btn.click(
        fn=predict_fraud, 
        inputs=[monto, v14, v10, v12, v17, v11], 
        outputs=[label_plot, risk_pct, verdict_text, roi_box]
    )

# Lanzamiento de la aplicación
if __name__ == "__main__":
    demo.launch()
