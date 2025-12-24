import os
import sys
from pathlib import Path

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

print(f"[INFO] Directorio base: {BASE_DIR}")
print(f"[INFO] Directorio de modelos: {MODELS_DIR}")

# ============================================================================
# NOMBRES DE ARCHIVOS - SINCRONIZADOS CON TU ESTRUCTURA
# ============================================================================
METADATA_FILE = "risk_ops_backup.pkl"
NN_MODEL_FILE = "risk_ops_nn.keras"

# ============================================================================
# CARGA DE DEPENDENCIAS CON VALIDACIÓN
# ============================================================================

print("\n[INFO] Verificando dependencias...")

# Importaciones core
try:
    import joblib
    import numpy as np
    import pandas as pd
    print("✓ NumPy, Pandas, Joblib")
except ImportError as e:
    print(f"✗ Error importando dependencias básicas: {e}")
    sys.exit(1)

# TensorFlow/Keras
try:
    from tensorflow import keras
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
except ImportError:
    print("✗ TensorFlow no disponible - pip install tensorflow")
    keras = None

# Scikit-learn
try:
    from sklearn.preprocessing import StandardScaler
    print("✓ Scikit-learn")
except ImportError:
    print("⚠ Scikit-learn no disponible (opcional)")
    StandardScaler = None

# XGBoost
try:
    import xgboost as xgb
    print("✓ XGBoost")
except ImportError:
    print("⚠ XGBoost no disponible (opcional)")
    xgb = None

# Gradio
try:
    import gradio as gr
    print(f"✓ Gradio {gr.__version__}")
except ImportError:
    print("✗ Gradio no disponible - pip install gradio")
    gr = None

print()

# ============================================================================
# FUNCIONES DE CARGA DE MODELOS
# ============================================================================

def load_metadata():
    """Carga el archivo de metadatos (estadísticas de entrenamiento)."""
    metadata_path = MODELS_DIR / METADATA_FILE
    
    print(f"[INFO] Cargando metadatos: {metadata_path}")
    
    if not metadata_path.exists():
        print(f"[ERROR] Archivo no encontrado: {metadata_path}")
        print(f"[INFO] Contenido de {MODELS_DIR}:")
        for file in MODELS_DIR.iterdir():
            print(f"  - {file.name}")
        raise FileNotFoundError(f"No se encontró {METADATA_FILE}")
    
    try:
        metadata = joblib.load(metadata_path)
        print(f"[OK] Metadatos cargados - Llaves: {list(metadata.keys())}")
        
        # Validar estructura mínima
        required = ['means_normal', 'stds_normal', 'means_fraud', 'stds_fraud']
        missing = [k for k in required if k not in metadata]
        
        if missing:
            print(f"[WARNING] Llaves faltantes en metadata: {missing}")
            print("[WARNING] Usando valores por defecto para llaves faltantes")
            
            # Valores por defecto
            defaults = {
                'means_normal': {'amount': 100.0, 'merchant_risk': 0.3, 'hour': 12, 'day_of_week': 3},
                'stds_normal': {'amount': 80.0, 'merchant_risk': 0.2, 'hour': 6, 'day_of_week': 2},
                'means_fraud': {'amount': 500.0, 'merchant_risk': 0.8, 'hour': 2, 'day_of_week': 5},
                'stds_fraud': {'amount': 300.0, 'merchant_risk': 0.15, 'hour': 4, 'day_of_week': 1}
            }
            
            for key in missing:
                metadata[key] = defaults.get(key, {})
        
        return metadata
        
    except Exception as e:
        print(f"[ERROR] Error al cargar metadatos: {e}")
        print("[WARNING] Retornando metadatos por defecto")
        return {
            'means_normal': {'amount': 100.0, 'merchant_risk': 0.3, 'hour': 12, 'day_of_week': 3},
            'stds_normal': {'amount': 80.0, 'merchant_risk': 0.2, 'hour': 6, 'day_of_week': 2},
            'means_fraud': {'amount': 500.0, 'merchant_risk': 0.8, 'hour': 2, 'day_of_week': 5},
            'stds_fraud': {'amount': 300.0, 'merchant_risk': 0.15, 'hour': 4, 'day_of_week': 1}
        }


def load_nn_model():
    """Carga el modelo de red neuronal."""
    if keras is None:
        print("[ERROR] TensorFlow no disponible - no se puede cargar modelo NN")
        return None
    
    model_path = MODELS_DIR / NN_MODEL_FILE
    
    print(f"[INFO] Cargando modelo NN: {model_path}")
    
    if not model_path.exists():
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        return None
    
    try:
        model = keras.models.load_model(model_path)
        print(f"[OK] Modelo NN cargado - Capas: {len(model.layers)}")
        return model
    except Exception as e:
        print(f"[ERROR] Error al cargar modelo NN: {e}")
        return None


def load_optional_models():
    """Carga modelos opcionales (XGBoost, RandomForest, Scaler)."""
    models = {}
    
    optional_files = {
        'xgb': 'risk_xgb_model.pkl',
        'rf': 'risk_rf_model.pkl',
        'scaler': 'risk_scaler.pkl'
    }
    
    for key, filename in optional_files.items():
        filepath = MODELS_DIR / filename
        
        if filepath.exists():
            try:
                models[key] = joblib.load(filepath)
                print(f"[OK] {key.upper()} cargado desde {filename}")
            except Exception as e:
                print(f"[WARNING] Error al cargar {filename}: {e}")
                models[key] = None
        else:
            models[key] = None
    
    return models


# ============================================================================
# INICIALIZACIÓN GLOBAL
# ============================================================================

print("="*60)
print("INICIALIZANDO SISTEMA DE DETECCIÓN DE FRAUDE")
print("="*60)
print()

# Crear directorio de modelos si no existe
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Cargar componentes
try:
    metadata = load_metadata()
    means_normal = metadata['means_normal']
    stds_normal = metadata['stds_normal']
    means_fraud = metadata['means_fraud']
    stds_fraud = metadata['stds_fraud']
except Exception as e:
    print(f"[CRITICAL] Fallo al cargar metadatos: {e}")
    print("[CRITICAL] El sistema no puede funcionar sin metadatos válidos")
    sys.exit(1)

nn_model = load_nn_model()

if nn_model is None:
    print("[CRITICAL] Modelo de red neuronal no disponible")
    print("[CRITICAL] El sistema requiere al menos el modelo NN para funcionar")
    sys.exit(1)

optional = load_optional_models()
xgb_model = optional.get('xgb')
rf_model = optional.get('rf')
scaler = optional.get('scaler')

print()
print("="*60)
print("RESUMEN DE CARGA")
print("="*60)
print(f"Red Neuronal:  {'✓ CARGADA' if nn_model else '✗ NO DISPONIBLE'}")
print(f"XGBoost:       {'✓ Cargado' if xgb_model else '✗ No disponible'}")
print(f"Random Forest: {'✓ Cargado' if rf_model else '✗ No disponible'}")
print(f"Scaler:        {'✓ Cargado' if scaler else '✗ No disponible'}")
print("="*60)
print()


# ============================================================================
# FUNCIÓN DE PREDICCIÓN
# ============================================================================

def predict_fraud_risk(amount, merchant_risk, hour, day_of_week):
    """
    Predice el riesgo de fraude para una transacción.
    
    Args:
        amount: Monto de la transacción (float)
        merchant_risk: Score de riesgo del comerciante 0-1 (float)
        hour: Hora del día 0-23 (int)
        day_of_week: Día de la semana 0-6 donde 0=Lunes (int)
    
    Returns:
        dict con predicción y metadatos
    """
    
    try:
        # Preparar features como array
        features = np.array([[amount, merchant_risk, hour, day_of_week]], dtype=np.float32)
        
        # Aplicar escalado si el scaler está disponible
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        # Predicción con red neuronal (modelo principal)
        nn_pred = float(nn_model.predict(features_scaled, verbose=0)[0][0])
        
        # Predicciones con modelos adicionales
        predictions = [nn_pred]
        
        if xgb_model is not None:
            try:
                xgb_pred = float(xgb_model.predict_proba(features)[0][1])
                predictions.append(xgb_pred)
            except:
                xgb_pred = None
        else:
            xgb_pred = None
        
        if rf_model is not None:
            try:
                rf_pred = float(rf_model.predict_proba(features)[0][1])
                predictions.append(rf_pred)
            except:
                rf_pred = None
        else:
            rf_pred = None
        
        # Ensemble: promedio de predicciones disponibles
        final_score = float(np.mean(predictions))
        
        # Clasificación por umbral
        if final_score < 0.3:
            risk_level = "BAJO"
            recommendation = "✅ APROBAR"
        elif final_score < 0.7:
            risk_level = "MEDIO"
            recommendation = "⚠️ REVISAR MANUALMENTE"
        else:
            risk_level = "ALTO"
            recommendation = "🚫 BLOQUEAR"
        
        return {
            'score': round(final_score, 4),
            'level': risk_level,
            'recommendation': recommendation,
            'nn_prediction': round(nn_pred, 4),
            'xgb_prediction': round(xgb_pred, 4) if xgb_pred else None,
            'rf_prediction': round(rf_pred, 4) if rf_pred else None,
            'ensemble_size': len(predictions)
        }
        
    except Exception as e:
        print(f"[ERROR] Error en predicción: {e}")
        return {
            'score': 0.5,
            'level': "ERROR",
            'recommendation': "⚠️ ERROR EN PREDICCIÓN",
            'error': str(e)
        }


# ============================================================================
# INTERFAZ GRADIO
# ============================================================================

def create_gradio_interface():
    """Crea y retorna la interfaz Gradio."""
    
    if gr is None:
        print("[ERROR] Gradio no está instalado")
        print("[ERROR] Instala con: pip install gradio")
        return None
    
    def analyze_transaction(amount, merchant_risk, hour, day_of_week):
        """Wrapper para Gradio - formatea la salida."""
        
        result = predict_fraud_risk(amount, merchant_risk, hour, day_of_week)
        
        # Emoji según nivel de riesgo
        emoji_map = {
            "BAJO": "🟢",
            "MEDIO": "🟡",
            "ALTO": "🔴",
            "ERROR": "⚠️"
        }
        
        emoji = emoji_map.get(result['level'], "❓")
        
        # Formatear output
        output = f"## {emoji} Nivel de Riesgo: **{result['level']}**\n\n"
        output += f"**Score de Fraude:** {result['score']:.2%}\n\n"
        output += f"**Recomendación:** {result['recommendation']}\n\n"
        output += "---\n\n"
        output += "### Detalle de Predicciones\n\n"
        output += f"- 🤖 Red Neuronal: {result['nn_prediction']:.2%}\n"
        
        if result['xgb_prediction']:
            output += f"- 🌳 XGBoost: {result['xgb_prediction']:.2%}\n"
        
        if result['rf_prediction']:
            output += f"- 🌲 Random Forest: {result['rf_prediction']:.2%}\n"
        
        output += f"\n*Ensemble de {result['ensemble_size']} modelo(s)*"
        
        return output
    
    # Crear interfaz
    with gr.Blocks(
        title="Risk Ops Console",
        theme=gr.themes.Soft(primary_hue="blue")
    ) as interface:
        
        gr.Markdown("# 🛡️ Risk Ops Console")
        gr.Markdown("### Sistema de Detección de Fraude en Tiempo Real")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 📊 Parámetros de Transacción")
                
                amount = gr.Number(
                    label="💰 Monto ($)",
                    value=150.0,
                    minimum=0,
                    info="Monto de la transacción en dólares"
                )
                
                merchant_risk = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.01,
                    label="🏪 Riesgo del Comerciante",
                    info="Score de reputación: 0=confiable, 1=sospechoso"
                )
                
                hour = gr.Slider(
                    minimum=0,
                    maximum=23,
                    value=14,
                    step=1,
                    label="🕐 Hora del Día",
                    info="Hora en formato 24h (0-23)"
                )
                
                day_of_week = gr.Slider(
                    minimum=0,
                    maximum=6,
                    value=2,
                    step=1,
                    label="📅 Día de la Semana",
                    info="0=Lunes, 6=Domingo"
                )
                
                analyze_btn = gr.Button(
                    "🔍 Analizar Transacción",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("#### 📈 Resultado del Análisis")
                
                output = gr.Markdown(
                    value="*Esperando análisis...*",
                    label="Resultado"
                )
        
        # Ejemplos predefinidos
        gr.Markdown("---")
        gr.Markdown("### 💡 Ejemplos de Prueba")
        
        gr.Examples(
            examples=[
                [50.0, 0.2, 14, 2, "Transacción normal - día laboral"],
                [800.0, 0.9, 3, 5, "Alta sospecha - monto alto, hora inusual"],
                [150.0, 0.5, 10, 0, "Riesgo moderado - valores mixtos"],
                [2000.0, 0.85, 2, 6, "Fraude probable - múltiples banderas rojas"]
            ],
            inputs=[amount, merchant_risk, hour, day_of_week],
            label=None
        )
        
        # Conectar función
        analyze_btn.click(
            fn=analyze_transaction,
            inputs=[amount, merchant_risk, hour, day_of_week],
            outputs=output
        )
    
    return interface


# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("INICIANDO INTERFAZ WEB")
    print("="*60)
    print()
    
    interface = create_gradio_interface()
    
    if interface is None:
        print("[CRITICAL] No se pudo crear la interfaz Gradio")
        sys.exit(1)
    
    print("🚀 Lanzando servidor Gradio...")
    print("📍 La interfaz se abrirá en: http://127.0.0.1:7860")
    print("⚠️ Presiona CTRL+C para detener el servidor")
    print()
    
    try:
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n\n[INFO] Servidor detenido por el usuario")
    except Exception as e:
        print(f"\n[ERROR] Error al lanzar Gradio: {e}")
        sys.exit(1)
