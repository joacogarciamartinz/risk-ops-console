"""
Risk Ops Console - Interfaz de Detección de Fraude
====================================================
Versión: 1.0-Hybrid
Arquitectura: Ensemble de 3 modelos (RF + XGB + Deep Learning)
Dataset: Credit Card Fraud Detection (Kaggle) - Desbalanceado
Estrategia: SMOTE en entrenamiento + Consenso en inferencia
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
print(f"📂 Directorio base: {BASE_DIR}")
print(f"📂 Directorio de modelos: {MODELS_DIR}")
print()

# ============================================================================
# NOMBRES DE ARCHIVOS - SINCRONIZADOS CON MAIN.PY
# ============================================================================

# El archivo .pkl contiene TODO el stack tradicional:
# - Random Forest
# - XGBoost
# - StandardScaler
# - means_normal / means_fraud (metadatos para UI)
BACKUP_FILE = "risk_ops_backup.pkl"

# El archivo .keras contiene la red neuronal profunda
NN_MODEL_FILE = "risk_ops_nn.keras"

# ============================================================================
# VALIDACIÓN DE DEPENDENCIAS
# ============================================================================

print("[INIT] Verificando dependencias críticas...")

dependencies_ok = True

try:
    import joblib
    import numpy as np
    import pandas as pd
    print("  ✓ NumPy, Pandas, Joblib")
except ImportError as e:
    print(f"  ✗ Error: {e}")
    dependencies_ok = False

try:
    from tensorflow import keras
    import tensorflow as tf
    print(f"  ✓ TensorFlow {tf.__version__}")
    
    if tf.__version__ < "2.15.0":
        print(f"  ⚠ WARNING: TensorFlow {tf.__version__} < 2.15.0 (puede fallar con .keras)")
except ImportError:
    print("  ✗ TensorFlow no disponible")
    print("     Instala con: pip install tensorflow>=2.15.0")
    keras = None
    dependencies_ok = False

try:
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    print("  ✓ Scikit-learn, XGBoost")
except ImportError as e:
    print(f"  ⚠ Importación opcional falló: {e}")

try:
    import gradio as gr
    print(f"  ✓ Gradio {gr.__version__}")
except ImportError:
    print("  ✗ Gradio no disponible")
    print("     Instala con: pip install gradio")
    gr = None
    dependencies_ok = False

if not dependencies_ok:
    print("\n[CRITICAL] Faltan dependencias críticas")
    print("Ejecuta: pip install -r requirements.txt")
    sys.exit(1)

print()

# ============================================================================
# CARGA DE MODELOS - ESTRATEGIA HÍBRIDA
# ============================================================================

def load_hybrid_models():
    """
    Carga el stack completo de modelos desde los archivos serializados.
    
    Arquitectura:
        risk_ops_backup.pkl:
            - 'rf_model': Random Forest Classifier
            - 'xgb_model': XGBoost Classifier
            - 'scaler': StandardScaler (fitted)
            - 'means_normal': dict con medias de transacciones legítimas
            - 'means_fraud': dict con medias de transacciones fraudulentas
        
        risk_ops_nn.keras:
            - Red Neuronal Densa (Keras Sequential)
    
    Returns:
        dict con todos los componentes cargados
    """
    
    models = {}
    
    # ========================================================================
    # PASO 1: Cargar el backup completo (modelos tradicionales + metadatos)
    # ========================================================================
    
    backup_path = MODELS_DIR / BACKUP_FILE
    
    print(f"[LOAD] Cargando backup híbrido: {backup_path}")
    
    if not backup_path.exists():
        print(f"[ERROR] Archivo no encontrado: {backup_path}")
        print(f"[INFO] Contenido de {MODELS_DIR}:")
        if MODELS_DIR.exists():
            for file in MODELS_DIR.iterdir():
                print(f"  - {file.name}")
        else:
            print("  (directorio no existe)")
        raise FileNotFoundError(f"No se encontró {BACKUP_FILE}")
    
    try:
        backup_data = joblib.load(backup_path)
        print(f"[OK] Backup cargado - Llaves disponibles: {list(backup_data.keys())}")
        
        # Extraer componentes individuales
        models['rf_model'] = backup_data.get('rf_model')
        models['xgb_model'] = backup_data.get('xgb_model')
        models['scaler'] = backup_data.get('scaler')
        models['means_normal'] = backup_data.get('means_normal')
        models['means_fraud'] = backup_data.get('means_fraud')
        
        # Validar que los modelos críticos estén presentes
        if models['rf_model'] is None:
            print("[WARNING] Random Forest no encontrado en backup")
        else:
            print(f"  ✓ Random Forest cargado")
        
        if models['xgb_model'] is None:
            print("[WARNING] XGBoost no encontrado en backup")
        else:
            print(f"  ✓ XGBoost cargado")
        
        if models['scaler'] is None:
            print("[WARNING] Scaler no encontrado en backup")
        else:
            print(f"  ✓ StandardScaler cargado")
        
        # Validar metadatos para UI
        if models['means_normal'] is None or models['means_fraud'] is None:
            print("[WARNING] Metadatos de UI incompletos - usando valores por defecto")
            models['means_normal'] = {
                'V14': -1.0, 'V10': -0.5, 'V12': -0.3, 
                'V17': -0.2, 'V11': 0.5
            }
            models['means_fraud'] = {
                'V14': -5.0, 'V10': -3.0, 'V12': -2.0,
                'V17': -1.5, 'V11': 2.0
            }
        else:
            print(f"  ✓ Metadatos UI cargados")
            print(f"    - Features en means_normal: {list(models['means_normal'].keys())}")
        
    except Exception as e:
        print(f"[ERROR] Error al cargar backup: {e}")
        raise
    
    # ========================================================================
    # PASO 2: Cargar red neuronal profunda
    # ========================================================================
    
    nn_path = MODELS_DIR / NN_MODEL_FILE
    
    print(f"\n[LOAD] Cargando red neuronal: {nn_path}")
    
    if not nn_path.exists():
        print(f"[ERROR] Modelo no encontrado: {nn_path}")
        raise FileNotFoundError(f"No se encontró {NN_MODEL_FILE}")
    
    try:
        models['nn_model'] = keras.models.load_model(nn_path)
        print(f"[OK] Red Neuronal cargada")
        print(f"  - Arquitectura: {len(models['nn_model'].layers)} capas")
        print(f"  - Input shape: {models['nn_model'].input_shape}")
        
    except Exception as e:
        print(f"[ERROR] Error al cargar red neuronal: {e}")
        raise
    
    return models


# ============================================================================
# INICIALIZACIÓN DEL SISTEMA
# ============================================================================

print("[INIT] Cargando sistema de detección de fraude...")
print()

try:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS = load_hybrid_models()
    
    # Extraer componentes para acceso global
    rf_model = MODELS['rf_model']
    xgb_model = MODELS['xgb_model']
    nn_model = MODELS['nn_model']
    scaler = MODELS['scaler']
    means_normal = MODELS['means_normal']
    means_fraud = MODELS['means_fraud']
    
    print()
    print("="*70)
    print("RESUMEN DE CARGA DEL SISTEMA")
    print("="*70)
    print(f"Random Forest:    {'✓ Operacional' if rf_model else '✗ No disponible'}")
    print(f"XGBoost:          {'✓ Operacional' if xgb_model else '✗ No disponible'}")
    print(f"Red Neuronal:     {'✓ Operacional' if nn_model else '✗ No disponible'}")
    print(f"Scaler:           {'✓ Operacional' if scaler else '✗ No disponible'}")
    print(f"Metadatos UI:     {'✓ Disponibles' if means_normal else '✗ No disponibles'}")
    print("="*70)
    print()
    
    # Validar que al menos tengamos los 3 modelos principales
    if not all([rf_model, xgb_model, nn_model]):
        print("[CRITICAL] Falta uno o más modelos del ensemble")
        print("El sistema requiere RF + XGB + NN para funcionar")
        sys.exit(1)
    
    if scaler is None:
        print("[CRITICAL] StandardScaler no disponible")
        print("El sistema requiere el scaler para normalizar inputs")
        sys.exit(1)
    
    print("[OK] Sistema inicializado correctamente")
    print()

except Exception as e:
    print(f"\n[CRITICAL] Error fatal en la inicialización: {e}")
    print("\nAsegúrate de que:")
    print("  1. Ejecutaste main.py para generar los modelos")
    print("  2. Los archivos existen en /models:")
    print(f"     - {BACKUP_FILE}")
    print(f"     - {NN_MODEL_FILE}")
    sys.exit(1)


# ============================================================================
# FUNCIÓN DE PREDICCIÓN - ENSEMBLE HÍBRIDO
# ============================================================================

def predict_fraud_ensemble(features_dict):
    """
    Realiza predicción usando el ensemble de 3 modelos.
    
    Arquitectura de Inferencia:
        1. Normaliza features con StandardScaler
        2. Obtiene probabilidad de cada modelo:
           - Random Forest
           - XGBoost
           - Deep Learning
        3. Calcula promedio ponderado (consenso)
        4. Clasifica según umbral (>0.5 = Fraude)
    
    Args:
        features_dict: dict con las features V1-V28, Amount, Time
    
    Returns:
        dict con score, predicciones individuales y recomendación
    """
    
    try:
        # Preparar array de features (debe coincidir con el orden de entrenamiento)
        # Típicamente: [V1, V2, ..., V28, Amount, Time]
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
        feature_vector = np.array([[features_dict.get(f, 0.0) for f in feature_names]])
        
        # Normalizar con el scaler entrenado
        features_scaled = scaler.transform(feature_vector)
        
        # ====================================================================
        # PREDICCIONES INDIVIDUALES
        # ====================================================================
        
        # Random Forest: predict_proba retorna [prob_clase_0, prob_clase_1]
        rf_prob = float(rf_model.predict_proba(features_scaled)[0][1])
        
        # XGBoost: mismo formato
        xgb_prob = float(xgb_model.predict_proba(features_scaled)[0][1])
        
        # Red Neuronal: output directo (sigmoid en última capa)
        nn_prob = float(nn_model.predict(features_scaled, verbose=0)[0][0])
        
        # ====================================================================
        # ENSEMBLE: Promedio Ponderado
        # ====================================================================
        # Puedes ajustar estos pesos según performance en validación
        
        weights = {
            'rf': 0.30,   # Random Forest: bueno en interpretabilidad
            'xgb': 0.35,  # XGBoost: mejor en datos tabulares
            'nn': 0.35    # Deep Learning: captura no-linealidades
        }
        
        ensemble_score = (
            weights['rf'] * rf_prob +
            weights['xgb'] * xgb_prob +
            weights['nn'] * nn_prob
        )
        
        # ====================================================================
        # CLASIFICACIÓN Y RECOMENDACIÓN
        # ====================================================================
        
        is_fraud = ensemble_score > 0.5
        
        if ensemble_score >= 0.8:
            risk_level = "CRÍTICO"
            recommendation = "🚫 BLOQUEAR TRANSACCIÓN INMEDIATAMENTE"
            color = "red"
        elif ensemble_score >= 0.5:
            risk_level = "ALTO"
            recommendation = "⚠️ REVISAR MANUALMENTE - Posible Fraude"
            color = "orange"
        elif ensemble_score >= 0.3:
            risk_level = "MEDIO"
            recommendation = "⚡ MONITOREAR - Actividad sospechosa"
            color = "yellow"
        else:
            risk_level = "BAJO"
            recommendation = "✅ APROBAR - Transacción legítima"
            color = "green"
        
        return {
            'ensemble_score': round(ensemble_score, 4),
            'is_fraud': is_fraud,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'color': color,
            'individual_predictions': {
                'random_forest': round(rf_prob, 4),
                'xgboost': round(xgb_prob, 4),
                'neural_network': round(nn_prob, 4)
            },
            'consensus': {
                'agree_fraud': sum([rf_prob > 0.5, xgb_prob > 0.5, nn_prob > 0.5]),
                'agree_legit': sum([rf_prob <= 0.5, xgb_prob <= 0.5, nn_prob <= 0.5])
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Error en predicción: {e}")
        return {
            'ensemble_score': 0.5,
            'is_fraud': None,
            'risk_level': "ERROR",
            'recommendation': f"❌ Error: {str(e)}",
            'color': "gray",
            'error': str(e)
        }


# ============================================================================
# INTERFAZ GRADIO - UI PARA ANALISTA DE RIESGO
# ============================================================================

def create_gradio_interface():
    """
    Crea la interfaz web interactiva para analistas de riesgo.
    
    Features:
        - Sliders para Top 5 features más importantes (V14, V10, V12, V17, V11)
        - Input de Amount y Time
        - Visualización en tiempo real del ensemble score
        - Breakdown de predicciones individuales
    """
    
    if gr is None:
        print("[ERROR] Gradio no está disponible")
        return None
    
    def analyze_transaction(v14, v10, v12, v17, v11, amount, time):
        """
        Wrapper para Gradio - maneja inputs de la UI y formatea output.
        """
        
        # Construir diccionario de features
        # Las features no ajustables se inicializan en 0 (valores normalizados)
        features = {f'V{i}': 0.0 for i in range(1, 29)}
        
        # Actualizar con valores ajustables del UI
        features['V14'] = v14
        features['V10'] = v10
        features['V12'] = v12
        features['V17'] = v17
        features['V11'] = v11
        features['Amount'] = amount
        features['Time'] = time
        
        # Ejecutar predicción
        result = predict_fraud_ensemble(features)
        
        # Formatear salida para Gradio
        emoji_map = {
            "CRÍTICO": "🔴",
            "ALTO": "🟠",
            "MEDIO": "🟡",
            "BAJO": "🟢",
            "ERROR": "⚠️"
        }
        
        emoji = emoji_map.get(result['risk_level'], "❓")
        
        output = f"## {emoji} Nivel de Riesgo: **{result['risk_level']}**\n\n"
        output += f"### Score de Fraude del Ensemble: **{result['ensemble_score']:.2%}**\n\n"
        output += f"{result['recommendation']}\n\n"
        output += "---\n\n"
        output += "### 📊 Predicciones Individuales\n\n"
        
        ind = result['individual_predictions']
        output += f"- 🌳 **Random Forest:** {ind['random_forest']:.2%}\n"
        output += f"- 🚀 **XGBoost:** {ind['xgboost']:.2%}\n"
        output += f"- 🧠 **Red Neuronal:** {ind['neural_network']:.2%}\n\n"
        
        cons = result['consensus']
        output += f"**Consenso:** {cons['agree_fraud']}/3 modelos detectan fraude\n"
        
        return output
    
    # ========================================================================
    # DISEÑO DE LA INTERFAZ
    # ========================================================================
    
    with gr.Blocks(
        title="Risk Ops Console",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="red")
    ) as interface:
        
        gr.Markdown("# 🛡️ Risk Ops Console")
        gr.Markdown("### Sistema Híbrido de Detección de Fraude")
        gr.Markdown("*Ensemble de Random Forest + XGBoost + Deep Learning*")
        
        gr.Markdown("---")
        
        with gr.Row():
            # ================================================================
            # PANEL IZQUIERDO: Controles de Features
            # ================================================================
            with gr.Column(scale=1):
                gr.Markdown("#### 🎛️ Top Features (Ajustables)")
                gr.Markdown("*Basado en importancia del modelo*")
                
                # Pre-poblar con valores de transacción normal
                v14_input = gr.Slider(
                    minimum=-10, maximum=5, 
                    value=means_normal.get('V14', -1.0),
                    step=0.1,
                    label="V14 (Feature más importante)",
                    info="Valores negativos típicos"
                )
                
                v10_input = gr.Slider(
                    minimum=-10, maximum=5,
                    value=means_normal.get('V10', -0.5),
                    step=0.1,
                    label="V10"
                )
                
                v12_input = gr.Slider(
                    minimum=-10, maximum=5,
                    value=means_normal.get('V12', -0.3),
                    step=0.1,
                    label="V12"
                )
                
                v17_input = gr.Slider(
                    minimum=-10, maximum=5,
                    value=means_normal.get('V17', -0.2),
                    step=0.1,
                    label="V17"
                )
                
                v11_input = gr.Slider(
                    minimum=-5, maximum=5,
                    value=means_normal.get('V11', 0.5),
                    step=0.1,
                    label="V11"
                )
                
                gr.Markdown("#### 💰 Detalles de Transacción")
                
                amount_input = gr.Number(
                    value=100.0,
                    label="Monto ($)",
                    info="Monto de la transacción"
                )
                
                time_input = gr.Number(
                    value=0,
                    label="Time (segundos desde primera TX)",
                    info="Timestamp relativo"
                )
                
                analyze_btn = gr.Button(
                    "🔍 ANALIZAR TRANSACCIÓN",
                    variant="primary",
                    size="lg"
                )
            
            # ================================================================
            # PANEL DERECHO: Resultados
            # ================================================================
            with gr.Column(scale=1):
                gr.Markdown("#### 📈 Resultado del Análisis")
                
                output_markdown = gr.Markdown(
                    value="*Esperando análisis...*"
                )
        
        # ====================================================================
        # EJEMPLOS PREDEFINIDOS
        # ====================================================================
        
        gr.Markdown("---")
        gr.Markdown("### 💡 Casos de Prueba")
        
        gr.Examples(
            examples=[
                # [V14, V10, V12, V17, V11, Amount, Time, Descripción]
                [
                    means_normal.get('V14', -1.0),
                    means_normal.get('V10', -0.5),
                    means_normal.get('V12', -0.3),
                    means_normal.get('V17', -0.2),
                    means_normal.get('V11', 0.5),
                    50.0, 1000,
                    "Transacción Normal"
                ],
                [
                    means_fraud.get('V14', -5.0),
                    means_fraud.get('V10', -3.0),
                    means_fraud.get('V12', -2.0),
                    means_fraud.get('V17', -1.5),
                    means_fraud.get('V11', 2.0),
                    1500.0, 5000,
                    "Fraude Típico"
                ],
                [
                    -2.5, -1.5, -1.0, -0.8, 1.2,
                    300.0, 2500,
                    "Caso Ambiguo"
                ]
            ],
            inputs=[v14_input, v10_input, v12_input, v17_input, v11_input, 
                   amount_input, time_input],
            label=None
        )
        
        # Conectar botón de análisis
        analyze_btn.click(
            fn=analyze_transaction,
            inputs=[v14_input, v10_input, v12_input, v17_input, v11_input,
                   amount_input, time_input],
            outputs=output_markdown
        )
    
    return interface


# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("LANZANDO INTERFAZ WEB")
    print("="*70)
    print()
    
    interface = create_gradio_interface()
    
    if interface is None:
        print("[CRITICAL] No se pudo crear la interfaz")
        sys.exit(1)
    
    print("🚀 Servidor Gradio iniciando...")
    print("📍 URL: http://127.0.0.1:7860")
    print("⚠️  Presiona CTRL+C para detener")
    print()
    
    try:
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )
    except KeyboardInterrupt:
        print("\n\n[INFO] Servidor detenido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Error al lanzar servidor: {e}")
        sys.exit(1)
