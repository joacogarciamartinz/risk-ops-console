"""
Script de Diagnóstico - Risk Ops Console
=========================================
Valida la integridad de los archivos de modelos serializados.
Ejecutar: python test_models.py
"""

import sys
from pathlib import Path

print("="*70)
print("DIAGNÓSTICO DE MODELOS - Risk Ops Console")
print("="*70)
print()

# ============================================================================
# VERIFICAR DEPENDENCIAS
# ============================================================================

try:
    import joblib
    import numpy as np
    print("✓ Joblib disponible")
except ImportError as e:
    print(f"✗ Error importando joblib: {e}")
    sys.exit(1)

try:
    from tensorflow import keras
    print("✓ TensorFlow/Keras disponible")
except ImportError:
    print("⚠ TensorFlow no disponible (solo se validará el .pkl)")
    keras = None

print()

# ============================================================================
# RUTAS DE ARCHIVOS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
BACKUP_FILE = MODELS_DIR / "risk_ops_backup.pkl"
NN_FILE = MODELS_DIR / "risk_ops_nn.keras"

print(f"📂 Directorio de modelos: {MODELS_DIR}")
print()

# ============================================================================
# TEST 1: VERIFICAR EXISTENCIA DE ARCHIVOS
# ============================================================================

print("[TEST 1] Verificando existencia de archivos...")
print("-" * 70)

if not MODELS_DIR.exists():
    print(f"✗ Carpeta /models no existe: {MODELS_DIR}")
    sys.exit(1)

print(f"✓ Carpeta /models existe")

if BACKUP_FILE.exists():
    size_mb = BACKUP_FILE.stat().st_size / (1024 * 1024)
    print(f"✓ risk_ops_backup.pkl encontrado ({size_mb:.2f} MB)")
else:
    print(f"✗ risk_ops_backup.pkl NO ENCONTRADO")
    print(f"  Ruta esperada: {BACKUP_FILE}")

if NN_FILE.exists():
    size_mb = NN_FILE.stat().st_size / (1024 * 1024)
    print(f"✓ risk_ops_nn.keras encontrado ({size_mb:.2f} MB)")
else:
    print(f"✗ risk_ops_nn.keras NO ENCONTRADO")
    print(f"  Ruta esperada: {NN_FILE}")

print()

# ============================================================================
# TEST 2: VALIDAR CONTENIDO DEL PICKLE
# ============================================================================

if not BACKUP_FILE.exists():
    print("[TEST 2] SALTADO - risk_ops_backup.pkl no existe")
    sys.exit(1)

print("[TEST 2] Validando contenido del pickle...")
print("-" * 70)

try:
    backup = joblib.load(BACKUP_FILE)
    print(f"✓ Pickle cargado correctamente")
    print(f"  Tipo: {type(backup)}")
    
    if not isinstance(backup, dict):
        print(f"✗ ERROR: Se esperaba un dict, se encontró {type(backup)}")
        sys.exit(1)
    
    print(f"\n📋 Llaves disponibles en el pickle:")
    for key in backup.keys():
        print(f"   - {key}")
    
    print()
    
    # Validar llaves requeridas
    required_keys = ['rf_model', 'xgb_model', 'scaler', 'means_normal', 'means_fraud']
    missing_keys = [k for k in required_keys if k not in backup]
    
    if missing_keys:
        print(f"⚠ WARNING: Faltan llaves requeridas:")
        for key in missing_keys:
            print(f"   ✗ {key}")
    else:
        print("✓ Todas las llaves requeridas están presentes")
    
    print()
    
    # Validar modelos
    print("🔍 Validando modelos individuales:")
    
    if 'rf_model' in backup:
        rf = backup['rf_model']
        if rf is not None:
            print(f"   ✓ Random Forest: {type(rf).__name__}")
            if hasattr(rf, 'n_estimators'):
                print(f"     - Árboles: {rf.n_estimators}")
        else:
            print(f"   ✗ Random Forest: None")
    
    if 'xgb_model' in backup:
        xgb = backup['xgb_model']
        if xgb is not None:
            print(f"   ✓ XGBoost: {type(xgb).__name__}")
        else:
            print(f"   ✗ XGBoost: None")
    
    if 'scaler' in backup:
        scaler = backup['scaler']
        if scaler is not None:
            print(f"   ✓ Scaler: {type(scaler).__name__}")
            if hasattr(scaler, 'mean_'):
                print(f"     - Features: {len(scaler.mean_)}")
        else:
            print(f"   ✗ Scaler: None")
    
    print()
    
    # Validar metadatos UI
    print("🎨 Validando metadatos de UI:")
    
    if 'means_normal' in backup:
        means_n = backup['means_normal']
        if means_n is not None and isinstance(means_n, dict):
            print(f"   ✓ means_normal: {len(means_n)} features")
            print(f"     Features: {list(means_n.keys())}")
            print(f"     Valores de ejemplo:")
            for k, v in list(means_n.items())[:3]:
                print(f"       - {k}: {v:.4f}")
        else:
            print(f"   ✗ means_normal: Inválido o None")
    
    if 'means_fraud' in backup:
        means_f = backup['means_fraud']
        if means_f is not None and isinstance(means_f, dict):
            print(f"   ✓ means_fraud: {len(means_f)} features")
            print(f"     Features: {list(means_f.keys())}")
            print(f"     Valores de ejemplo:")
            for k, v in list(means_f.items())[:3]:
                print(f"       - {k}: {v:.4f}")
        else:
            print(f"   ✗ means_fraud: Inválido o None")
    
except Exception as e:
    print(f"✗ ERROR al cargar pickle: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================================================
# TEST 3: VALIDAR RED NEURONAL
# ============================================================================

if not NN_FILE.exists():
    print("[TEST 3] SALTADO - risk_ops_nn.keras no existe")
else:
    print("[TEST 3] Validando red neuronal...")
    print("-" * 70)
    
    if keras is None:
        print("⚠ TensorFlow no disponible - test saltado")
    else:
        try:
            nn_model = keras.models.load_model(NN_FILE)
            print(f"✓ Modelo cargado correctamente")
            print(f"  Tipo: {type(nn_model).__name__}")
            print(f"\n🧠 Arquitectura:")
            print(f"   - Capas: {len(nn_model.layers)}")
            print(f"   - Input shape: {nn_model.input_shape}")
            print(f"   - Output shape: {nn_model.output_shape}")
            
            print(f"\n📊 Resumen de capas:")
            for i, layer in enumerate(nn_model.layers):
               try:
    shape = layer.output_shape
except AttributeError:
    shape = "Multiple/Dynamic"
print(f"   {i+1}. {layer.__class__.__name__} - Output: {shape}")
            
        except Exception as e:
            print(f"✗ ERROR al cargar red neuronal: {e}")
            import traceback
            traceback.print_exc()

print()

# ============================================================================
# TEST 4: SIMULACIÓN DE PREDICCIÓN
# ============================================================================

print("[TEST 4] Simulación de predicción...")
print("-" * 70)

try:
    # Crear features de prueba (30 features típicas)
    test_features = np.zeros(30)
    test_features = test_features.reshape(1, -1)
    
    # Test Random Forest
    if 'rf_model' in backup and backup['rf_model'] is not None:
        try:
            rf_pred = backup['rf_model'].predict_proba(test_features)
            print(f"✓ Random Forest predice correctamente")
            print(f"  Output shape: {rf_pred.shape}")
        except Exception as e:
            print(f"✗ Random Forest falla: {e}")
    
    # Test XGBoost
    if 'xgb_model' in backup and backup['xgb_model'] is not None:
        try:
            xgb_pred = backup['xgb_model'].predict_proba(test_features)
            print(f"✓ XGBoost predice correctamente")
            print(f"  Output shape: {xgb_pred.shape}")
        except Exception as e:
            print(f"✗ XGBoost falla: {e}")
    
    # Test Red Neuronal
    if keras is not None and NN_FILE.exists():
        try:
            nn_pred = nn_model.predict(test_features, verbose=0)
            print(f"✓ Red Neuronal predice correctamente")
            print(f"  Output shape: {nn_pred.shape}")
        except Exception as e:
            print(f"✗ Red Neuronal falla: {e}")
    
except Exception as e:
    print(f"✗ ERROR en simulación: {e}")

print()
print("="*70)
print("DIAGNÓSTICO COMPLETADO")
print("="*70)
print()
print("✅ Si todos los tests pasaron, el sistema está listo para usar")
print("⚠️ Si hay errores, revisa los mensajes arriba para diagnóstico")
