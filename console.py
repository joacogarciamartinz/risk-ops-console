import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE CPU ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['OMP_NUM_THREADS'] = '1'

# 1. CARGA DE CEREBROS
# Asegúrate de haber creado la carpeta 'models' y metido los archivos dentro
try:
    print("📂 Cargando inteligencia de riesgo desde carpeta models/...")
    pack = joblib.load('models/risk_metadata.pkl')
    nn_model = tf.keras.models.load_model('models/risk_nn_model.keras')

    rf = pack['rf']
    scaler = pack['scaler']
    top_features = pack['top_features']
    means_normal = pack['means_normal']
    means_fraud = pack['means_fraud']
    print("✅ Sistema cargado exitosamente.\n")
except Exception as e:
    print(f"❌ ERROR: No se pudieron cargar los modelos.")
    print(f"Asegúrate de que existan: 'models/risk_metadata.pkl' y 'models/risk_nn_model.keras'")
    exit()

# 2. FUNCIONES OPERATIVAS (Aquí están pegadas las funciones reales)
def analyze_velocity(current_time, current_amount, history):
    """Analiza la frecuencia de transacciones en los últimos 10 minutos."""
    recent = [tx for tx in history if (current_time - tx['time']) <= 600]
    risk = 50 if len(recent) >= 2 else 0
    if current_amount < 5 and any(tx['amount'] < 5 for tx in recent):
        risk += 40
    return risk

def print_financial_impact(amount, is_block):
    """Muestra el impacto económico de la decisión."""
    fee = amount * 0.02
    if is_block:
        print(f"💰 IMPACTO: BLOQUEO PREVENTIVO. Evitas posible pérdida de ${amount:.2f}")
    else:
        print(f"💰 IMPACTO: APROBADO. Ganancia estimada (Fee): ${fee:.2f}")

def plot_diagnostics(user_inputs):
    """Genera el gráfico de barras comparativo."""
    x = np.arange(len(top_features))
    u_vals = [user_inputs[f] for f in top_features]
    n_vals = [means_normal[f] for f in top_features]
    f_vals = [means_fraud[f] for f in top_features]
    
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, n_vals, 0.2, label='Patrón Normal', color='green', alpha=0.5)
    plt.bar(x, f_vals, 0.2, label='Patrón Fraude', color='red', alpha=0.5)
    plt.bar(x + 0.2, u_vals, 0.2, label='Tu Simulación', color='blue')
    plt.title('Diagnóstico Visual: Usuario vs Patrones Históricos')
    plt.xticks(x, top_features)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()

# 3. LÓGICA DE SIMULACIÓN
def run_simulation(history):
    print("\n" + "="*60)
    print("🛡️  NUEVA TRANSACCIÓN - RISK OPS CONSOLE")
    print("="*60)
    
    try:
        amount = float(input("💳 Monto ($): "))
        time = float(input("⏱️  Tiempo (seg): "))
        
        user_vals = {}
        print(f"\n--- AJUSTE DE VECTORES CRÍTICOS ---")
        for f in top_features:
            print(f"   {f} [Normal: {means_normal[f]:.2f} | Fraude: {means_fraud[f]:.2f}]")
            user_vals[f] = float(input(f"   > Valor para {f}: "))
        
        # Construcción del vector completo (Relleno con promedio normal)
        vector = [time] + [means_normal.get(f'V{i}', 0) for i in range(1, 29)] + [amount]
        cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        for i, col in enumerate(cols):
            if col in user_vals: vector[i] = user_vals[col]
        
        # Predicción
        scaled = scaler.transform([vector])
        ml_prob = nn_model.predict(scaled, verbose=0)[0][0] * 100
        v_risk = analyze_velocity(time, amount, history)
        
        total_risk = max(ml_prob, v_risk)
        
        print("\n" + "—"*60)
        print(f"🔍 RIESGO DETECTADO: {total_risk:.2f}%")
        
        if total_risk > 85:
            print("🟥 DECISIÓN: HARD BLOCK (Rechazo Total)")
            is_block = True
        elif total_risk > 45:
            print("🟨 DECISIÓN: SOFT BLOCK (Fricción / SMS)")
            is_block = True
        else:
            print("🟩 DECISIÓN: APPROVE (Aprobación Libre)")
            is_block = False
        
        print_financial_impact(amount, is_block)
        history.append({'time': time, 'amount': amount})
        plot_diagnostics(user_vals)

    except ValueError:
        print("❌ Error: Ingresa solo números.")

if __name__ == "__main__":
    session_history = []
    while True:
        run_simulation(session_history)
        if input("\n¿Analizar otro caso? (s/n): ").lower() != 's':
            print("Cerrando consola...")
            break