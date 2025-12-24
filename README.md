## Creado por: Joaquín García Martínez, con Google Gemini como al(IA)do

## 🛡️ Risk Ops Console: Hybrid Fraud Detection
Este proyecto es una solución integral de Risk Operations que cierra la brecha entre el modelo de Machine Learning y la toma de decisiones humana. El sistema detecta transacciones fraudulentas utilizando un ensamble híbrido de modelos y ofrece una interfaz de consola para que los analistas evalúen riesgos en tiempo real.

## 🚀 Capacidades Principales
Ensamble de Inteligencia Híbrida: Utiliza Redes Neuronales (TensorFlow), XGBoost y Random Forest para una puntuación de riesgo precisa.
Pipeline de Entrenamiento Profesional: Incluye preprocesamiento con balanceo de clases (SMOTE) y exportación de metadatos optimizados.
Consola Operativa: Interfaz interactiva para simular transacciones, evaluar la "velocidad" (frecuencia de intentos) y visualizar diagnósticos comparativos.
Matriz de Impacto Financiero: Calcula automáticamente el ROI y el ahorro preventivo por cada bloqueo de fraude.

## 🛠️ Stack Tecnológico
Categoría     | Herramientas
Lenguaje      | Python 3.x
IA / ML       |TensorFlow, Scikit-Learn, XGBoost, Imbalanced-learn (SMOTE)
Data          | Pandas, Numpy, Joblib
Visualización | Matplotlib, Seaborn 

## 📖 Guía del Usuario: Flujo de Trabajo Operativo
Esta sección detalla cómo utilizar la consola para la gestión diaria de alertas y análisis de riesgo.

## 1. Inicialización del Sistema
Al ejecutar el script principal, el sistema carga automáticamente los modelos pre-entrenados y los escaladores. Verás un mensaje de confirmación indicando que el Ensamble Híbrido está listo para procesar datos.

## 2. Evaluación de Transacciones (Simulación)
Dentro de la consola, puedes ingresar parámetros de transacciones en tiempo real:
Monto y Tiempo: Define el valor de la operación y el desfase temporal.
Análisis de Velocidad: El sistema detectará automáticamente ráfagas de transacciones (frecuencia inusual) que suelen indicar ataques de bots o "carding".

## 3. Interpretación de Resultados
Cada evaluación devuelve un diagnóstico detallado:
Puntaje de Riesgo (0-1): Donde valores cercanos a 1 indican una alta probabilidad de fraude.
Veredicto del Ensamble: Comparativa de los tres modelos. Si hay discrepancia, el sistema prioriza la seguridad basándose en la sensibilidad configurada.
Ahorro Preventivo: Si la transacción es bloqueada, se mostrará el monto total de pérdida evitada.

## 4. Visualización de Métricas de Negocio
Puedes generar reportes rápidos desde la consola para visualizar:
Matriz de Confusión: Para entender la precisión del sistema.
Curva de Precisión-Recall: Crucial para ajustar el umbral de "falsos positivos" y no afectar a clientes legítimos.

## 💡 Nota de Risk Ops:
Recuerda que el análisis de "velocidad" es dinámico. Si un usuario realiza más de 5 intentos en menos de 10 minutos, el sistema elevará el nivel de riesgo independientemente del monto.


## 🛠️ Instalación y Uso Rápido

Este repositorio ya incluye los modelos entrenados en la carpeta `/models`, por lo que no es necesario descargar el dataset original para probar la herramienta.

1. **Clonar el repositorio:**
   git clone [https://github.com/tu-usuario/risk-ops-console.git](https://github.com/tu-usuario/risk-ops-console.git)
   cd risk-ops-console

2. **Crear entorno virtual (Recomendado Python 3.10 o 3.11):**
python -m venv venv
.\venv\Scripts\activate   # En Mac: source venv/bin/activate

3. **Instalar dependencias:**
pip install -r requirements.txt

4. **Lanzar la Consola Visual:**
python dashboard.py
