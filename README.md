## 🕵️ Guía para el Analista de Riesgo
## Creado por: Joaquín García Martínez, con Google Gemini como al(IA)do

Al ejecutar `console.py`, el sistema evaluará la transacción basándose en tres capas de defensa:

1. **Capa IA (Probabilidad):** Evalúa patrones abstractos en las variables V1-V28. Un score > 80% indica una anomalía de comportamiento alta.
2. **Capa de Secuencia (Velocity):** Si detectas un "Soft Block", revisa si el mensaje indica "Alta Frecuencia". Esto sugiere un ataque de fuerza bruta o bot.
3. **Capa Financiera:** Antes de decidir, observa la Matriz de Impacto. Si el monto es muy alto ($5,000+), un "Soft Block" es preferible a un "Hard Block" para evitar la pérdida del cliente si el modelo cometió un error (Falso Positivo).

### Interpretación del Gráfico
- **Barras Rojas:** Perfil promedio de un estafador.
- **Barras Verdes:** Perfil promedio de un cliente legítimo.
- **Barra Azul:** Tu transacción actual. Si la barra azul se alinea con las rojas en variables como **V14** o **V17**, la probabilidad de fraude es casi certera.
