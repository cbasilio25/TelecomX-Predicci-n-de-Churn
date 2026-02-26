# 📊 TelecomX — Predicción de Churn (Proyecto End-to-End de Machine Learning)

## 🚀 Problema de Negocio

La cancelación de clientes (churn) representa una de las principales amenazas para empresas basadas en suscripción.

TelecomX necesita identificar de forma anticipada qué clientes tienen mayor probabilidad de cancelar su servicio para implementar estrategias de retención focalizadas y reducir pérdidas de ingresos.

Este proyecto desarrolla un pipeline completo de Machine Learning para predecir churn y generar insights accionables para negocio.

---

# 🎯 Objetivos del Proyecto

- Construir un modelo predictivo robusto para churn.
- Manejar el desbalance de clases.
- Optimizar el rendimiento mediante validación cruzada.
- Interpretar el modelo con técnicas explicativas (SHAP).
- Traducir resultados técnicos en recomendaciones estratégicas.
- Entregar un modelo listo para producción.

---

# 📁 Estructura del Proyecto
├── CHURN_TELECOM_X_PARTE_2.ipynb
├── modelo_final_churn.pkl
├── TelecomX_diccionario.md
├── TelecomX_diccionario
└── README.md
# 📊 Descripción del Dataset

El dataset contiene información sobre:

- Datos demográficos (género, adultos mayores, dependientes)
- Antigüedad del cliente (tenure)
- Tipo de contrato
- Método de pago
- Servicios adicionales (seguridad online, soporte técnico, streaming)
- Cargos mensuales y totales
- Variable objetivo: `Churn`

Problema de clasificación binaria:
- 1 → Cliente cancela
- 0 → Cliente permanece

---

# 🧠 Metodología

## 1️⃣ Preprocesamiento

- Eliminación de identificadores
- Conversión de variables numéricas
- Codificación One-Hot para variables categóricas
- Escalado de variables numéricas
- División estratificada Train/Test (70/30)
- Análisis de desbalance de clases

---

## 2️⃣ Análisis Exploratorio (EDA)

### Hallazgos Clave

- La antigüedad (`tenure`) tiene fuerte correlación negativa con churn.
- Los contratos mensuales presentan mayor tasa de cancelación.
- Cargos mensuales elevados incrementan probabilidad de churn.
- Servicios adicionales reducen riesgo de cancelación.

### Insight Estratégico

El mayor riesgo se concentra en clientes nuevos con contratos mensuales y cargos elevados.

---

## 3️⃣ Modelado

Modelos evaluados:

- Regresión Logística (baseline)
- Random Forest
- Regresión Logística + SMOTE
- Random Forest optimizado con GridSearchCV (modelo final)

Métricas utilizadas:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC (métrica principal de optimización)

---

# 🏆 Modelo Final

### Random Forest optimizado

Optimización realizada mediante GridSearchCV:

- n_estimators
- max_depth
- Validación cruzada de 5 folds

Criterio de selección:
Mayor ROC-AUC con equilibrio entre precision y recall.

Modelo exportado como: modelo_final_churn.pkl

Incluye pipeline completo (preprocesamiento + modelo entrenado).

---

# 📈 Resultados del Modelo

- Mejor desempeño que el baseline.
- Buen balance entre detección de churn (recall) y control de falsos positivos.
- Capacidad de capturar relaciones no lineales.
- Robusto ante variables categóricas múltiples.

SMOTE mejoró recall pero aumentó falsos positivos.
Se seleccionó el modelo considerando el trade-off negocio.

---

# 🔎 Interpretabilidad (SHAP)

Se utilizó SHAP para:

- Analizar impacto global de variables
- Entender dirección del efecto
- Explicar predicciones individuales

### Variables más influyentes:

1. Antigüedad (tenure)
2. Cargos mensuales
3. Tipo de contrato
4. Seguridad online
5. Soporte técnico
6. Método de pago

### Interpretación

- Baja antigüedad → mayor probabilidad de churn.
- Contratos mensuales → mayor riesgo.
- Servicios adicionales → reducen cancelación.
- Cargos elevados → aumentan riesgo.

---

# 💡 Recomendaciones de Negocio

- Incentivar contratos anuales o bianuales.
- Diseñar campañas de retención en los primeros 6 meses.
- Ofrecer paquetes combinados (seguridad + soporte).
- Monitorear clientes con cargos altos.
- Promover métodos de pago automáticos.

---

# ⚙️ Tecnologías Utilizadas

- Python
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- SHAP
- Matplotlib / Seaborn
- Joblib

---

# 📦 Uso del Modelo

```python
import joblib

modelo = joblib.load("modelo_final_churn.pkl")

predicciones = modelo.predict(nuevos_datos)
probabilidades = modelo.predict_proba(nuevos_datos)

👤 Autor

Cesar Basilio

Data Science | Machine Learning | Analítica Predictiva
