# Roadmap del Bot de Scalping Apollo

Este documento detalla el plan de desarrollo para el bot de scalping 'Apollo', que operará en Binance utilizando modelos de Machine Learning para predecir movimientos de precios de ETH.

## Fase 1: Recolección y Análisis de Datos

### Tareas:

- [ ] **Script de Recolección de Datos:**
  - [ ] Conectar a la API de Binance para obtener datos históricos de precios (velas/candlesticks) de ETH.
  - [ ] Definir granularidades de tiempo para la recolección (ej. 1m, 5m, 15m).
  - [ ] Almacenar datos de manera eficiente (CSV, SQLite, Parquet).
  - [ ] Considerar la recolección de otros datos relevantes (volumen, indicadores técnicos).

- [ ] **Análisis Exploratorio de Datos (EDA):**
  - [ ] Analizar el comportamiento del precio de ETH.
  - [ ] Identificar patrones, tendencias y volatilidad.
  - [ ] Determinar porcentajes de subida (X%) y marcos de tiempo (Y%) realistas para la predicción, considerando comisiones.
  - [ ] Visualizar los datos para obtener insights.

## Fase 2: Desarrollo de Modelos de Machine Learning

### Tareas:

- [ ] **Preprocesamiento de Datos:**
  - [ ] Preparar datos para el entrenamiento (normalización, creación de características).
  - [ ] Definir etiquetas (targets) para los modelos de predicción.

- [ ] **Selección y Entrenamiento de Modelos:**
  - [ ] Investigar y seleccionar algoritmos de ML adecuados (ej. LSTM, Random Forest, Gradient Boosting).
  - [ ] Entrenar tres modelos distintos para diferentes probabilidades de subida (ej. 1% en 10m, 2% en 30m, 3% en 60m).
  - [ ] Evaluar el rendimiento de los modelos (precisión, recall, F1-score, AUC).

- [ ] **Optimización y Validación:**
  - [ ] Ajustar hiperparámetros para mejorar el rendimiento.
  - [ ] Validar modelos con datos históricos no vistos.

## Fase 3: Implementación de la Estrategia de Trading (Bot Apollo)

### Tareas:

- [ ] **Conexión con la API de Binance:**
  - [ ] Configurar conexión segura para operaciones de trading.

- [ ] **Lógica de Compra:**
  - [ ] Integrar modelos de ML para decisiones de compra.
  - [ ] Definir umbrales de probabilidad para activar compras.

- [ ] **Lógica de Venta:**
  - [ ] Calcular precio mínimo de venta (precio de compra + 0.16%).
  - [ ] Implementar Stop Loss al 1% por debajo del precio de compra.
  - [ ] Implementar Trailing Stop:
    - [ ] Activar cuando el precio supere el precio de compra en un 0.16%.
    - [ ] Monitorear precio máximo registrado.
    - [ ] Vender cuando el precio caiga un 0.2% por debajo del máximo, siempre que sea mayor al precio mínimo de venta.
    - [ ] *Pendiente: Filtro para picos de venta que causan caídas temporales.*

- [ ] **Gestión de Órdenes y Errores:**
  - [ ] Implementar sistema robusto para gestión de órdenes.
  - [ ] Manejar errores de la API y excepciones.

- [ ] **Gestión de Riesgos y Capital:**
  - [ ] Definir tamaño de posiciones y gestión del capital.

## Fase 4: Backtesting y Optimización

### Tareas:

- [ ] **Backtesting:**
  - [ ] Simular la estrategia completa con datos históricos.
  - [ ] Analizar métricas de rentabilidad y riesgo.

- [ ] **Optimización de Parámetros:**
  - [ ] Ajustar parámetros de la estrategia basándose en resultados del backtesting.

## Fase 5: Despliegue y Monitoreo

### Tareas:

- [ ] **Despliegue:**
  - [ ] Desplegar el bot en un entorno de ejecución confiable.

- [ ] **Monitoreo y Alertas:**
  - [ ] Implementar sistema de monitoreo en tiempo real.
  - [ ] Configurar alertas para eventos importantes.

- [ ] **Mantenimiento y Actualizaciones:**
  - [ ] Realizar mantenimiento regular y actualizaciones.

## Fase 6: Documentación y Entrega Final

### Tareas:

- [ ] **Documentación:**
  - [ ] Crear documentación técnica del bot.
  - [ ] Elaborar guía de usuario.

- [ ] **Entrega Final:**
  - [ ] Presentar el bot y la documentación al usuario.


