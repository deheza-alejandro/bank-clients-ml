# Decisiones de arquitectura

Registro de las decisiones de diseño que condicionan el notebook y el paquete.
Cada entrada indica el contexto, la decisión adoptada y sus consecuencias.

## ADR-001: Polars como librería de procesamiento

- **Contexto:** el proyecto procesa el historial mensual de clientes con muchas
  columnas y transformaciones agregadas.
- **Decisión:** usar Polars como librería principal de manipulación de datos.
- **Consecuencias:** transformaciones de datos vectorizadas con menos código que
  mantener con respecto a pandas.

## ADR-002: Configuración centralizada con Pydantic Settings

- **Contexto:** las rutas, los nombres de columna y los parámetros varían
  entre entornos y deben ser reproducibles.
- **Decisión:** centralizar la configuración en `Settings` con el prefijo `ML_`
  y un archivo `.env` opcional.
- **Consecuencias:** una sola fuente de verdad; los entornos se ajustan sin
  modificar código.

## ADR-003: Target encoding en lugar de one-hot

- **Contexto:** `Client_Age_grp`, `Region` y `CreditCard_Product` son
  categóricas y one-hot ampliaría la dimensionalidad.
- **Decisión:** codificar cada categoría con su porcentaje de target.
- **Consecuencias:** se captura la relación con el target en una sola columna,
  sin expandir la dimensionalidad.

## ADR-004: Imputación de valores nulos

- **Contexto:** hay nulos en `SavingAccount_Balance_Average`, `Region` y
  `CreditCard_Product`; algunos son vacíos de carga y no cambios reales.
- **Decisión:** completar los nulos con aproximaciones calculadas o con
  información de la ventana de predicción, sin imputar productos inexistentes.
- **Consecuencias:** se preserva el universo; el impacto en el modelo es bajo.

## ADR-005: Fechas relativas en lugar de absolutas

- **Contexto:** las fechas absolutas cambian con el calendario y no son
  comparables entre clientes.
- **Decisión:** reemplazarlas por la diferencia entre el primer y el último
  producto y por la recencia entre el último producto y el mes siguiente al
  último mes de entrenamiento.
- **Consecuencias:** se expresa la misma información en una escala relativa y
  estable en el tiempo; el modelo no depende del calendario.

## ADR-006: Reducción de dimensionalidad en tres etapas

- **Contexto:** después del feature engineering hay columnas que no aportan
  información.
- **Decisión:** eliminar en orden columnas constantes, columnas binarias
  desbalanceadas y columnas numéricas con correlación alta.
- **Consecuencias:** menor costo computacional y menos ruido antes del ranking;
  los descartes se conservan para inspección.

## ADR-007: Ranking de importancia por grupos de negocio

- **Contexto:** es necesario comparar el aporte de cada fuente de negocio y
  seleccionar un conjunto reducido.
- **Decisión:** entrenar un modelo por grupo de columnas y otro con todas; tomar
  las mejores de cada grupo.
- **Consecuencias:** identifica el origen más predictivo y evita que las
  fuentes con más columnas dominen el ranking.

## ADR-008: Binning con codificación por porcentaje de target

- **Contexto:** las variables finales tienen relaciones no lineales con el
  target y se busca una mejor interpretabilidad.
- **Decisión:** agrupar las features relevantes en bines y codificar cada bin
  con su porcentaje de target; las categóricas ya codificadas solo se agrupan
  en bines.
- **Consecuencias:** se captura la no linealidad sin ampliar dimensiones; los
  bines fusionados simplifican la lectura.

## ADR-009: LightGBM sin estandarizar ni balancear

- **Contexto:** los algoritmos basados en árboles no requieren estandarización.
  Y la proporción del target, cercana al 30 %, no está muy desbalanceada.
- **Decisión:** entrenar LightGBM sobre las variables transformadas sin
  estandarizar ni realizar oversampling.
- **Consecuencias:** pipeline más simple.

## ADR-010: Validación cruzada y búsqueda de hiperparámetros

- **Contexto:** el conjunto de datos es pequeño y la selección de
  hiperparámetros puede generar overfitting.
- **Decisión:** usar `RandomizedSearchCV` con `StratifiedKFold`, métrica ROC AUC
  y `n_jobs=1` en el clasificador.
- **Consecuencias:** estimación estratificada y reproducible; el clasificador no
  compite con los procesos de la búsqueda.
