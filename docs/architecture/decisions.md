# Decisiones de arquitectura

Registro de las decisiones que condicionan el pipeline, las dependencias de
desarrollo y el entorno de trabajo. Cada entrada indica el contexto, la
decisión y sus consecuencias; el contexto recoge las alternativas evaluadas y
las consecuencias los trade-offs.

## Decisiones del pipeline

### ADR-001: Polars como librería de procesamiento

- **Contexto:** el proyecto procesa el historial mensual de clientes con muchas
  columnas y transformaciones agregadas.
- **Decisión:** usar Polars como librería principal de manipulación de datos.
- **Consecuencias:** transformaciones de datos vectorizadas con menos código que
  mantener con respecto a pandas.

### ADR-002: Configuración centralizada con Pydantic Settings

- **Contexto:** las rutas, los nombres de columna y los parámetros varían
  entre entornos y deben ser reproducibles.
- **Decisión:** centralizar la configuración en `Settings` con el prefijo `ML_`
  y un archivo `.env` opcional.
- **Consecuencias:** una sola fuente de verdad; los entornos se ajustan sin
  modificar código.

### ADR-003: Target encoding en lugar de one-hot

- **Contexto:** `Client_Age_grp`, `Region` y `CreditCard_Product` son
  categóricas y one-hot ampliaría la dimensionalidad.
- **Decisión:** codificar cada categoría con su porcentaje de target.
- **Consecuencias:** se captura la relación con el target en una sola columna,
  sin expandir la dimensionalidad.

### ADR-004: Imputación de valores nulos

- **Contexto:** hay nulos en `SavingAccount_Balance_Average`, `Region` y
  `CreditCard_Product`; algunos son vacíos de carga y no cambios reales.
- **Decisión:** completar los nulos con aproximaciones calculadas o con
  información de la ventana de predicción, sin imputar productos inexistentes.
- **Consecuencias:** se preserva el universo; el impacto en el modelo es bajo.

### ADR-005: Fechas relativas en lugar de absolutas

- **Contexto:** las fechas absolutas cambian con el calendario y no son
  comparables entre clientes.
- **Decisión:** reemplazarlas por la diferencia entre el primer y el último
  producto y por la recencia entre el último producto y el mes siguiente al
  último mes de entrenamiento.
- **Consecuencias:** se expresa la misma información en una escala relativa y
  estable en el tiempo; el modelo no depende del calendario.

### ADR-006: Reducción de dimensionalidad en tres etapas

- **Contexto:** después del feature engineering hay columnas que no aportan
  información.
- **Decisión:** aplicar tres filtros sucesivos: columnas constantes, columnas
  binarias desbalanceadas con una clase por debajo del 10 % y columnas numéricas
  con correlación absoluta superior a 0,80.
- **Consecuencias:** el orden importa porque cada filtro reduce el conjunto del
  siguiente; se obtiene menor costo computacional y menos ruido antes del
  ranking, y los descartes se conservan para inspección.

### ADR-007: Ranking de importancia por grupos de negocio

- **Contexto:** es necesario comparar el aporte de cada fuente de negocio y
  seleccionar un conjunto reducido.
- **Decisión:** entrenar un modelo por grupo de columnas y otro con todas; tomar
  las mejores de cada grupo.
- **Consecuencias:** identifica el origen más predictivo y evita que las
  fuentes con más columnas dominen el ranking.

### ADR-008: Binning con codificación por porcentaje de target

- **Contexto:** las variables finales tienen relaciones no lineales con el
  target y se busca una mejor interpretabilidad.
- **Decisión:** agrupar las features relevantes en bines y codificar cada bin
  con su porcentaje de target; las categóricas ya codificadas solo se agrupan
  en bines.
- **Consecuencias:** se captura la no linealidad sin ampliar dimensiones; los
  bines fusionados simplifican la lectura.

### ADR-009: LightGBM sin estandarizar ni balancear

- **Contexto:** los algoritmos basados en árboles no requieren estandarización.
  Y la proporción del target, cercana al 30 %, no está muy desbalanceada.
- **Decisión:** entrenar LightGBM sobre las variables transformadas sin
  estandarizar ni realizar oversampling.
- **Consecuencias:** pipeline más simple.

### ADR-010: Validación cruzada y búsqueda de hiperparámetros

- **Contexto:** el conjunto de datos es pequeño y la selección de
  hiperparámetros puede generar overfitting.
- **Decisión:** usar validación cruzada estratificada y búsqueda aleatoria de
  hiperparámetros, con ROC AUC como métrica y sin paralelismo en el
  clasificador.
- **Consecuencias:** la estimación es estratificada y reproducible; el
  paralelismo se reserva a la búsqueda para evitar competencia por CPU. Los
  nombres de clases y parámetros concretos quedan en `training.py`.

## Dependencias de desarrollo

### ADR-011: Commitizen para commits convencionales

- **Contexto:** el historial necesita mensajes consistentes y un versionado
  ligado a uv. Se evaluaron commitlint, que agrega una herramienta Node, y la
  disciplina manual, que no se puede verificar.
- **Decisión:** adoptar Commitizen para validar el mensaje de commit y
  gestionar la versión y el changelog.
- **Consecuencias:** el hook de commit puede rechazar mensajes fuera del
  formato; la versión se obtiene del proyecto uv y el changelog se actualiza en
  cada bump. La ejecución en los hooks se describe en
  [Hooks de Git](../development/hooks.md).

### ADR-012: Lefthook como orquestador de hooks

- **Contexto:** se necesitan hooks rápidos y paralelos para validaciones de
  Python y JavaScript sin incorporar un framework de hooks basado en Node. Se
  evaluaron pre-commit, con más dependencias y enfoque Python, y husky, que
  depende del ecosistema Node.
- **Decisión:** adoptar Lefthook como orquestador de los hooks de Git.
- **Consecuencias:** los trabajos se ejecutan en paralelo y la instalación se
  limita a un binario. Los hooks son opt-in y su configuración debe mantenerse
  alineada con [Comandos de desarrollo](../development/commands.md); los
  detalles de activación están en [Hooks de Git](../development/hooks.md).

### ADR-013: Marimo como entorno de notebook

- **Contexto:** el análisis necesita un notebook, reproducible. Se evaluó
  Jupyter, con estado oculto y diffs de JSON.
- **Decisión:** adoptar Marimo para el notebook del pipeline.
- **Consecuencias:** notebook almacenado como Python, con diffs legibles y
  ejecución sin interfaz. El uso se documenta en [Primeros pasos](../getting-started.md).

### ADR-014: Pyrefly para comprobación de tipos

- **Contexto:** el código usa anotaciones de Python 3.13 y necesita una
  comprobación de tipos. Se evaluaron mypy y pyright.
- **Decisión:** adoptar Pyrefly, escrito en rust, para comprobación de tipos.
- **Consecuencias:** el chequeo es rápido y la configuración se mantiene
  mínima. La validación se ejecuta en [Hooks de Git](../development/hooks.md).

### ADR-015: Ruff para lint y formato

- **Contexto:** se necesita lint y formato para Python. Se evaluaron Black con  
  Flake8 e isort, que multiplican herramientas y configuraciones, y pylint, más
  lento y con un enfoque distinto.
- **Decisión:** adoptar Ruff para lint y formato.
- **Consecuencias:** una sola herramienta cubre ambas tareas y la selección de
  reglas vive en `pyproject.toml`. Los comandos de validación están en
  [Comandos de desarrollo](../development/commands.md).

### ADR-016: rumdl para Markdown

- **Contexto:** la documentación necesita lint y formato. Se evaluaron
  markdownlint-cli2, que agrega otra herramienta Node; Prettier, con un estilo
  distinto y enfoque JavaScript; y la revisión manual, que no escala.
- **Decisión:** adoptar rumdl para lint y formato de Markdown.
- **Consecuencias:** el binario se instala con uv y los bloques Python dentro de
  los archivos markdown se validan con Ruff. La configuración vive en
  `pyproject.toml` y los comandos en [Comandos de desarrollo](../development/commands.md).

## Dependencias de JavaScript

### ADR-017: cspell para corrección ortográfica

- **Contexto:** la documentación y el código mezclan español e inglés, además de
  términos de dominio. Se necesita una herramienta con diccionarios. Se evaluaron
  codespell, que se limita a errores tipográficos; y prescindir del chequeo.
- **Decisión:** adoptar cspell para la corrección ortográfica.
- **Consecuencias:** la herramienta se ejecuta con Bun y su configuración de
  diccionarios y palabras propias vive en `cspell.json`. Puede producir falsos
  positivos. La ejecución se describe en [Hooks de Git](../development/hooks.md).

### ADR-018: SVGO para optimización de SVG

- **Contexto:** el notebook genera gráficos SVG que deben ocupar menos espacio,
  sin metadata ni comentarios, y evitar colisiones de IDs. Se evaluaron la
  limpieza manual, propensa a errores; y prescindir de la optimización, con
  archivos más pesados.
- **Decisión:** adoptar SVGO para optimizar los SVG generados.
- **Consecuencias:** agrega una dependencia JavaScript ejecutada con Bun y los
  tests que escriben SVG requieren SVGO.

## Herramientas del entorno

### ADR-019: uv como gestor del entorno Python

- **Contexto:** el proyecto necesita resolver, bloquear e instalar dependencias
  Python. Se evaluaron pip con venv y pip-tools, que reparten la tarea en varias
  herramientas; Poetry, con otro formato de lockfile; y conda, más pesado para
  este alcance.
- **Decisión:** adoptar uv para gestionar el entorno y las dependencias Python.
- **Consecuencias:** una sola herramienta cubre entorno, lockfile y ejecución; el
  lockfile habilita `--locked` y el contenedor desactiva la descarga de
  intérpretes. La versión de uv debe permanecer alineada con el backend de
  construcción y con el chequeo de versiones. Los detalles del contenedor están
  en [Docker](../development/docker.md).

### ADR-020: Bun como runtime y gestor JavaScript

- **Contexto:** las herramientas JavaScript, como cspell y SVGO, necesitan un
  runtime y un gestor de dependencias. Se evaluaron Node con npm o pnpm, que
  agregan más piezas, y Deno, con diferencias de compatibilidad
  con el ecosistema existente.
- **Decisión:** adoptar Bun como runtime y gestor de dependencias JavaScript.
- **Consecuencias:** un solo binario cubre instalación y ejecución, con un
  lockfile propio. El proyecto incorpora un segundo toolchain: `node_modules`
  se conserva en un volumen y el entrypoint lo sincroniza según el hash de los
  manifiestos. Los detalles están en [Docker](../development/docker.md).
