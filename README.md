# bank-clients-ml

## Objetivo del proyecto

Desarrollar un modelo de Machine Learning capaz de predecir qué clientes tienen
mayor probabilidad de adquirir un paquete específico de productos financieros,
optimizando las campañas de marketing del banco.

## Descripción general

El repositorio contiene:

- Un notebook de Marimo que documenta el flujo completo.
- Un paquete de Python con la lógica reutilizable.
- Tests unitarios y de integración.
- Un entorno de desarrollo reproducible con Docker o con uv y Bun en local.

El flujo incluye análisis exploratorio, limpieza, feature engineering, selección
de variables, entrenamiento con LightGBM y evaluación. Las visualizaciones se
guardan como SVG optimizados con SVGO en `notebooks/images/`. El directorio
`data/` contiene el conjunto de datos de ejemplo.

## Tecnologías

- Python 3.13 y uv.
- Marimo para el notebook interactivo.
- Polars, NumPy, SciPy, scikit-learn, LightGBM y Matplotlib.
- Pydantic Settings para la configuración mediante variables de entorno.
- pytest, Ruff, Pyrefly y rumdl.
- Bun, SVGO y cspell.
- Docker y Docker Compose para el entorno de desarrollo.
- Lefthook y Commitizen para los hooks de Git opcionales.

## Requisitos para probar este proyecto

Clonar el repositorio y entrar al directorio:

```bash
git clone <URL-del-repositorio>
cd bank-clients-ml
```

Después, elegir una de las dos opciones.

### Opción 1: Docker

- Docker Engine o Docker Desktop con Docker Compose v2.
- No es necesario instalar Python, uv ni Bun en la máquina host.
- En Windows se recomienda usar WSL2 para el montaje; ver
  [rendimiento del montaje](#el-proyecto-está-en-windows-y-el-montaje-es-lento).

<details>
  <summary><b>Inicio rápido con Docker</b></summary>
  <br>

```bash
docker compose up
```

Abrir <http://localhost:2718> en el navegador. La primera vez construye la
imagen y descarga Python, uv, Bun y las dependencias; las ejecuciones siguientes
arrancan en segundos. Marimo se publica únicamente en `127.0.0.1` (así que no es
accesible desde otra máquina de la red) y se ejecuta sin token de acceso. El
contenedor arranca Marimo con `--watch` y recarga el notebook cuando el archivo
se guarda desde el host.

Para detener y eliminar el contenedor:

```bash
docker compose down
```

Para ejecutarlo en segundo plano:

```bash
docker compose up -d
```

#### Actualizar el entorno

Los archivos montados con bind mount se actualizan sin reconstruir la imagen.

| Cambio | Acción | ¿Reconstruir? |
| --- | --- | --- |
| `src/`, `tests/`, `notebooks/`, `README.md`, `data/` y configuración de herramientas (excepto `pyproject.toml`) | Nada: están montados con bind mount | No |
| `pyproject.toml`, `uv.lock` | `docker compose restart dev` (o `docker compose up -d` si está detenido) | No obligatorio |
| `package.json`, `bun.lock` | `docker compose restart dev` (o `docker compose up -d` si está detenido) | No obligatorio |
| `Dockerfile`, `.dockerignore`, `scripts/docker_entrypoint.py` | `docker compose up --build` | Sí |
| `.python-version`, `PYTHON_VERSION` o imágenes base | Validar versiones (ver [Comandos comunes](#comandos-comunes)) y `docker compose build --pull && docker compose up` | Sí |

Para los cambios de locks, reconstruir es opcional para el contenedor actual,
pero recomendado para mantener la imagen reproducible y actualizar la copia base
de `/opt/venv` y `node_modules` para contenedores nuevos. El primer arranque
después de un cambio necesita red si hay paquetes nuevos que descargar.

El entrypoint ejecuta `uv sync --locked` contra `/opt/venv` para Python; si los
locks cambiaron, actualiza el entorno. Para JavaScript, ejecuta
`bun install --frozen-lockfile` en el volumen `node_modules` cuando el hash de
`package.json`, `bun.lock` y la versión de Bun difiere de
`node_modules/.bun-install-hash`. Si el entorno ya está al día, no reinstala
nada.

Como el entrypoint usa `--locked` y `--frozen-lockfile`, si los locks no están
sincronizados la sincronización falla; ver
[Dependencias y locks](#dependencias-y-locks).

Los comandos con `--entrypoint` no ejecutan esta sincronización; ver
[Comandos comunes](#comandos-comunes).

El proyecto se instala en `/opt/venv` desde el código montado la primera vez que
arranca un contenedor nuevo. `/opt/venv` no es un volumen: cada contenedor parte
de la copia de la imagen y sincroniza si hace falta; al detener el contenedor se
descartan los cambios. `node_modules` sí es un named volume.

Para restablecer el volumen
`node_modules`, ver [Dependencias y locks](#dependencias-y-locks). Ninguna de
estas operaciones borra `data/` ni `notebooks/images/`, porque forman parte del
montaje del proyecto. Para actualizar la copia base de `/opt/venv`, reconstruir
la imagen.

<details>
  <summary><b>Detalles de la imagen Docker</b></summary>
  <br>

El Dockerfile separa Python y uv (`python-base`), las librerías del sistema
(`system-deps`), las dependencias de Python (`py-deps`) y las de JavaScript
(`js-deps`). `py-deps` instala solo dependencias de terceros, de modo que los
cambios en `src/` no invalidan la capa de dependencias. La imagen final es el
entorno `dev`, que incluye Python, uv, Marimo, tests, linters, Bun, SVGO y
cspell; no hay un entorno de producción todavía.

Los entornos de dependencias pueden construirse en paralelo; cambiar `apt`, el
UID/GID o Bun no invalida las dependencias de Python. `system-deps` usa cache
mounts de BuildKit para el índice y los paquetes de `apt`, y el primer `uv sync`
monta `.python-version` para fallar rápido si la versión no coincide con la
imagen. El entrypoint no descarga intérpretes (`UV_PYTHON_DOWNLOADS=never`); solo
sincroniza paquetes con el intérprete de la imagen.

El entrypoint vive en `scripts/docker_entrypoint.py`; se copia a
`/usr/local/bin` durante la construcción, así que sus cambios requieren
reconstruir la imagen. Si cambia la versión de Bun en el Dockerfile, hay que
reconstruir y el volumen se actualiza en el siguiente arranque. La primera vez
que se usa un volumen creado con una versión anterior del entrypoint, el marker
no existe y se ejecuta una instalación completa.

</details>

</details>

### Opción 2: Local

- [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.12.1, o una
  version superior que sea menor a 0.13.0.
- [Bun](https://bun.sh/docs/installation) 1.4 o superior.
- Python 3.13; uv puede instalarlo automáticamente.

<details>
  <summary><b>Inicio rápido en local</b></summary>
  <br>

```bash
uv python install 3.13
uv sync --locked
bun install --frozen-lockfile
uv run marimo edit notebooks/notebook.py
```

El comando `uv sync` instala las dependencias de Python, incluido el grupo de
desarrollo. `bun install` instala SVGO, cspell y el resto de las herramientas de
JavaScript. Marimo abre el notebook en el navegador.

Para ejecutar el notebook en modo de solo lectura:

```bash
uv run marimo run notebooks/notebook.py
```

Para no abrir el navegador y recargar el notebook cuando el archivo se guarda
desde otro editor:

```bash
uv run marimo edit notebooks/notebook.py --headless --watch
```

`--headless` deja la URL en la terminal para abrirla manualmente y `--watch`
aplica los cambios guardados fuera de Marimo.

</details>

## Comandos comunes

Ejecutar desde la raíz del proyecto. En local, usar el comando tal cual; con
Docker, anteponer `docker compose run --rm dev`, que crea un contenedor temporal
y reutiliza la imagen y el volumen de `node_modules`. El entrypoint sincroniza
las dependencias Python y JavaScript antes de ejecutar el comando; ver
[Actualizar el entorno](#actualizar-el-entorno). Por ejemplo:

```bash
docker compose run --rm dev uv run pytest -m "not slow"
```

Los comandos con `--entrypoint` reemplazan el entrypoint y se saltan la
sincronización de dependencias y la instalación del proyecto. Usarlos solo para
herramientas que no necesitan `bank_clients_ml`, como `uv lock` o `bun install`.
Para tests o scripts que importan el paquete, usar
`docker compose run --rm dev ...`.

En Docker y en local, si se usa `--watch` al ejecutar el notebook de Marimo,
evitar editar la misma celda en dos lugares al mismo tiempo.

| Tarea | Comando |
| --- | --- |
| Ejecutar tests rápidos | `uv run pytest -m "not slow"` |
| Ejecutar todos los tests con cobertura | `uv run pytest --cov=bank_clients_ml --cov-report=term-missing --cov-fail-under=80` |
| Revisar el código con Ruff | `uv run ruff check .` |
| Comprobar el formato | `uv run ruff format --check .` |
| Comprobar tipos | `uv run pyrefly check` |
| Validar versiones de Python y uv | `uv run python scripts/check_python_versions.py` |
| Revisar Markdown | `uv run rumdl check .` |
| Revisar ortografía | `bun cspell --no-must-find-files "**"` |

### Ciclo de vida del contenedor

Los comandos `exec` y `logs` requieren que el servicio esté en ejecución:
`docker compose up -d`

| Tarea | Comando |
| --- | --- |
| Abrir una terminal temporal | `docker compose run --rm dev bash` |
| Abrir una terminal | `docker compose exec dev bash` |
| Ver los registros | `docker compose logs -f dev` |

Para aplicar cambios de dependencias, usar `docker compose restart dev` como se
describe en [Actualizar el entorno](#actualizar-el-entorno).
`docker compose exec dev ...` no ejecuta el entrypoint, porque el contenedor ya
está iniciado.

## Configuración

La configuración central está en `src/bank_clients_ml/config.py`. La clase
`Settings` agrupa las rutas, los nombres de las columnas y los parámetros
globales, como `random_state` y `n_jobs`. Los valores por defecto están
definidos en ese archivo y se pueden redefinir con variables de entorno.

Las variables de la aplicación usan el prefijo `ML_`. Los campos anidados se
separan con `__`. Algunos ejemplos de variables que se pueden definir en un
`.env`:

```bash
ML_DEBUG=true
ML_RANDOM_STATE=42
ML_N_JOBS=4
ML_PATHS__RAW_DATA_FILE_NAME=example_data.parquet
ML_DATASET__TARGET=Target
```

El listado completo de variables y sus valores por defecto está en `Settings`,
`DirectoryPaths`, `DatasetColumns` y `AnalysisColumns`, en
`src/bank_clients_ml/config.py`. Las rutas `raw_data` e `images` se calculan a
partir de `paths.project`; `raw_data` es
`paths.project / "data" / paths.raw_data_file_name`. No se configuran de forma
independiente. En el uso normal no hace falta definir `ML_PATHS__PROJECT`: se
calcula desde la ubicación de `config.py` y en el contenedor de desarrollo
corresponde a `/app`.

También se puede crear un archivo `.env` en la raíz del proyecto. Es opcional y
está ignorado por Git. Del archivo solo se leen las variables con prefijo `ML_`,
así que las variables de Docker Compose pueden convivir en el mismo `.env`. En
el contenedor de desarrollo se lee desde `/app/.env` mediante el montaje del
proyecto; no se copia dentro de la imagen.

Las variables de entorno del sistema tienen prioridad sobre el `.env`, y este
sobre los valores por defecto.

Si se reemplaza el conjunto de datos, conservar el nombre
`example_data.parquet` o definir `ML_PATHS__RAW_DATA_FILE_NAME`. Después de
modificar el `.env` o las variables de entorno, reiniciar el proceso o el
contenedor. `Settings.settings_cached()` guarda la configuración en caché y
un proceso ya iniciado no detecta los cambios. Para invalidar esa caché,
llamar a `Settings.settings_clear_cache()`.

## Decisiones de arquitectura (ADRs)

Registro de las decisiones de diseño que condicionan el flujo del notebook y
del paquete. Cada entrada indica el contexto, la decisión adoptada y sus
consecuencias.

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
  sin expandir la dimensionalidad; se valida que el target esté presente en
  cada categoría.

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
- **Decisión:** eliminar en orden columnas constantes, columnas binarias
  desbalanceadas y columnas numéricas con correlación alta.
- **Consecuencias:** menor costo computacional y menos ruido antes del ranking;
  los descartes se conservan para inspección.

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
- **Consecuencias:** flujo más simple.

### ADR-010: Validación cruzada y búsqueda de hiperparámetros

- **Contexto:** el conjunto de datos es pequeño y la selección de
  hiperparámetros puede generar overfitting.
- **Decisión:** usar `RandomizedSearchCV` con `StratifiedKFold`, métrica ROC AUC
  y `n_jobs=1` en el clasificador.
- **Consecuencias:** estimación estratificada y reproducible; el clasificador no
  compite con los procesos de la búsqueda.

## Hooks de Git (opcionales)

Los hooks no se instalan solos. Si se clona el repositorio para desarrollar o
hacer un fork, se pueden activar para mantener la consistencia y detectar
problemas antes de subir cambios. El stack de validación es rápido: uv, Ruff,
Pyrefly y rumdl están escritos en Rust, y cspell y SVGO corren con Bun. Ruff y
rumdl trabajan solo sobre los archivos agregados al staging; la parte más pesada
es pytest.

Requieren usar el proyecto en local (ver [Opción 2: Local](#opción-2-local)), ya
que dependen de uv y Bun. Para probar el proyecto con Docker, se puede omitir
esta sección y realizar las validaciones manualmente con los comandos de la tabla
de [Comandos comunes](#comandos-comunes).

Para instalarlos:

```bash
uv run lefthook install
```

Con los hooks activos:

- Commit: se comprueban los lockfiles si cambian; se formatean y revisan con
  Ruff los archivos Python; se ejecuta pytest sin los tests "slow" con
  cobertura mínima del 80 %; se revisan los Markdown con rumdl y rumdl fmt; se
  validan la configuración de Docker Compose, la de Lefthook y el mensaje de
  commit con Commitizen.
- Push: se ejecuta Pyrefly, el chequeo de versiones de Python y uv, todos los
  tests de pytest (unitarios + integración) y cspell.
- Checkout, merge y rewrite: se sincroniza el entorno con `uv sync --locked`.

El hook `python-versions-check` ejecuta `scripts/check_python_versions.py`
cuando cambian el Dockerfile, `.python-version`, `pyproject.toml` o el propio
script. Si falla, corregir los valores para que `ARG PYTHON_VERSION`,
`.python-version`, `requires-python` y `UV_VERSION` queden alineados con el
rango de `uv_build`.

Para desactivarlos:

```bash
uv run lefthook uninstall
```

## Solución de problemas

<details>
  <summary><b>Con Docker</b></summary>
  <br>

### Dependencias y locks

Esta sección cubre los errores de `uv sync --locked` y
`bun install --frozen-lockfile`. Si el error menciona `--locked` o
`--frozen-lockfile`, el lock no está sincronizado. Actualizar el lock:

| Lock | En local | En el contenedor |
| --- | --- | --- |
| `uv.lock` | `uv lock` | `docker compose run --rm --entrypoint dev uv lock` |
| `bun.lock` | `bun install` | `docker compose run --rm --entrypoint dev bun install` |

Los comandos con `--entrypoint` solo actualizan el lock (ver
[Comandos comunes](#comandos-comunes)). Los cambios se escriben en el host a
través del bind mount; revisar el diff y reiniciar:

```bash
docker compose restart dev
```

Si el error ocurrió durante `docker compose build`, reconstruir después de
actualizar el lock. Si el error es de red, comprobar la conexión y reiniciar.

Si el error menciona que no se encuentra el intérprete de Python, ejecutar
`uv run python scripts/check_python_versions.py` y revisar `.python-version` y
`PYTHON_VERSION` en el Dockerfile. Cambiar la versión de Python requiere
reconstruir la imagen.

Como el proyecto se instala al arrancar el contenedor, un error de empaquetado
(por ejemplo, un `pyproject.toml` inválido o un `README.md` faltante) también
aparece en este paso, no durante `docker compose build`. Si el contenedor se
inició con `--entrypoint`, ese paso no se ejecuta; usar `uv sync --locked` dentro
del contenedor o `docker compose run --rm dev ...`.

Para resetear el volumen de `node_modules`:

```bash
docker compose down -v
docker compose up --build
```

Si la construcción falla durante `uv sync`, comprobar la conexión y reintentar
sin caché:

```bash
docker compose build --no-cache --pull
```

### El puerto 2718 del host está ocupado

Cambiar el puerto publicado:

```bash
MARIMO_HOST_PORT=3000 docker compose up
```

En PowerShell:

```powershell
$env:MARIMO_HOST_PORT = "3000"
docker compose up
```

### Aparece "Permission denied" al escribir archivos

En Linux, el UID o el GID del contenedor no coinciden con los del usuario de la
máquina host. Reconstruir la imagen con los valores correctos y resetear el
volumen:

```bash
docker compose down -v
docker compose build --build-arg UID="$(id -u)" --build-arg GID="$(id -g)"
docker compose up
```

Si el error aparece solo al escribir en `node_modules`, el volumen puede venir
de una imagen anterior con otro propietario; `docker compose down -v` lo recrea
con los permisos actuales. Si aparece en `/opt/venv`, la copia del entorno en la
imagen quedó con otro propietario; reconstruir la imagen con los UID/GID
correctos.

### Marimo no abre en el navegador

Comprobar que el servicio esté en ejecución y revisar los registros:

```bash
docker compose ps
docker compose logs -f dev
```

Verificar que se esté accediendo a <http://localhost:2718> y que el puerto no
esté ocupado por otro proceso.

### El proyecto está en Windows y el montaje es lento

Se recomienda clonar el repositorio dentro del sistema de archivos de WSL2 y
trabajar desde allí. Docker Desktop debe tener habilitada la integración con la
distribución de WSL2 correspondiente.

</details>

<details>
  <summary><b>En local</b></summary>
  <br>

### Bun no está instalado

Instalar Bun desde <https://bun.sh/docs/installation> y volver a abrir la
terminal. Sin Bun, los tests que requieren SVG se omiten y el notebook no
puede generar las visualizaciones.

### Aparece "ModuleNotFoundError"

Sincronizar el entorno virtual:

```bash
uv sync --locked
```

</details>

### El notebook no encuentra el conjunto de datos

Verificar que exista `data/example_data.parquet`. Si se cambió el nombre del
archivo, definir `ML_PATHS__RAW_DATA_FILE_NAME`. Si se movió la raíz del
proyecto, definir `ML_PATHS__PROJECT` con una ruta válida dentro del entorno
donde se ejecuta el notebook. La derivación de `raw_data` y la ruta `/app` del
contenedor están en [Configuración](#configuración).

## Contribución

Este repositorio se mantiene como un proyecto personal, por lo que no se aceptan
contribuciones externas. Ver [CONTRIBUTING.md](CONTRIBUTING.md).

## Licencia

Este proyecto está bajo los términos de la **Licencia MIT**. Consultar el
archivo [LICENSE](LICENSE) para más detalles. Las dependencias externas
utilizadas mantienen sus respectivas licencias, tales como Apache 2.0.
