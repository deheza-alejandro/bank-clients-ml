# bank-clients-ml

## Objetivo del proyecto

Desarrollar un modelo de Machine Learning capaz de
predecir qué clientes tienen mayor probabilidad de adquirir un paquete
específico de productos financieros, optimizando las campañas de marketing
del banco.

## Descripción general

El repositorio contiene:

- Un notebook de Marimo que documenta el flujo completo de modelado.
- Un paquete de Python con la lógica reutilizable del proyecto.
- Tests unitarios y de integración.
- Un entorno de desarrollo reproducible con Docker.
- Una alternativa local basada en uv y Bun.

El flujo incluye análisis exploratorio, limpieza, feature engineering,
selección de variables, entrenamiento con LightGBM y evaluación. El notebook
principal está desarrollado en Marimo y las visualizaciones se guardan como SVG
optimizados con SVGO.

El directorio `notebooks/images/` contiene las visualizaciones generadas por el
notebook. El directorio `data/` contiene el conjunto de datos de ejemplo.

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

- Primero se debe clonar el repositorio con git.

```bash
git clone <URL-del-repositorio>
```

- Luego moverse al directorio del proyecto.

```bash
cd bank-clients-ml
```

- Luego hay 2 opciones para continuar:

### Opción 1: Docker

- Docker Engine o Docker Desktop con Docker Compose v2.
- No es necesario instalar Python, uv ni Bun en la máquina host.
- En Windows se recomienda usar WSL2 para mejorar el rendimiento del montaje.

<details>
  <summary><b>Inicio rápido con Docker</b></summary>
  <br>

```bash
docker compose up
```

Abrir <http://localhost:2718> en el navegador.

La primera vez construye la imagen y descarga Python, uv, Bun y las
dependencias. Las ejecuciones siguientes arrancan en
segundos.
Marimo se publica únicamente en `localhost` y se ejecuta sin token de acceso.
El contenedor arranca Marimo con `--watch`, así que también recarga el
notebook cuando el archivo se guarda desde el host. Evitar editar la misma
celda en dos lugares al mismo tiempo.

Para detener el entorno:

```bash
docker compose down
```

Para ejecutarlo en segundo plano:

```bash
docker compose up -d
```

#### Cuándo hace falta reconstruir la imagen

| Cambio | ¿Rebuild? |
| --- | --- |
| `src/`, `tests/`, `notebooks/`, `README.md`, `data/` | No: están montados con bind mount |
| `Dockerfile`, `.dockerignore`, `scripts/docker_entrypoint.py` | Sí |
| `uv.lock`, `pyproject.toml`, `.python-version` | Sí (recomendado) |
| `package.json`, `bun.lock` | Sí (recomendado) |

Usar el siguiente comando sólo si cambió algo de la tabla anterior que requiere
un rebuild:

```bash
docker compose up --build
```

#### Cuando No hace falta rebuild

El código se monta desde la máquina host, por lo que los cambios en `src/`,
`tests/`, `notebooks/`, `README.md`, `data/` y la configuración de las
herramientas se reflejan sin reconstruir la imagen. Marimo arranca con
`--watch` y recarga el notebook cuando el archivo se guarda fuera del
navegador; el entrypoint sincroniza el proyecto y las dependencias al
reiniciar el servicio.

El archivo `.env` de la aplicación también se lee desde `/app/.env` mediante el
montaje y no se copia dentro de la imagen.

#### Dos matices

- Si cambian `pyproject.toml` o `uv.lock` y no querés reconstruir, `docker
  compose restart dev` vuelve a ejecutar el entrypoint y sincroniza `/opt/venv`.
  Lo mismo aplica a `package.json` o `bun.lock` con el volumen de
  `node_modules`.
- `docker compose down` elimina el contenedor; `docker compose down -v` además
  elimina el volumen `node_modules` y el próximo arranque lo regenera desde la
  imagen. Ninguno de los dos borra `data/` ni `notebooks/images/`, porque forman
  parte del montaje del proyecto.

El entorno virtual de Python está ubicado en `/opt/venv` y `node_modules` se
guarda en un named volume.

El Dockerfile separa Python y uv (`python-base`), las librerías del sistema
(`system-deps`), las dependencias de Python (`py-deps`) y las de JavaScript
(`js-deps`). `py-deps` instala solamente las dependencias de terceros: el
proyecto se instala en `/opt/venv` al arrancar el contenedor, de modo que los
cambios en `src/` no invalidan la capa de dependencias. Los comandos que
reemplazan el entrypoint no ejecutan esa instalación; ver
[Comandos comunes](#comandos-comunes). La imagen final se construye en el
entorno `dev`, que incluye Python, uv, Marimo, los tests, los
linters, Bun, SVGO y cspell. Los entornos de dependencias pueden construirse en
paralelo, y cambiar `apt`, el UID/GID o Bun no invalida las dependencias de
Python. `system-deps` usa cache mounts de BuildKit para el índice y los paquetes
de `apt`, y el primer `uv sync` monta `.python-version` para fallar rápido si la
versión no coincide con la imagen. No hay un entorno de producción todavía. El
entrypoint vive en `scripts/docker_entrypoint.py`; como se copia a
`/usr/local/bin` durante la construcción, los cambios en ese archivo requieren
reconstruir la imagen.

El entrypoint ejecuta `uv sync --locked` antes de ejecutar el comando. Si
`pyproject.toml` o `uv.lock` cambiaron, uv actualiza `/opt/venv`; si el entorno
ya está al día, no reinstala nada. En la imagen, `/opt/venv` contiene solo las
dependencias de terceros; el proyecto se instala desde el código montado la
primera vez que arranca un contenedor nuevo. Los comandos que reemplazan el
entrypoint no ejecutan esta sincronización y no tienen el proyecto instalado;
ver [Comandos comunes](#comandos-comunes). Como usa `--locked`, si el lock no
está actualizado el arranque falla; consultar
[Cambios en dependencias](#cambios-en-dependencias), en la sección de
[Solución de problemas](#solución-de-problemas). El entorno virtual no es un
volumen: cada contenedor nuevo parte de la copia de la imagen y sincroniza si
hace falta.

Para JavaScript, el entrypoint compara un hash de `package.json`, `bun.lock` y
la versión de Bun con `node_modules/.bun-install-hash`. Si cambió, ejecuta
`bun install --frozen-lockfile` antes de iniciar Marimo. Los cambios en
`package.json` o `bun.lock` se aplican al reiniciar el servicio. La primera vez
que se usa un volumen creado antes de este cambio, el marker no existe y la
instalación se ejecuta una vez.

</details>

### Opción 2: Local

- [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.12 o superior.
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
aplica los cambios guardados fuera de Marimo. Evitar editar la misma celda en
dos lugares al mismo tiempo.

</details>

## Comandos comunes

Ejecutar desde la raíz del proyecto. En local, usar el comando tal cual; con
Docker, anteponer `docker compose run --rm dev`, que crea un contenedor
temporal, reutilizando la imagen y el volumen de `node_modules`. El entrypoint
sincroniza las dependencias JavaScript y Python antes de ejecutar el comando;
si ya están al día, la comprobación es rápida. Por ejemplo, para ejecutar los
tests rápidos:
`docker compose run --rm dev uv run pytest -m "not slow"`

Los comandos que usan `--entrypoint` reemplazan el entrypoint y, por lo tanto,
se saltan la sincronización de dependencias y la instalación del proyecto. Usarlos
solo para herramientas que no necesitan `bank_clients_ml`, como `uv lock` o
`bun install`. Para tests o scripts que importan el paquete, usar
`docker compose run --rm dev ...`, que sí ejecuta el entrypoint.

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

`docker compose restart dev` vuelve a ejecutar el entrypoint, por lo que aplica
cambios en `package.json`, `bun.lock`, `pyproject.toml` o `uv.lock`.
`docker compose exec dev ...` no lo ejecuta, porque el contenedor ya está
iniciado.

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

El listado completo de variables disponibles y sus valores por defecto está en
`Settings`, `DirectoryPaths`, `DatasetColumns` y `AnalysisColumns`, en
`src/bank_clients_ml/config.py`. Las rutas `raw_data` e `images` son calculadas
a partir de `paths.project`; no se configuran de forma independiente. En el uso
normal no hace falta definir `ML_PATHS__PROJECT`: se calcula desde la ubicación
de `config.py` y en el contenedor de desarrollo corresponde a `/app`.

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

## Hooks de Git (opcionales)

Los hooks no se instalan solos, pero si clonas el repositorio para desarrollar o
hacer un fork, podes activarlos para mantener la consistencia y detectar
problemas antes de subir cambios. El stack de validación es rápido: uv, Ruff,
Pyrefly y rumdl están escritos en Rust, y cspell y SVGO corren con Bun. Además,
Ruff y rumdl trabajan solo sobre los archivos agregados al staging. la parte más
pesada es pytest.

Requiere usar este proyecto en local (ver [Opción 2: Local](#opción-2-local)),
ya que los hooks dependen de uv y bun. Para probar este proyecto con docker,
podes omitir esta sección y realizar las validaciones manualmente con los
comandos de la tabla de [Comandos comunes](#comandos-comunes).

Para instalarlos en local:

```bash
uv run lefthook install
```

Con los hooks activos:

- Commit: comprueba los lockfiles si cambian, formatea y revisa con Ruff los
archivos Python agregados al staging, ejecuta pytest sin los tests "slow" con
cobertura mínima del 80 %, revisa los archivos Markdown agregados al staging con
rumdl y valida el mensaje de commit con Commitizen.
- Push: ejecuta Pyrefly, valida que `ARG PYTHON_VERSION`,
`.python-version` y `requires-python` estén alineados, verifica que `UV_VERSION`
cumpla el rango de `uv_build`, todos los tests de pytest (unitarios +
integración) y cspell.
- Checkout y merge: sincroniza el entorno con uv sync --locked.

Para desactivarlos:

```bash
uv run lefthook uninstall
```

## Solución de problemas

<details>
  <summary><b>Con Docker</b></summary>
  <br>

### Cambios en dependencias

Los cambios de código, datos, documentación y configuración de herramientas no
requieren reconstruir la imagen. Para las dependencias:

- Si cambian `pyproject.toml` o `uv.lock`:

  No es obligatorio reconstruir la imagen ni ejecutar `uv sync` a mano.

  Con el servicio en ejecución:

  ```bash
  docker compose restart dev
  ```

  Si el contenedor está detenido:

  ```bash
  docker compose up -d
  ```

  El entrypoint ejecuta `uv sync --locked` contra `/opt/venv`. Si el entorno ya
  está al día, no reinstala nada. Como el comando usa `--locked`, no modifica
  `uv.lock`; si `pyproject.toml` y `uv.lock` no están sincronizados, el arranque
  falla. En ese caso, ver
  [El contenedor falla al instalar dependencias Python](#el-contenedor-falla-al-instalar-dependencias-python).

  Reconstruir la imagen es opcional para el contenedor actual, pero es lo
  recomendado para mantener la imagen reproducible y la copia de `/opt/venv`
  actualizada para los contenedores nuevos. El primer arranque después de un
  cambio necesita red si hay paquetes nuevos que descargar.

- Si cambia `package.json` o `bun.lock`:

  No es obligatorio reconstruir la imagen ni ejecutar `bun install` a mano.

  Con el servicio en ejecución:

  ```bash
  docker compose restart dev
  ```

  Si el contenedor está detenido:

  ```bash
  docker compose up -d
  ```

  El entrypoint detecta el cambio y ejecuta `bun install --frozen-lockfile` dentro
  del volumen `node_modules`. Como el comando usa `--frozen-lockfile`, si
  `package.json` y `bun.lock` no están sincronizados, la instalación falla. En ese
  caso, ver
  [El contenedor falla al instalar dependencias JavaScript](#el-contenedor-falla-al-instalar-dependencias-javascript)

- Si cambia la versión de Python en `.python-version`, `PYTHON_VERSION` en el
  Dockerfile o se quieren actualizar las imágenes base:

  Antes de reconstruir, ejecutar el chequeo de versiones:

  ```bash
  uv run python scripts/check_python_versions.py
  ```

  El hook `python-versions-check` también lo ejecuta cuando cambian el Dockerfile,
  `.python-version` o `pyproject.toml`. Si falla, corregir los valores para que
  `ARG PYTHON_VERSION`, `.python-version`, `requires-python` y el rango de
  `uv_build` queden alineados.

  ```bash
  docker compose build --pull
  docker compose up
  ```

  El entrypoint no descarga intérpretes de Python; solo sincroniza paquetes con
  el intérprete que ya está en la imagen (`UV_PYTHON_DOWNLOADS=never`).

- `docker compose down -v` elimina el volumen de `node_modules`, pero no borra
  `data/` ni `notebooks/images/`, porque esos directorios forman parte del
  montaje del proyecto. No hay un volumen de `/opt/venv`: ese entorno vive en la
  imagen y en la capa editable del contenedor, así que `down` descarta los
  cambios del contenedor. Para actualizar la copia base de la imagen hay que
  reconstruirla.

### El contenedor falla al instalar dependencias Python

Si el error menciona `--locked` o que `uv.lock` está desactualizado,
`pyproject.toml` y `uv.lock` no están sincronizados. Actualizar el lock en
local:

```bash
uv lock
```

Si uv no está instalado en el host, actualizar el lock desde el contenedor:

```bash
docker compose run --rm --entrypoint uv dev lock
```

Ese comando reemplaza el entrypoint, así que no ejecuta `uv sync` ni instala el
proyecto; es intencional, porque solo actualiza el lock. No lo uses para
comandos que importan `bank_clients_ml`.

Modifica `uv.lock` a través del bind mount. Revisar el diff de
`uv.lock` y reiniciar:

```bash
docker compose restart dev
```

Reconstruir la imagen es opcional para el contenedor actual, pero es lo
recomendado para mantener la imagen reproducible y la copia de `/opt/venv`
actualizada para los contenedores nuevos. Si el error ocurrió durante
`docker compose build`, volver a construir la imagen después de actualizar el
lock.

Si el error es de red, comprobar la conexión y reiniciar; el primer sync después
de un cambio necesita descargar los paquetes nuevos. Si el error menciona que no
se encuentra el intérprete de Python, ejecutar
`uv run python scripts/check_python_versions.py` y revisar `.python-version` y
`PYTHON_VERSION` en el Dockerfile: cambiar la versión de Python requiere
reconstruir la imagen.

Como el proyecto se instala en `/opt/venv` al arrancar el contenedor, un error de
empaquetado (por ejemplo, un `pyproject.toml` inválido o un `README.md`
faltante) también aparece en este paso, no durante `docker compose build`. Si el
contenedor se inició con `--entrypoint`, este paso no se ejecuta; en ese caso
usar `uv sync --locked` dentro del contenedor o usar
`docker compose run --rm dev ...`.

### El contenedor falla al instalar dependencias JavaScript

Si el error menciona `--frozen-lockfile`, `package.json` y `bun.lock` no están
sincronizados. Actualizar el lock con Bun en local:

```bash
bun install
```

Si Bun no está instalado en el host, actualizar el lock desde el contenedor:

```bash
docker compose run --rm --entrypoint bun dev install
```

Ese comando reemplaza el entrypoint, así que no ejecuta la sincronización de
dependencias; es intencional, porque solo actualiza `bun.lock`.

Modifica `bun.lock` a través del bind mount. Revisar el diff de
`bun.lock` y reiniciar:

```bash
docker compose restart dev
```

Reconstruir la imagen es opcional para el contenedor actual, pero es lo
recomendado para mantener la imagen reproducible y la copia de `node_modules`
actualizada para los volúmenes nuevos.
Si cambia la versión de Bun en el Dockerfile, hay que reconstruir; el hash
incluye `bun --version` y actualiza el volumen en el siguiente arranque.

Si el error ocurrió durante `docker compose build`, volver a construir la imagen
después de actualizar el lock.

Para resetear el volumen de `node_modules` por completo:

```bash
docker compose down -v
docker compose up --build
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
máquina host. Reconstruir la imagen con los valores correctos y resetear
el volumen:

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
esté ocupado por otro proceso. El contenedor publica el puerto solo en
`127.0.0.1`, así que no es accesible desde otra máquina de la red.

### La construcción de Docker falla durante uv sync

Comprobar la conexión a Internet. También se puede reintentar sin caché y
actualizando las imágenes base:

```bash
docker compose build --no-cache --pull
```

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
donde se ejecuta el notebook; en el contenedor de desarrollo, la raíz es
`/app`. Recordar que `raw_data` se calcula como
`paths.project / "data" / paths.raw_data_file_name`.

## Contribución

Este repositorio se mantiene como un proyecto personal, por lo que no se aceptan
contribuciones de código ni Pull Requests en este momento.

Agradezco el interés en colaborar, pero te invito a hacer un fork del
repositorio si deseas adaptarlo o modificarlo para tus propios fines bajo los
términos de la licencia.

Si haces un fork y buscas mantener la misma consistencia, podes activar los
[hooks de Git](#hooks-de-git-opcionales). Son opcionales y no son necesarios
para revisar el proyecto.

## Licencia

Este proyecto está bajo los términos de la **Licencia MIT**. Consulta el archivo
[LICENSE] para más detalles. Las dependencias externas utilizadas
mantienen sus respectivas licencias, tales como Apache 2.0.
