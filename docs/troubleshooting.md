# Solución de problemas

## El notebook no encuentra el conjunto de datos

Verificar que exista `data/example_data.parquet`. Si se cambió el nombre del
archivo, definir `ML_PATHS__RAW_DATA_FILE_NAME`. Si se movió la raíz del
proyecto, definir `ML_PATHS__PROJECT` con una ruta válida dentro del entorno
donde se ejecuta el notebook. La derivación de `raw_data` y la ruta `/app` del
contenedor están en [Configuración](configuration.md).

## Con Docker

### Dependencias y locks

Si el error menciona `--locked` o `--frozen-lockfile`, el lock no está
sincronizado. Para actualizarlo, ver
[Docker: actualizar locks](development/docker.md#comandos-para-actualizar-locks).
Si el error ocurrió durante `docker compose build`, reconstruir después de
actualizar el lock.

Si el error es de red, comprobar la conexión y reiniciar.

Si el error menciona que no se encuentra el intérprete de Python, ejecutar
`uv run python scripts/check_python_versions.py` y revisar `.python-version` y
`PYTHON_VERSION` en el Dockerfile. Cambiar la versión de Python requiere
reconstruir la imagen.

Como el proyecto se instala al arrancar el contenedor, un error de empaquetado
(por ejemplo, un `pyproject.toml` inválido o un `README.md` faltante) también
aparece en este paso, no durante `docker compose build`. Si el contenedor se
inició con `--entrypoint`, ese paso no se ejecuta; usar `uv sync --locked`
dentro del contenedor o `docker compose run --rm dev ...`.

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

## En local

### Bun no está instalado

Instalar Bun desde <https://bun.sh/docs/installation> y volver a abrir la
terminal. Sin Bun, los tests que requieren SVG se omiten y el notebook no
puede generar las visualizaciones.

### Aparece "ModuleNotFoundError"

Sincronizar el entorno virtual con `uv sync --locked`; ver
[Primeros pasos](getting-started.md#inicio-rápido-en-local).
