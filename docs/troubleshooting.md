# Solución de problemas

## El notebook no encuentra el conjunto de datos

Verificar que exista `data/example_data.parquet`. Si se cambió el nombre del
archivo, definir `ML_PATHS__RAW_DATA_FILE_NAME`. Si se movió la raíz del
proyecto, definir `ML_PATHS__PROJECT` con una ruta válida dentro del entorno
donde se ejecuta el notebook. La derivación de rutas y la configuración del
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

Un error de empaquetado (por ejemplo, un `pyproject.toml` inválido o un
`README.md` faltante) aparece al iniciar el contenedor, no durante
`docker compose build`; el motivo está en
[Docker](development/docker.md#detalles-de-la-imagen). Si el contenedor se
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

### El sistema host es Windows y el montaje es lento

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

### Sincronización fallida tras checkout, merge o rewrite

El hook `uv-sync-locked` falla cuando `uv sync --locked` no puede sincronizar
el entorno después de una operación de checkout, merge o rewrite. Para
determinar la causa, ejecutar:

```bash
uv lock --check
```

Si el comando falla, el lockfile está desactualizado, falta o no se puede
resolver. En ese caso:

1. Revisar `git status` y guardar temporalmente o descartar los cambios
   locales en `pyproject.toml` que no se vayan a incluir en el commit, ya que
   `uv lock` los incorporaría al lockfile.
2. Verificar la versión de uv si el error menciona un lockfile no soportado;
   una versión incompatible puede reescribir el lockfile con un formato
   distinto.
3. Ejecutar `uv lock`.
4. Revisar el diff de `uv.lock` y confirmar que solo cambió lo esperado para
   no mezclar el arreglo con otras dependencias.
5. Ejecutar `uv sync --locked`.
6. Agregar `uv.lock` al commit.

Si `uv.lock` tiene marcadores de conflicto o no se puede leer, restaurarlo
desde Git o resolver el conflicto antes de ejecutar `uv lock`. Si `uv lock`
falla por red, credenciales o un conflicto de versiones, corregir esa causa
antes de reintentar.

Si `uv lock --check` pasa, el problema no es el lockfile. Las causas posibles
son:

| Causa | Acción |
| --- | --- |
| Falta el intérprete de Python | Ejecutar `uv python install <versión>` con la versión de `.python-version`. |
| Red, índice o credenciales | Revisar la conexión y la configuración del índice; reintentar. |
| Falla la compilación de una dependencia | Revisar el error; instalar las dependencias del sistema o usar Docker. |
| Permisos, disco o caché | Corregir permisos y espacio; si se sospecha corrupción, ejecutar `uv cache clean` y, si el problema persiste, eliminar `.venv`. |

En un checkout de solo lectura, sin intención de crear un commit,
`uv sync --frozen` instala desde el lockfile sin comprobar si está
actualizado. Es una salida temporal; no corrige la inconsistencia.

## GitHub Actions

### El check `CI success` falla

Abrir el job que falló en la pestaña Actions. Las causas frecuentes son:

| Causa | Acción |
| --- | --- |
| Lockfile desactualizado | Actualizar el lock correspondiente; ver [Docker: actualizar locks](development/docker.md#comandos-para-actualizar-locks). |
| Fallo del healthcheck de Docker | Revisar los logs del job y reproducir el arranque con `docker compose up --build`; ver [Docker](development/docker.md). |

### El job Docker falla al escribir en el bind mount

CI construye la imagen con el UID y el GID del runner. Si se cambia el entorno
de ejecución, alinear esos valores con los del usuario que ejecuta el
contenedor. El detalle de los permisos está en
[esta misma página](#aparece-permission-denied-al-escribir-archivos).
