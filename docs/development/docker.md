# Docker

El proyecto se instala en `/opt/venv` desde el código montado la primera vez
que arranca un contenedor nuevo. `/opt/venv` no es un volumen: cada contenedor
parte de la copia de la imagen y sincroniza si hace falta; al detener el
contenedor se descartan los cambios.

`node_modules` sí es un named volume. En caso de que se reconstruya la imagen,
`node_modules` se actualiza en el siguiente arranque.

## Actualizar el entorno

Los archivos montados con bind mount se actualizan sin reconstruir la imagen.

| Cambio | Acción | ¿Reconstruir? |
| --- | --- | --- |
| `src/`, `tests/`, `notebooks/`, `README.md`, `docs/`, `data/` y configuración de herramientas (excepto `pyproject.toml`) | Nada: están montados con bind mount | No |
| `pyproject.toml`, `uv.lock` | `docker compose restart dev` (o `docker compose up -d` si está detenido) | No obligatorio, pero recomendado |
| `package.json`, `bun.lock` | `docker compose restart dev` (o `docker compose up -d` si está detenido) | No obligatorio, pero recomendado |
| `Dockerfile`, `.dockerignore`, `scripts/docker_entrypoint.py` | `docker compose up --build` | Sí |
| `.python-version`, `PYTHON_VERSION` o imágenes base | Validar versiones (ver [Comandos de desarrollo](commands.md)) y `docker compose build --pull && docker compose up` | Sí |

Para los cambios de locks, reconstruir es opcional para el contenedor actual,
pero recomendado para mantener la imagen reproducible y actualizar la copia
base de `/opt/venv` y `node_modules` para contenedores nuevos. El primer
arranque después de un cambio necesita red si hay paquetes nuevos que
descargar.

Para restablecer el volumen de `node_modules`, ver
[Dependencias y locks](../troubleshooting.md#dependencias-y-locks). Ninguna de
estas operaciones borra `data/` ni `notebooks/images/`, porque forman parte del
montaje del proyecto. Para actualizar la copia base de `/opt/venv`, reconstruir
la imagen.

## Entrypoint

El entrypoint definido en el `Dockerfile` sincroniza las dependencias Python y
JavaScript antes de iniciar un contenedor. Este entrypoint ejecuta
`uv sync --locked` contra `/opt/venv` para Python; si los locks cambiaron,
actualiza el entorno. Para JavaScript, ejecuta `bun install --frozen-lockfile`
en el volumen `node_modules` cuando el hash de `package.json`, `bun.lock` y la
versión de Bun difiere de `node_modules/.bun-install-hash`. Si el entorno ya
está al día, no reinstala nada. Como el entrypoint usa `--locked` y
`--frozen-lockfile`, si los locks no están sincronizados la sincronización
falla; ver [Dependencias y locks](../troubleshooting.md#dependencias-y-locks).

La primera vez que se usa un volumen creado con una versión anterior del
entrypoint, el marker no existe y se ejecuta una instalación completa.

El entrypoint no descarga intérpretes (`UV_PYTHON_DOWNLOADS=never`);
solo sincroniza paquetes con el intérprete de la imagen.

## Comandos para actualizar locks

| Lock | En local | En el contenedor |
| --- | --- | --- |
| `uv.lock` | `uv lock` | `docker compose run --rm --entrypoint dev uv lock` |
| `bun.lock` | `bun install` | `docker compose run --rm --entrypoint dev bun install` |

Los cambios se escriben en el host a través del bind mount; revisar el diff y reiniciar:

```bash
docker compose restart dev
```

Los comandos con `--entrypoint` reemplazan el [entrypoint](docker.md#entrypoint)
y se saltan la sincronización de dependencias y la instalación del proyecto.
Usarlos solo para herramientas que no necesitan `bank_clients_ml`, como
`uv lock` o `bun install`. Para tests o scripts que importan el paquete, ver
[Comandos de desarrollo con Docker](commands.md#comandos-de-desarrollo).

## Detalles de la imagen

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
imagen.
