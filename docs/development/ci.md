# Integración continua

El workflow de GitHub Actions ejecuta las validaciones del proyecto en cada
push a `main` y en cada pull request. Esta página describe el pipeline y las
decisiones que lo rodean.

Estado actual: `main` admite pushes directos. El workflow se ejecuta, pero los
checks no bloquean la escritura.

## Disparadores

| Evento | Cuándo se ejecuta |
| --- | --- |
| `push` | Cada push a `main`. |
| `pull_request` | Cada pull request hacia `main`. |
| `workflow_dispatch` | Ejecución manual desde la pestaña Actions. |

No se usan filtros por rutas a nivel de workflow: un cambio de documentación
también ejecuta el workflow para que el check requerido no quede pendiente.

## Jobs

| Job | Validación |
| --- | --- |
| `python-quality` | Lockfile y sincronización, formato, lint, tipos, versiones de herramientas y configuración de Lefthook. |
| `markdown` | Lint y formato de Markdown con rumdl. |
| `spelling` | Ortografía con cspell. |
| `tests` | Suite completa de pytest, cobertura mínima del 80 % y publicación del reporte en el job summary. |
| `docker` | Configuración de Compose, build del target `dev` y healthcheck del servicio. |
| `ci-success` | Agrega los resultados; es el check que debe exigirse en la protección de rama. |

Los comandos concretos son los de [Comandos de desarrollo](commands.md). CI usa
`uv sync --locked` y `bun install --frozen-lockfile` para verificar que los
lockfiles estén al día.

`CI success` agrupa los jobs para que la protección de rama dependa de un solo nombre y no
haya que actualizarla cuando se agregue o renombre un job interno.

## Docker

El job valida `compose.yaml`, construye el target `dev` con BuildKit y levanta
el servicio con `docker compose up --wait`, que espera el healthcheck definido
en el archivo. La construcción reutiliza la caché de GitHub Actions.

El arranque comprueba el entrypoint, la instalación del proyecto y el volumen
de `node_modules`. La suite completa no se ejecuta dentro de la imagen: el job
de tests la cubre en el runner y el job Docker se concentra en el entorno de
desarrollo. El motivo del empaquetado diferido está en
[Docker](docker.md#detalles-de-la-imagen).

La construcción usa el UID y el GID del runner para que el bind mount sea
escribible. Solo las ejecuciones de `main` (`push` o `workflow_dispatch`)
publican capas en la caché de GitHub Actions; los pull requests y las
ejecuciones manuales de otras ramas reutilizan la caché existente.

## Caché y reproducibilidad

- uv se instala con la versión fijada en el workflow, alineada con
  `ARG UV_VERSION` del `Dockerfile`.
- Bun se instala con la versión fijada en el workflow, alineada con
  `ARG BUN_VERSION` del `Dockerfile`.
- En CI no se ejecuta `uv lock --check` por separado: `uv sync --locked` ya
  falla si el lockfile no está actualizado.
- El lockfile de Bun se valida con `--frozen-lockfile`.
- Las validaciones que ejecutan herramientas instaladas en el runner consumen
  el entorno que CI acaba de sincronizar: `uv run --no-sync` y
  `bun --no-install` no vuelven a resolver ni instalar dependencias.
- Las actions se fijan a un SHA completo con un comentario de versión.

La actualización de uv requiere modificar el `Dockerfile` y el workflow; si la
nueva versión queda fuera del rango de `uv_build`, también `pyproject.toml`.
`scripts/check_python_versions.py` valida Python y uv. La versión de Bun se
mantiene alineada entre el `Dockerfile` y el workflow, y
[Primeros pasos](../getting-started.md#opción-2-local) declara el mínimo.

## Diagnóstico

Las causas frecuentes de fallo y su resolución están en
[Solución de problemas](../troubleshooting.md#github-actions).
