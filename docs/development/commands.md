# Comandos de desarrollo

Ejecutar los comandos desde la raíz del proyecto.
En local, usar el comando tal cual; con Docker, anteponer:

```bash
docker compose run --rm dev
```

Esto crea un contenedor temporal y reutiliza la imagen y el volumen de `node_modules`.
Por ejemplo:

```bash
docker compose run --rm dev uv run pytest -m "not slow"
```

La cobertura mínima exigida es del 80 %;
el comando de todos los tests la aplica con `--cov-fail-under=80`.

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

## Con Docker

### Interactuar con el contenedor

Los comandos `exec` y `logs` requieren que el servicio esté
[en ejecución en segundo plano](../getting-started.md#ejecutar-en-segundo-plano).

| Tarea | Comando |
| --- | --- |
| Abrir una terminal temporal | `docker compose run --rm dev bash` |
| Abrir una terminal | `docker compose exec dev bash` |
| Ver logs | `docker compose logs -f dev` |

`docker compose exec dev ...` no ejecuta el [entrypoint](docker.md#entrypoint),
porque el contenedor ya está iniciado.

Para aplicar cambios de dependencias, ver [Docker](docker.md#actualizar-el-entorno).
