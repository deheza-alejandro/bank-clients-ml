# Comandos de desarrollo

Ejecutar los comandos desde la raíz del proyecto. En local, usar el comando tal
cual; con Docker, anteponer `docker compose run --rm dev`, que crea un
contenedor temporal y reutiliza la imagen y el volumen de `node_modules`. El
entrypoint sincroniza las dependencias Python y JavaScript antes de ejecutar el
comando; ver [Docker: actualizar el entorno](docker.md#actualizar-el-entorno).
Por ejemplo:

```bash
docker compose run --rm dev uv run pytest -m "not slow"
```

Los comandos con `--entrypoint` reemplazan el entrypoint y se saltan la
sincronización de dependencias y la instalación del proyecto. Usarlos solo para
herramientas que no necesitan `bank_clients_ml`, como `uv lock` o
`bun install`. Para tests o scripts que importan el paquete, usar
`docker compose run --rm dev ...`.

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
