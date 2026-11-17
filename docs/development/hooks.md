# Hooks de Git

Los hooks no se instalan solos. Si se clona el repositorio para desarrollar o
hacer un fork, se pueden activar para mantener la consistencia y detectar
problemas antes de subir cambios. El stack de validación es rápido: uv, Ruff,
Pyrefly y rumdl están escritos en Rust, y cspell y SVGO corren con Bun. Ruff y
rumdl trabajan solo sobre los archivos agregados al staging; la parte más
pesada es pytest.

Requieren usar el proyecto en local (ver
[Primeros pasos](../getting-started.md#opción-2-local)), ya que dependen de uv
y Bun. Para probar el proyecto con Docker, se puede omitir esta sección y
realizar las validaciones manualmente con los comandos de
[Comandos de desarrollo](commands.md).

Para instalarlos:

```bash
uv run lefthook install
```

Con los hooks activos:

- Commit: se comprueban los lockfiles si cambian; se formatean y revisan con
  Ruff los archivos Python; se ejecuta pytest sin los tests `slow` con
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
