# Hooks de Git

Los hooks no se instalan solos. Si se clona el repositorio para desarrollar o
hacer un fork, se pueden activar para mantener la consistencia y detectar
problemas antes de subir cambios. Ruff y rumdl trabajan solo sobre los archivos
agregados al staging; la parte más pesada es pytest.

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

- Commit: se comprueban los lockfiles si cambian; se ejecutan las validaciones
  de Python y Markdown, y se comprueba el mensaje de commit; se validan la
  configuración de Docker Compose y la de Lefthook. Los comandos concretos
  están en [Comandos de desarrollo](commands.md).
- Push: se ejecuta Pyrefly, el chequeo de versiones de Python y uv, todos los
  tests de pytest (unitarios + integración) y cspell. Los comandos concretos
  están en [Comandos de desarrollo](commands.md).
- Checkout, merge y rewrite: se sincroniza el entorno con `uv sync --locked`,
  que no modifica el lockfile y permite detectar inconsistencias en lugar de
  corregirlas automáticamente. Si falla, ver
  [Solución de problemas](../troubleshooting.md#sincronización-fallida-tras-checkout-merge-o-rewrite).

El hook `python-versions-check` ejecuta `scripts/check_python_versions.py`
cuando cambian el Dockerfile, `.python-version`, `pyproject.toml` o el propio
script. Si falla, corregir los valores para que `ARG PYTHON_VERSION`,
`.python-version`, `requires-python` y `UV_VERSION` queden alineados con el
rango de `uv_build`.

Para desactivarlos:

```bash
uv run lefthook uninstall
```

## Relación con CI

Los hooks son opcionales y buscan anticipar las mismas validaciones antes de un
commit o un push. No reemplazan al workflow de CI: la referencia completa de los jobs
y del check requerido está en [Integración continua](ci.md).
