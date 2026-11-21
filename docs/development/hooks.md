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
  configuración de Docker Compose y la de Lefthook.
- Push: en cada push se sincronizan las dependencias de Python y JavaScript; después se
  ejecutan las validaciones que correspondan: Pyrefly y todos los tests de
  pytest (unitarios + integración), el chequeo de
  versiones, y cspell.
- Checkout, merge y rewrite: se sincronizan uv y Bun cuando cambian
  `pyproject.toml`/`uv.lock` o `package.json`/`bun.lock`. Los flags `--locked`
  y `--frozen-lockfile` no modifican los lockfiles y permiten detectar
  inconsistencias en lugar de corregirlas automáticamente.

El diagnóstico de lockfiles y sincronización está en
[Solución de problemas](../troubleshooting.md#sincronización-fallida-del-entorno).

Los hooks de commit asumen que el entorno ya está instalado. En `pre-push`,
la sincronización es un paso previo: las validaciones siguientes se ejecutan con
`uv run --no-sync` y `bun --no-install` para que las herramientas no resuelvan
ni instalen dependencias por su cuenta, y `--no-progress` solo suprime las
barras de progreso. Así, el resultado refleja el entorno bloqueado y no un
estado resuelto en el momento.

El hook `python-versions` ejecuta `scripts/check_python_versions.py`
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
commit o un push, incluida la sincronización con los lockfiles en el push. No
reemplazan al workflow de CI: la referencia completa de los jobs y del check
requerido está en [Integración continua](ci.md).
