# Primeros pasos

Guía para ejecutar bank-clients-ml con Docker o en local. En ambos casos se
requiere clonar el repositorio.

## Clonar el repositorio

```bash
git clone <URL-del-repositorio>
cd bank-clients-ml
```

Después, elegir una de las dos opciones.

## Opción 1: Docker

- Docker Engine o Docker Desktop con Docker Compose v2.
- No es necesario instalar Python, uv ni Bun en la máquina host.
- En Windows se recomienda usar WSL2 para el montaje; ver
  [el proyecto está en Windows y el montaje es lento](troubleshooting.md#el-proyecto-está-en-windows-y-el-montaje-es-lento).

### Inicio rápido con Docker

```bash
docker compose up
```

Abrir <http://localhost:2718> en el navegador. La primera vez construye la
imagen y descarga Python, uv, Bun y las dependencias; las ejecuciones
siguientes arrancan en segundos. Marimo se publica únicamente en `127.0.0.1`
(así que no es accesible desde otra máquina de la red) y se ejecuta sin token
de acceso. El contenedor arranca Marimo con `--watch` y recarga el notebook
cuando el archivo se guarda desde el host.

### Detener y eliminar el contenedor

```bash
docker compose down
```

### Ejecutar en segundo plano

```bash
docker compose up -d
```

## Opción 2: Local

- [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.12.1, o una
  versión superior menor a 0.13.0.
- [Bun](https://bun.sh/docs/installation) 1.4 o superior.
- Python 3.13; uv puede instalarlo automáticamente.

### Inicio rápido en local

```bash
uv python install 3.13
uv sync --locked
bun install --frozen-lockfile
uv run marimo edit notebooks/notebook.py
```

`uv sync` instala las dependencias de Python, incluido el grupo de desarrollo.
`bun install` instala SVGO, cspell y el resto de las herramientas de
JavaScript. Marimo abre el notebook en el navegador.

### Modo de solo lectura

```bash
uv run marimo run notebooks/notebook.py
```

### Sin abrir el navegador y con recarga automática

```bash
uv run marimo edit notebooks/notebook.py --headless --watch
```

`--headless` deja la URL en la terminal para abrirla manualmente y `--watch`
aplica los cambios guardados fuera de Marimo.

En Docker y en local, si se usa `--watch` al ejecutar el notebook de Marimo,
evitar editar la misma celda en dos lugares al mismo tiempo.

## Ciclo de vida del contenedor

Los comandos `exec` y `logs` requieren que el servicio esté en ejecución en
segundo plano; ver [Ejecutar en segundo plano](#ejecutar-en-segundo-plano).

| Tarea | Comando |
| --- | --- |
| Abrir una terminal temporal | `docker compose run --rm dev bash` |
| Abrir una terminal | `docker compose exec dev bash` |
| Ver los registros | `docker compose logs -f dev` |

`docker compose exec dev ...` no ejecuta el entrypoint, porque el contenedor
ya está iniciado.

Para aplicar cambios de dependencias, ver
[Docker](development/docker.md#actualizar-el-entorno).
