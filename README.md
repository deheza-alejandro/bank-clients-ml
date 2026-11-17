# bank-clients-ml

Pipeline de Machine Learning reproducible para priorizar clientes con mayor
propensión a comprar un paquete financiero.

## Resumen

El proyecto aborda un caso de marketing dirigido: identificar qué clientes
tienen mayor probabilidad de adquirir un paquete específico y optimizar las
campañas del banco. El flujo incluye análisis exploratorio, limpieza,
definición de universo y target, ingeniería de features, selección de
variables, entrenamiento con LightGBM y evaluación.

El repositorio contiene:

- Un notebook de Marimo que documenta el flujo completo.
- Un paquete de Python con la lógica reutilizable.
- Tests unitarios y de integración.
- Un entorno de desarrollo reproducible con Docker o con uv y Bun en local.

## Resultados

Con el conjunto de datos de ejemplo, el decil de mayor propensión obtiene un
lift de 2,4 en entrenamiento y 2,39 en prueba. La estadística KS es de 47,03 y
48,19, respectivamente, con una diferencia de lift de 0,01 entre ambos
conjuntos. Estos valores son ilustrativos y pueden variar si cambia el
conjunto de datos.

## Etapas del pipeline

| Etapa | Resultado |
| --- | --- |
| 1. EDA y limpieza inicial | Dataset consistente y tipado |
| 2. Universo, target y ventanas temporales | Primera compra definida sin fugas de información |
| 3. Feature engineering | Variables de saldos, actividad, tarjetas y recencia |
| 4. ABT y partición entrenamiento/prueba | Tabla analítica base y split estratificado |
| 5. Selección de features | Eliminación de redundancias y ranking por grupos |
| 6. Entrenamiento y evaluación | Ranking por deciles, lift y KS |

La ejecución detallada está en `notebooks/notebook.py`. Las decisiones de
diseño están en [Decisiones de arquitectura](docs/architecture/decisions.md).

## Aspectos de ingeniería

- Lógica reutilizable en `src/bank_clients_ml`, separada del notebook.
- Configuración centralizada con Pydantic Settings.
- Tests unitarios y de integración con una cobertura mínima del 80 %.
- Entorno reproducible con Docker, uv, Bun y lockfiles.
- Calidad verificada con Ruff, Pyrefly, rumdl, cspell y hooks opcionales.
- Decisiones de arquitectura documentadas como ADRs.

## Cómo probarlo

El proyecto se ejecuta con Docker o con uv y Bun en local. Los requisitos y los
pasos de instalación y arranque están en
[Primeros pasos](docs/getting-started.md).

## Estructura del repositorio

```text
src/bank_clients_ml/    Paquete con la lógica del pipeline
notebooks/              Notebook de Marimo y visualizaciones
tests/                  Pruebas unitarias y de integración
scripts/                Entrypoint de Docker y validaciones
data/                   Conjunto de datos de ejemplo
docs/                   Documentación detallada del proyecto
```

## Documentación

| Documento | Contenido |
| --- | --- |
| [Primeros pasos](docs/getting-started.md) | Instalación y arranque con Docker o local |
| [Arquitectura](docs/architecture/overview.md) | Módulos, pruebas y limitaciones |
| [Decisiones de arquitectura](docs/architecture/decisions.md) | ADR-001 a ADR-010 |
| [Configuración](docs/configuration.md) | Variables `ML_*` y archivo `.env` |
| [Comandos de desarrollo](docs/development/commands.md) | Tests, lint, tipos y calidad |
| [Docker](docs/development/docker.md) | Imagen, actualización y locks |
| [Hooks de Git](docs/development/hooks.md) | Validaciones opcionales antes de commit y push |
| [Solución de problemas](docs/troubleshooting.md) | Errores comunes y recuperación |

## Stack tecnológico

| Área | Herramientas |
| --- | --- |
| Datos y modelo | Polars, NumPy, SciPy, scikit-learn, LightGBM, Matplotlib |
| Notebook | Marimo |
| Configuración | Pydantic Settings |
| Calidad | pytest, Ruff, Pyrefly, rumdl, cspell, SVGO |
| Entorno | Python 3.13, uv, Bun, Docker Compose, Lefthook, Commitizen |

## Contribución

Este repositorio se mantiene como un proyecto personal, por lo que no se
aceptan contribuciones externas. Ver [CONTRIBUTING.md](CONTRIBUTING.md).

## Licencia

Este proyecto está bajo los términos de la **Licencia MIT**. Consultar el
archivo [LICENSE](LICENSE) para más detalles. Las dependencias externas
utilizadas mantienen sus respectivas licencias, tales como Apache 2.0.
