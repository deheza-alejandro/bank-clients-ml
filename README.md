# bank-clients-ml

Pipeline de Machine Learning reproducible para priorizar clientes con mayor
propensión a comprar un paquete financiero.

## Resumen

El proyecto aborda un caso de marketing dirigido: identificar qué clientes
tienen mayor probabilidad de adquirir un paquete específico y optimizar las
campañas del banco.

El repositorio contiene:

- Un notebook de Marimo que documenta el pipeline completo.
- Un paquete de Python con la lógica reutilizable en `src/bank_clients_ml`,
  separada del notebook.
- Tests unitarios y de integración con una cobertura mínima del 80 %.
- Un entorno de desarrollo reproducible con Docker, o con uv y Bun en local.
- Configuración centralizada con Pydantic Settings.
- Calidad verificada con Ruff, Pyrefly, rumdl, cspell y hooks opcionales.
- Decisiones de arquitectura documentadas como ADRs.

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
| 2. Universo, target y ventanas temporales | Primera compra definida sin fugas de información temporales |
| 3. Feature engineering | Variables derivadas de saldos, operaciones, tarjetas y recencia |
| 4. ABT y partición entrenamiento/prueba | Tabla analítica base y split estratificado |
| 5. Selección de features | Eliminación de redundancias y ranking por grupos |
| 6. Entrenamiento y evaluación | Evaluación por deciles, lift y KS |

La ejecución detallada está en `notebooks/notebook.py`. Las decisiones de
diseño están en [Decisiones de arquitectura](docs/architecture/decisions.md).

## Cómo probarlo

El proyecto se ejecuta con Docker, o con uv y Bun en local. Los requisitos y los
pasos de instalación y puesta en marcha están en
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
| [Primeros pasos](docs/getting-started.md) | Instalación y puesta en marcha con Docker o en local |
| [Arquitectura](docs/architecture/overview.md) | Módulos, pruebas y limitaciones |
| [Decisiones de arquitectura](docs/architecture/decisions.md) | ADR-001 a ADR-010 |
| [Configuración](docs/configuration.md) | Variables de entorno `ML_*` y archivo `.env` |
| [Comandos de desarrollo](docs/development/commands.md) | Tests, lint, comprobación estática de tipos y calidad |
| [Docker](docs/development/docker.md) | Imagen, entrypoint, actualización del entorno y locks |
| [Hooks de Git](docs/development/hooks.md) | Validaciones opcionales antes de commit y push |
| [Solución de problemas](docs/troubleshooting.md) | Diagnóstico de errores y cómo resolverlos |

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
