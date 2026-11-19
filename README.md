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

## Pipeline

El pipeline cubre seis etapas, desde el EDA y la limpieza hasta la evaluación
por deciles. La tabla de etapas y los diagramas están en
[Pipeline](docs/architecture/pipeline.md#etapas-del-pipeline).

## Resultados

El detalle de lift, KS y deciles está en
[Resultados del pipeline](docs/architecture/pipeline.md#resultados).

![Deciles en entrenamiento y prueba][deciles]

*Figura. El decil superior mantiene el mayor lift en entrenamiento y prueba.*

## Cómo probarlo

El proyecto se ejecuta con Docker, o con uv y Bun en local. Los requisitos y los
pasos de instalación y puesta en marcha están en
[Primeros pasos](docs/getting-started.md).

## Estructura del repositorio

```text
src/bank_clients_ml/    Paquete con la lógica del pipeline
notebooks/              Notebook de Marimo
tests/                  Pruebas unitarias y de integración
scripts/                Entrypoint de Docker y validaciones
data/                   Conjunto de datos de ejemplo
docs/                   Documentación del proyecto
docs/assets/            Diagramas, gráficos y capturas
```

## Documentación

La documentación completa está en [docs/index.md](docs/index.md).

## Stack tecnológico

| Área | Herramientas |
| --- | --- |
| Datos y modelo | Polars, NumPy, SciPy, scikit-learn, LightGBM, Matplotlib |
| Notebook | Marimo |
| Diagramas | Mermaid, SVG |
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

[deciles]: docs/assets/plots/plot_evaluation_metrics/deciles.svg
