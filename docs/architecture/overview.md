# Arquitectura

El notebook orquesta y documenta el análisis; el paquete `bank_clients_ml`
implementa la lógica reutilizable del pipeline. Las decisiones de diseño se
registran en [Decisiones de arquitectura](decisions.md).

## Mapa de componentes

```mermaid
flowchart LR
  T[Tests] -.->|Validan| P[Paquete bank_clients_ml]
  N[Notebook Marimo] -->|Orquesta| P
  N -->|Lee| D[(Datos)]
  P -->|Escribe| F[(Gráficos SVG)]
  classDef focus fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A;
  classDef data fill:#F3F4F6,stroke:#6B7280,color:#111827;
  class N,P focus;
  class D,F,T data;
```

*Figura 1. El recorrido
visual está en [Pipeline](pipeline.md).*

## Módulos

```mermaid
flowchart TB
  subgraph S1["Preparación de datos"]
    direction LR
    CL[cleaning.py] ~~~ SA[sampling.py] ~~~ ED[eda.py]
  end
  subgraph S2["Ingeniería de features"]
    direction LR
    CG[column_groups.py] ~~~ FE[feature_engineering.py]
  end
  subgraph S3["Selección de features"]
    direction LR
    RC[redundant_column_filter.py] ~~~ BI[binning.py]
    BT[bivariate_table.py] ~~~ BA[bivariate_analysis.py]
  end
  subgraph S4["Entrenamiento"]
    TR[training.py]
  end
  subgraph S5["Evaluación y presentación"]
    direction LR
    EV[evaluation.py] ~~~ VI[visualization.py] ~~~ MU[marimo_utils.py]
  end
  subgraph S6["Soporte transversal"]
    CF[config.py]
  end
  S1 -->|"Construir features"| S2
  S2 -->|"Seleccionar features"| S3
  S3 -->|"Entrenar el modelo"| S4
  S4 -->|"Evaluar por deciles"| S5
  classDef focus fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A;
  classDef neutral fill:#F3F4F6,stroke:#6B7280,color:#111827;
  class CL,SA,CG,FE,RC,BI,BT focus;
  class BA,TR,EV,VI focus;
  class ED,MU,CF neutral;
```

*Figura 2. Los módulos se agrupan por etapa y por soporte transversal; las
flechas representan el avance del pipeline, no las dependencias de
importación.*

| Módulo | Responsabilidad |
| --- | --- |
| `cleaning.py` | Limpieza inicial de los datos crudos. |
| `sampling.py` | Ventanas temporales de entrenamiento y predicción; muestreo estratificado. |
| `eda.py` | Funciones auxiliares de inspección y descripción de DataFrames. |
| `column_groups.py` | Agrupación de columnas por fuente de negocio. |
| `feature_engineering.py` | Imputación, codificación, variables derivadas de fechas y agregación. |
| `redundant_column_filter.py` | Filtrado de columnas constantes, desbalanceadas o correlacionadas. |
| `binning.py` | Agrupación de variables en bines y codificación por porcentaje de target. |
| `bivariate_table.py` | Construcción de tablas para el análisis bivariado. |
| `bivariate_analysis.py` | Análisis bivariado y selección visual de features. |
| `training.py` | Entrenamiento de LightGBM, ranking por grupos y búsqueda de hiperparámetros. |
| `evaluation.py` | Métricas de clasificación binaria, lift y estadística KS por deciles. |
| `visualization.py` | Generación de gráficos para análisis, selección de features y evaluación. |
| `marimo_utils.py` | Helpers para mostrar resultados en cuadernos Marimo. |
| `config.py` | Configuración central del proyecto; el detalle está en [Configuración](../configuration.md). |

La generación de gráficos y diagramas se rige por
[Convenciones visuales](../development/visuals.md).

## Estrategia de pruebas

Las pruebas unitarias cubren los módulos del paquete; las de integración
validan la carga de datos, la preparación y el pipeline completo. Los marcadores
`slow`, `requires_svg` y `requires_real_env` identifican requisitos específicos
y se definen en `pyproject.toml`. La política de cobertura y los comandos de
validación están en [Comandos de desarrollo](../development/commands.md).

## Limitaciones y próximos pasos

El repositorio cubre el proceso analítico y de experimentación. No incluye un
servicio de inferencia, CI, monitoreo de drift ni registro de modelos.

Los próximos pasos posibles son:

- Batch scoring programado.
- Empaquetado y despliegue del modelo.
- CI para ejecutar las validaciones en cada cambio.
- Monitoreo de drift y alertas sobre la distribución de entradas y resultados.
- Registro de modelos y versionado de artefactos.
