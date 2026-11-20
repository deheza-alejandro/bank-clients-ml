# Documentación

Documentación técnica de bank-clients-ml.

## [Primeros pasos](getting-started.md)

Instalación y puesta en marcha con Docker o en local.

## Arquitectura

- [Pipeline](architecture/pipeline.md): interacción entre componentes, selección de variables
  y resultados.
- [Arquitectura general](architecture/overview.md): módulos, pruebas y
  limitaciones.
- [Decisiones de arquitectura](architecture/decisions.md): formato ADR.

## Desarrollo

- [Comandos útiles](development/commands.md): tests, lint, comprobación
  estática de tipos y calidad.
- [Docker](development/docker.md): imagen, entrypoint, actualización del
  entorno y locks.
- [Integración continua](development/ci.md): workflow, jobs, cache.
- [Hooks de Git](development/hooks.md): validaciones opcionales antes de
  commit y push.

## [Configuración](configuration.md)

Variables de entorno `ML_*` y archivo `.env`.

## [Solución de problemas](troubleshooting.md)

Diagnóstico de errores y cómo resolverlos.
