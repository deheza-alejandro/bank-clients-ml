# Convenciones visuales

Guía para que los diagramas, gráficos y capturas mantengan el mismo lenguaje
visual y se puedan reutilizar desde el README y la documentación.

## Ubicación de los recursos

| Carpeta | Contenido | Edición manual |
| --- | --- | --- |
| `docs/assets/diagrams/` | Diagramas manuales y su exportación | Sí |
| `docs/assets/plots/` | Gráficos SVG generados por el notebook | No |
| `docs/assets/images/` | Capturas e imágenes manuales | Sí |

Los diagramas Mermaid se ubican en la página que los explica. Cada recurso se
mantiene en un solo lugar; los demás documentos que necesiten un mismo recurso
enlazan a esa sección.

No usar `docs_dir = "."`: el sitio debe limitarse a `docs/` para no publicar
archivos ajenos a la documentación. No duplicar assets fuera de `docs/`.

## Regenerar los gráficos

Los gráficos del notebook se regeneran al ejecutarlo; `visualization.py` aplica
SVGO cuando está disponible. Los pasos para ejecutar el notebook están en
[Primeros pasos](../getting-started.md).
