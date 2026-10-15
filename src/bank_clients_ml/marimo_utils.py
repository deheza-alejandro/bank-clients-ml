"""add docs.

Las funciones están pensadas para uso interactivo en cuadernos Marimo,
donde algunos resultados se anexan directamente a la salida visible.

## Funciones exportadas:
    build_plot_tabs: Crea un componente de pestañas en Marimo mostrando imágenes.
"""

import pathlib
import types
from collections.abc import Mapping
from typing import Literal

import marimo as mo
import polars as pl

from bank_clients_ml import binning, training


def build_plot_tabs(
    renames: binning.Renames,
    directory: pathlib.Path,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
) -> mo.Html:
    """Crea un componente de pestañas en Marimo mostrando imágenes.

    ## Args:
        renames: Mapeo de títulos de pestañas a nombres de archivo.
        subdirectory: Directorio donde se encuentran las imágenes.
        orientation: Orientación de las pestañas ('horizontal' o 'vertical').
            Por defecto es 'horizontal'.

    ## Returns:
        Componente interactivo de marimo con las imágenes renderizadas.
    """
    return mo.ui.tabs(
        {
            pretty_name: mo.image(src=directory / f"{snake_case_name}.svg")
            for snake_case_name, pretty_name in renames.items()
        },
        orientation=orientation,
    )


def get_trainer_cv_results(
    trainer: training.LGBMTrainer, include_logs: bool = False
) -> mo.Html:
    """Retorna componentes de marimo para visualizar los resultados del entrenamiento.

    Permite visualizar las combinaciones de hiperparámetros de los modelos
    probados con validación cruzada, ordenados por la métrica obtenida
    (`rank_test_score`). Opcionalmente permite ver los logs.

    ## Args:
        include_logs: Si es True, incluye los logs de búsqueda en formato de texto.

    ## Returns:
        elementos visuales de marimo listos para mostrarse en el notebook.
    """
    marimo_items: list[object] = [trainer.cv_results]

    if include_logs:
        marimo_items.extend(
            [mo.md("### Search logs"), mo.md(f"```text\n{trainer.search_logs}\n```")]
        )

    return mo.vstack(marimo_items)


def get_trainers_cv_results(
    groups_trainer: training.GroupsLGBMTrainer, include_logs: bool = False
) -> types.MappingProxyType[str, mo.Html]:
    """Retorna componentes de marimo con las combinaciones de cada grupo.

    ## Args:
        include_logs: Si es True, incluye los logs de búsqueda
            junto a cada buscador en formato de texto.

    ## Returns:
        Un diccionario con elementos visuales de marimo organizados por
        nombre de grupo, listos para mostrarse en el notebook.
    """
    return types.MappingProxyType(
        {
            f"{group.pretty_name} Searcher": get_trainer_cv_results(
                trainer, include_logs
            )
            for group, trainer in groups_trainer.items()
        }
    )


def format_correlations_dict(
    correlations_dict: Mapping[str, pl.DataFrame],
) -> types.MappingProxyType[str, pl.DataFrame]:
    """Add docs."""
    return types.MappingProxyType(
        {
            f"Correlated with {column!r}": dataframe
            for column, dataframe in correlations_dict.items()
        }
    )
