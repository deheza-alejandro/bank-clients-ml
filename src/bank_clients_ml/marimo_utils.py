"""Helpers para visualización de resultados en cuadernos Marimo.

Las funciones están pensadas para uso interactivo en cuadernos Marimo,
donde algunos resultados se anexan directamente a la salida visible.

## Funciones exportadas:
    build_plot_tabs: Crea un componente de pestañas con imágenes SVG.
    get_trainer_cv_results: Muestra los resultados de validación cruzada de un entrenador.
    get_trainers_cv_results: Muestra los resultados de validación cruzada por grupo.
    format_correlations_dict: Formatea correlaciones para su presentación.
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
    """Crea un componente de pestañas con imágenes en Marimo.

    Construye una pestaña por cada entrada del mapeo, cargando la imagen SVG
    correspondiente desde el directorio indicado.

    ## Args:
        renames: Mapeo de nombres de archivo en snake_case a títulos legibles
            para las pestañas.
        directory: Directorio donde se encuentran las imágenes.
        orientation: Orientación de las pestañas.

    ## Returns:
        Componente interactivo de Marimo con las imágenes renderizadas.
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
    """Genera componentes visuales con los resultados de validación cruzada.

    Muestra las combinaciones de hiperparámetros evaluadas, ordenadas por la
    métrica obtenida (`rank_test_score`). Opcionalmente incluye los logs de la
    búsqueda.

    ## Args:
        trainer: Entrenador con los resultados de validación cruzada y los logs
            de búsqueda.
        include_logs: Si es True, añade los logs de búsqueda en formato de texto.

    ## Returns:
        Contenedor vertical de Marimo listo para mostrarse en el cuaderno.
    """
    marimo_items: list[object] = [trainer.cv_results]

    if include_logs:
        marimo_items.extend(
            [mo.md("### Search logs"), mo.md(f"```text\n{trainer.search_logs}\n```")]
        )

    return mo.vstack(marimo_items)


def get_trainers_cv_results(
    group_trainers: training.GroupsLGBMTrainer, include_logs: bool = False
) -> types.MappingProxyType[str, mo.Html]:
    """Genera componentes visuales con los resultados por grupo.

    Aplica la visualización de resultados de validación cruzada a cada
    entrenador del contenedor y organiza la salida por nombre de grupo.

    ## Args:
        group_trainers: Contenedor de entrenadores organizados por grupo.
        include_logs: Si es True, incluye los logs de búsqueda junto a cada
            resultado.

    ## Returns:
        Mapeo inmutable de nombres de grupo a contenedores visuales de Marimo.
    """
    return types.MappingProxyType(
        {
            f"{group.pretty_name} Searcher": get_trainer_cv_results(
                trainer, include_logs
            )
            for group, trainer in group_trainers.items()
        }
    )


def format_correlations_dict(
    correlations_dict: Mapping[str, pl.DataFrame],
) -> types.MappingProxyType[str, pl.DataFrame]:
    """Formatea un diccionario de correlaciones para su presentación.

    Genera etiquetas legibles para cada columna de referencia, de modo que
    los resultados puedan mostrarse directamente como pestañas o secciones
    en el cuaderno.

    ## Args:
        correlations_dict: Mapeo de nombres de columna a tablas con sus
            correlaciones.

    ## Returns:
        Mapeo inmutable de etiquetas legibles a tablas de correlaciones.
    """
    return types.MappingProxyType(
        {
            f"Correlated with {column!r}": dataframe
            for column, dataframe in correlations_dict.items()
        }
    )
