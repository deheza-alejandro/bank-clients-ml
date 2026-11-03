"""Tests unitarios para el módulo de configuración global del proyecto."""

import pathlib
from collections.abc import Iterator

import pydantic
import pytest

from bank_clients_ml import config


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Limpia la caché de la configuración global antes y después de cada test."""
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.mark.parametrize(
    "project",
    [pathlib.Path("repo"), pathlib.Path("repo/nested")],
    ids=["simple", "nested"],
)
def test_should_derive_data_and_images_paths_from_project(
    project: pathlib.Path,
) -> None:
    """Verifica que las rutas de datos e imágenes se deriven correctamente a partir del proyecto."""
    paths = config.DirectoryPaths(project=project)

    data_path = paths.data
    images_path = paths.images

    assert data_path == project / "data"
    assert images_path == project / "notebooks" / "images"


def test_should_resolve_paths_from_project_and_not_from_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Verifica que las rutas se resuelvan desde el directorio del proyecto.

    Y no desde el directorio de trabajo (cwd).
    """
    project = tmp_path / "project"
    monkeypatch.chdir(tmp_path)
    paths = config.DirectoryPaths(project=project)

    data_path = paths.data

    assert data_path == project / "data"


def test_should_return_cached_settings_instance() -> None:
    """Verifica se devuelva la misma instancia ante llamadas repetidas."""
    first_settings = config.get_settings()

    second_settings = config.get_settings()

    assert second_settings is first_settings


def test_should_use_all_cores_for_parallel_search_by_default() -> None:
    """Verifica que la búsqueda use todos los núcleos por defecto."""
    assert config.Settings().n_jobs == -1


@pytest.mark.parametrize(
    ("model_factory", "attribute", "new_value"),
    [
        (config.DirectoryPaths, "project", pathlib.Path("other-project")),
        (config.DatasetColumns, "id", "other_id"),
        (config.AnalysisColumns, "feature", "Other_feature"),
        (config.Settings, "debug", True),
    ],
    ids=["directory-paths", "dataset-columns", "analysis-columns", "settings"],
)
def test_should_reject_mutation_of_frozen_configuration_models(
    model_factory: type[pydantic.BaseModel],
    attribute: str,
    new_value: object,
) -> None:
    """Verifica que se rechacen los intentos de mutación en modelos de configuración inmutables."""
    model = model_factory()

    with pytest.raises(pydantic.ValidationError):
        setattr(model, attribute, new_value)
