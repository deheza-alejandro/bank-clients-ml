"""Tests para la configuración global del proyecto."""

import pathlib
from collections.abc import Iterator

import pydantic
import pytest

from bank_clients_ml import config


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Borrar la caché de la configuración global antes y después de cada prueba."""
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
    """Obtener ambas rutas calculadas a partir del proyecto configurado."""
    paths = config.DirectoryPaths(project=project)

    data_path = paths.data
    images_path = paths.images

    assert data_path == project / "data"
    assert images_path == project / "notebooks" / "images"


def test_should_resolve_paths_from_project_and_not_from_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Resuelve las rutas a partir del proyecto.

    Incluso cuando el directorio de trabajo actual (cwd) cambie.
    """
    project = tmp_path / "project"
    monkeypatch.chdir(tmp_path)
    paths = config.DirectoryPaths(project=project)

    data_path = paths.data

    assert data_path == project / "data"


def test_should_return_cached_settings_instance() -> None:
    """Devuelve la misma instancia de configuración almacenada en caché ante llamadas repetidas."""
    first_settings = config.get_settings()

    second_settings = config.get_settings()

    assert second_settings is first_settings


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
    """Rechazar los cambios en los modelos de configuración inmutables."""
    model = model_factory()

    with pytest.raises(pydantic.ValidationError):
        setattr(model, attribute, new_value)
