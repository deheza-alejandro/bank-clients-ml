"""Tests unitarios para el módulo de configuración global del proyecto."""

import pathlib

import pytest

from bank_clients_ml import config


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
