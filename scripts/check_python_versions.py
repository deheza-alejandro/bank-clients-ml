"""Valida que las versiones declaradas en el proyecto estén alineadas."""

import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import packaging.requirements
import packaging.specifiers
import packaging.version

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
PYTHON_VERSION_FILE = ROOT / ".python-version"
PYPROJECT = ROOT / "pyproject.toml"

ARG_PATTERN = re.compile(
    r"^\s*ARG\s+(?P<name>PYTHON_VERSION|UV_VERSION)"
    r"(?:=(?P<value>[^\s#]+))?\s*(?:#.*)?$",
    re.MULTILINE,
)
SIMPLE_VERSION_PATTERN = re.compile(r"^\d+(?:\.\d+){0,2}$")


def read_text(path: Path) -> str:
    """Lee un archivo de texto o termina con un mensaje claro.

    ## Args:
        path: Ruta del archivo.

    ## Returns:
        El contenido del archivo.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError as err:
        sys.exit(f"Could not read {path}: {err}")


def docker_arg(text: str, name: str) -> str | None:
    """Extrae el valor de un ARG del Dockerfile.

    ## Args:
        text: Contenido completo del Dockerfile.
        name: Nombre del ARG que se quiere leer.

    ## Returns:
        El valor del ARG, o None si no está definido o no tiene valor.
    """
    for match in ARG_PATTERN.finditer(text):
        if match.group("name") == name:
            value = match.group("value")
            return value.strip("\"'") if value else None
    return None


def python_version_file(path: Path) -> str:
    """Lee la primera versión útil de .python-version.

    ## Args:
        path: Ruta del archivo .python-version.

    ## Returns:
        La primera línea que no está vacía ni comentada.
    """
    for line in read_text(path).splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            return value
    sys.exit(f"{path} does not contain a version of Python.")


def load_pyproject(path: Path) -> dict[str, Any]:
    """Carga pyproject.toml.

    ## Args:
        path: Ruta del archivo pyproject.toml.

    ## Returns:
        El contenido del archivo como diccionario.
    """
    try:
        return tomllib.loads(read_text(path))
    except tomllib.TOMLDecodeError as err:
        sys.exit(f"Could not interpret {path}: {err}")


def project_requires_python(pyproject: dict[str, Any]) -> str:
    """Devuelve requires-python o termina si no está definido.

    ## Args:
        pyproject: Contenido de pyproject.toml.

    ## Returns:
        El especificador declarado en project.requires-python.
    """
    value = pyproject.get("project", {}).get("requires-python")
    if not isinstance(value, str) or not value:
        sys.exit("pyproject.toml does not define project.requires-python.")
    return value


def uv_build_requirement(
    pyproject: dict[str, Any],
) -> packaging.requirements.Requirement:
    """Devuelve el requisito de uv_build declarado en build-system.

    ## Args:
        pyproject: Contenido de pyproject.toml.

    ## Returns:
        El requisito de uv_build con su rango de versiones.
    """
    requirements = pyproject.get("build-system", {}).get("requires", [])
    if not isinstance(requirements, list):
        sys.exit("pyproject.toml does not define build-system.requires as a list.")

    for requirement in requirements:
        if not isinstance(requirement, str):
            continue
        try:
            parsed = packaging.requirements.Requirement(requirement)
        except packaging.requirements.InvalidRequirement as err:
            sys.exit(
                f"Invalid requirement in build-system.requires: {requirement!r}: {err}"
            )
        if parsed.name.replace("_", "-").lower() == "uv-build":
            return parsed

    sys.exit("pyproject.toml does not declare uv_build in build-system.requires.")


def validate_python(
    docker_value: str | None,
    file_value: str,
    requires_python: str,
) -> list[str]:
    """Valida las tres declaraciones de Python.

    ## Args:
        docker_value: Valor de ARG PYTHON_VERSION en el Dockerfile.
        file_value: Versión declarada en .python-version.
        requires_python: Especificador de pyproject.toml.

    ## Returns:
        La lista de errores encontrados; vacía si todo está alineado.
    """
    errors: list[str] = []

    if docker_value is None:
        errors.append(
            "Dockerfile: The ARG PYTHON_VERSION is missing or does not have an assigned value."
        )
    elif docker_value != file_value:
        errors.append(
            "Dockerfile: ARG PYTHON_VERSION="
            f"{docker_value} does not match .python-version ({file_value}). "
            "Update both values so that the base image and the UV use the same version"
        )

    if not SIMPLE_VERSION_PATTERN.fullmatch(file_value):
        errors.append(
            f".python-version ({file_value}) is not a simple version like 3.13 o 3.13.5;"
            "this check cannot compare it with requires-python."
        )
        return errors

    try:
        version = packaging.version.Version(file_value)
        specifier = packaging.specifiers.SpecifierSet(requires_python)
    except (
        packaging.version.InvalidVersion,
        packaging.specifiers.InvalidSpecifier,
    ) as err:
        errors.append(f"Could not parse .python-version o requires-python: {err}")
        return errors

    if not specifier.contains(version, prereleases=True):
        errors.append(
            f".python-version ({file_value}) does not satisfy requires-python "
            f"({requires_python}) from pyproject.toml. "
            "Adjust the version or the range."
        )

    return errors


def validate_uv(
    docker_value: str | None, requirement: packaging.requirements.Requirement
) -> list[str]:
    """Valida UV_VERSION contra el rango declarado para uv_build.

    ## Args:
        docker_value: Valor de ARG UV_VERSION en el Dockerfile.
        requirement: Requisito de uv_build declarado en pyproject.toml.

    ## Returns:
        La lista de errores encontrados; vacía si la versión es compatible.
    """
    if docker_value is None:
        return ["Dockerfile: ARG UV_VERSION is missing or has no assigned value."]

    try:
        version = packaging.version.Version(docker_value)
    except packaging.version.InvalidVersion as err:
        return [f"Dockerfile: UV_VERSION={docker_value} it is not valid: {err}"]

    if not requirement.specifier.contains(version, prereleases=True):
        return [
            (
                "Dockerfile: ARG UV_VERSION="
                f"{docker_value} does not satisfy the uv_build constraint "
                f"({requirement}) from pyproject.toml. Update both values."
            )
        ]

    return []


def main() -> None:
    """Ejecuta todas las validaciones y reporta los errores encontrados."""
    dockerfile = read_text(DOCKERFILE)
    python_file = python_version_file(PYTHON_VERSION_FILE)
    pyproject = load_pyproject(PYPROJECT)
    requires_python = project_requires_python(pyproject)
    uv_build = uv_build_requirement(pyproject)

    errors = [
        *validate_python(
            docker_arg(dockerfile, "PYTHON_VERSION"),
            python_file,
            requires_python,
        ),
        *validate_uv(docker_arg(dockerfile, "UV_VERSION"), uv_build),
    ]

    if errors:
        print("Misaligned versions:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        sys.exit(1)

    uv_version = docker_arg(dockerfile, "UV_VERSION")
    print(f"Aligned versions: Python {python_file}, uv {uv_version}.")


if __name__ == "__main__":
    main()
