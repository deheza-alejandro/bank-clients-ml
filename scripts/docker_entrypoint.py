"""Mantiene las dependencias de JavaScript y Python sincronizadas."""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

MARKER = Path("node_modules/.bun-install-hash")
HASH_VERSION = b"1"


def find_bun() -> str:
    """Devuelve la ruta absoluta de Bun o termina si no está en PATH."""
    bun = shutil.which("bun")
    if bun is None:
        sys.exit("Could not find bun in PATH.")
    return bun


def find_uv() -> str:
    """Devuelve la ruta absoluta de uv o termina si no está en PATH."""
    uv = shutil.which("uv")
    if uv is None:
        sys.exit("Could not find uv in PATH.")
    return uv


def current_hash(root: Path) -> str | None:
    """Calcula el hash de las entradas que afectan a node_modules.

    ## Args:
        root: Raíz del proyecto que contiene los manifiestos.

    ## Returns:
        El hash hexadecimal, o None si faltan los manifiestos.
    """
    package = root / "package.json"
    lock = root / "bun.lock"
    if not package.is_file() or not lock.is_file():
        return None

    bun = find_bun()

    try:
        completed = subprocess.run(  # noqa: S603
            [bun, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as err:
        sys.exit(f"Could not execute bun --version: {err}")

    bun_version = completed.stdout.strip()
    digest = hashlib.sha256()
    digest.update(HASH_VERSION + b"\0")
    digest.update(bun_version.encode() + b"\0")
    for path in (package, lock):
        digest.update(path.name.encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def write_marker(root: Path, value: str) -> None:
    """Escribe el hash de la instalación actual.

    ## Args:
        root: Raíz del proyecto.
        value: Hash calculado por `current_hash`.
    """
    marker = root / MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{value}\n", encoding="utf-8")


def ensure_dependencies(root: Path) -> None:
    """Ejecuta bun install solo cuando cambió el hash de dependencias.

    ## Args:
        root: Raíz del proyecto.
    """
    value = current_hash(root)
    if value is None:
        return

    marker = root / MARKER
    previous = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    if value == previous:
        return

    bun = find_bun()
    print(
        "package.json/bun.lock have changed; running bun install...",
        file=sys.stderr,
    )
    try:
        subprocess.run(  # noqa: S603
            [bun, "install", "--frozen-lockfile"],
            cwd=root,
            check=True,
        )
    except subprocess.CalledProcessError as err:
        sys.exit(err.returncode)

    write_marker(root, value)


def ensure_python_dependencies(root: Path) -> None:
    """Sincroniza /opt/venv con pyproject.toml y uv.lock.

    Se ejecuta en cada arranque; uv no reinstala nada si el entorno ya está al
    día. `--locked` evita que uv modifique el lockfile del host.

    ## Args:
        root: Raíz del proyecto que contiene los manifiestos.
    """
    if not (root / "pyproject.toml").is_file():
        return
    if not (root / "uv.lock").is_file():
        return

    try:
        subprocess.run(  # noqa: S603
            [find_uv(), "sync", "--locked"],
            cwd=root,
            check=True,
        )
    except subprocess.CalledProcessError as err:
        sys.exit(err.returncode)


def main() -> None:
    """Ejecuta la comprobación y, si corresponde, el comando recibido."""
    root = Path.cwd()
    args = sys.argv[1:]

    if args[:1] == ["--write-marker"]:
        if not (root / "node_modules").is_dir():
            sys.exit("node_modules does not exist; you need to run bun install")
        value = current_hash(root)
        if value is None:
            sys.exit("package.json or bun.lock are missing")
        write_marker(root, value)
        return

    ensure_dependencies(root)
    ensure_python_dependencies(root)

    if args:
        try:
            # S606: execvp replaces this process so that Docker signals
            # reach the actual command; a shell is intentionally not used.
            os.execvp(args[0], args)  # noqa: S606
        except FileNotFoundError as err:
            sys.exit(f"Command not found {args[0]!r}: {err}")


if __name__ == "__main__":
    main()
