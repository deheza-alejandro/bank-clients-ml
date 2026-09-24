"""Se ejecuta como hook `commit-msg` de Lefthook antes de la validación con Commitizen."""

import sys
import textwrap
from pathlib import Path


def format_commit_body(file_path: str) -> None:
    """Formatea el cuerpo del mensaje de commit a 72 caracteres por línea.

    El título se conserva sin cambios y los párrafos del cuerpo se
    reajustan por separado para mantener las líneas en blanco entre ellos.
    Las líneas de comentario que comienzan con `#` se descartan al procesar
    el mensaje. Si el archivo no existe, está vacío o no contiene cuerpo,
    el archivo se deja sin modificar.

    Args:
        file_path: Ruta al archivo temporal con el mensaje de commit.
    """
    commit_file = Path(file_path)

    if not commit_file.is_file():
        return

    content = commit_file.read_text(encoding="utf-8")
    message_lines = [line for line in content.splitlines() if not line.startswith("#")]
    if len(message_lines) < 1:
        return

    message_text = "\n".join(message_lines).strip()
    parts = message_text.split("\n\n", 1)
    if len(parts) < 2:
        return

    title, body = parts[0], parts[1]
    paragraphs = body.split("\n\n")

    wrapped_paragraphs = [
        textwrap.fill(
            paragraph,
            width=72,
            expand_tabs=False,
            replace_whitespace=False,
            drop_whitespace=True,
        )
        for paragraph in paragraphs
    ]
    formatted_body = "\n\n".join(wrapped_paragraphs)
    new_content = f"{title}\n\n{formatted_body}\n"
    commit_file.write_text(new_content, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        format_commit_body(sys.argv[1])
