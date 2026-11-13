"""Construcción de instancias de `config.Settings` para los tests.

Expone `build_settings_for_tests`, que crea configuraciones aisladas sin
pasar por `Settings.settings_cached`, para no compartir ni modificar la
instancia cacheada.
"""

import pathlib

from bank_clients_ml import config


def build_settings_for_tests(
    *, env_file: pathlib.Path | None = None, **overrides
) -> config.Settings:
    """Construye una configuración independiente para los tests.

    Crea una instancia nueva de `config.Settings` sin pasar por
    `Settings.settings_cached`, por lo que no comparte ni altera la instancia
    cacheada.

    Bajo pytest, la fixture autouse `clean_ml_environment` elimina las
    variables `ML_*`. Salvo que se indique `env_file`, los campos no
    reemplazados usan los valores por defecto de `config.py`.

    Si se llama esta función a nivel de módulo o dentro de un
    pytest.mark.parametrize, se ejecuta en tiempo de colección, antes de
    `clean_ml_environment`, y podría leer variables de entorno locales
    o el .env. Por eso, mejor construir la configuración dentro del test o de
    una fixture.

    ## Args:
        env_file: Archivo `.env` que se leerá. `None` desactiva la lectura del
            `.env` configurado en `Settings`.
        overrides: Campos de `Settings` que se reemplazan. Los overrides tienen
            prioridad sobre el entorno, `.env` y valores por defecto

    ## Returns:
        Nueva instancia de `config.Settings` con los valores indicados.
    """
    return config.Settings(_env_file=env_file, **overrides)
