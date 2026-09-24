"""Configuración central del proyecto.

Centraliza los parámetros globales compartidos por casi todos los módulos
del proyecto.

La configuración se expone mediante la clase `Settings` y se accede a
ella con la función `get_settings`, que devuelve una instancia única
almacenada en caché (Singleton).
"""

from functools import cache

# from pydantic import SecretStr
from pydantic_settings import BaseSettings  # , SettingsConfigDict


class Settings(BaseSettings):
    """Parámetros globales de configuración del proyecto.

    Define los valores por defecto utilizados en el proceso de modelado.
    Los nombres de columnas permiten mantener una referencia única y
    consistente en todos los módulos que consumen esta configuración.

    Attributes:
        debug: Indica si el modo de depuración está activado.
        random_state: Semilla utilizada para garantizar reproducibilidad.
        col_id: Nombre de la columna identificadora del cliente.
        col_target: Nombre de la columna target del modelo.
        col_feature: Nombre genérico de la columna de feature.
        col_importance: Nombre genérico de la columna de importancia de variable.
    """

    debug: bool = False
    random_state: int = 314
    col_id: str = "client_id"
    col_target: str = "Target"
    col_feature: str = "Feature"
    col_importance: str = "Importance"
    # api_key: SecretStr  # Keeps the secret hidden in logs and print statements.

    # model_config = SettingsConfigDict(
    #     env_file=".env",
    #     env_file_encoding="utf-8",
    #     extra="ignore",
    # )


@cache
def get_settings() -> Settings:
    """Obtiene la configuración global del proyecto.

    Devuelve una instancia única de `Settings` almacenada en caché (Singleton).

    Returns:
        Instancia compartida de la configuración global.

    Example:
        from bank_clients_ml.config import get_settings

        settings = get_settings()
        print(settings.col_id)
        print(settings.col_target)
    """
    return Settings()
