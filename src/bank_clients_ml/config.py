"""Configuración central del proyecto.

Centraliza los parámetros globales compartidos por casi todos los módulos
del proyecto.

La configuración se expone mediante la clase `Settings` y se accede a
ella con la función `get_settings`, que devuelve una instancia única
almacenada en caché (Singleton).
"""

import functools
import pathlib

import pydantic
import pydantic_settings


class Settings(pydantic_settings.BaseSettings):
    """Parámetros globales de configuración del proyecto.

    Define los valores por defecto utilizados en el proceso de modelado.
    Los nombres de columnas permiten mantener una referencia única y
    consistente en todos los módulos que consumen esta configuración.

    ## Attributes:
        debug: Indica si el modo de depuración está activado.
        random_state: Semilla utilizada para garantizar reproducibilidad.
        col_id: Nombre de la columna identificadora del cliente.
        col_target: Nombre de la columna target del modelo.
        col_month: Nombre de la columna que indica el mes.
        col_first_product_date: Nombre de la columna que indica la fecha de
            registro del primer producto.
        col_last_product_date: Nombre de la columna que indica la fecha de
            registro del ultimo producto.
        col_feature: Nombre genérico de la columna de feature.
        col_importance: Nombre genérico de la columna de importancia de variable.
        col_bin: Nombre genérico de la columna de Bin.
        col_clients: Nombre genérico de la columna de clientes.
        col_target_pct: Nombre genérico de la columna de porcentaje de target,
            calculado dinámicamente de col_target.
        project_dir: Directorio base del proyecto.
        data_dir: Directorio de datos calculado dinámicamente de project_dir.
        images_dir: Directorio de imágenes calculado dinámicamente de project_dir.
    """

    debug: bool = False
    random_state: int = 314
    col_id: str = "client_id"
    col_target: str = "Target"
    col_month: str = "Month"
    col_first_product_date: str = "First_product_dt"
    col_last_product_date: str = "Last_product_dt"
    col_feature: str = "Feature"
    col_importance: str = "Importance"
    col_bin: str = "Bin"
    col_clients: str = "Clients"
    project_dir: pathlib.Path = pathlib.Path(__file__).parent.parent.parent
    # api_key: pydantic.SecretStr  # Keeps the secret hidden in logs and print statements.

    # model_config = pydantic_settings.SettingsConfigDict(
    #     env_file=".env",
    #     env_file_encoding="utf-8",
    #     extra="ignore",
    # )

    @pydantic.computed_field
    @property
    def data_dir(self) -> pathlib.Path:
        """Directorio de datos calculado dinámicamente de project_dir."""
        return self.project_dir / "data"

    @pydantic.computed_field
    @property
    def images_dir(self) -> pathlib.Path:
        """Directorio de imágenes calculado dinámicamente de project_dir."""
        return self.project_dir / "notebooks" / "images"

    @pydantic.computed_field
    @property
    def col_target_pct(self) -> str:
        """Nombre de columna de porcentaje de target calculado dinámicamente."""
        return f"{self.col_target}_pct"


@functools.cache
def get_settings() -> Settings:
    """Obtiene la configuración global del proyecto.

    Devuelve una instancia única de `Settings` almacenada en caché (Singleton).

    ## Returns:
        Instancia compartida de la configuración global.

    ## Example:
        from bank_clients_ml import config

        settings = config.get_settings()
        print(settings.col_id)
        print(settings.col_target)
    """
    return Settings()
