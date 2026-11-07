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


class DirectoryPaths(pydantic.BaseModel):
    """Rutas base de directorios del proyecto.

    Centraliza la resolución de rutas para mantener una referencia única
    y consistente en todos los módulos que acceden a datos y artefactos.

    ## Attributes:
        project: Directorio raíz del proyecto.
        data: Directorio de datos calculado dinámicamente a partir de project.
        images: Directorio de imágenes calculado dinámicamente a partir de project.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    project: pathlib.Path = pathlib.Path(__file__).parent.parent.parent
    raw_data_file_name: str = "example_data.parquet"

    @pydantic.computed_field
    @property
    def raw_data(self) -> pathlib.Path:
        """Ruta completa del parquet de entrada del proyecto.

        Calculado dinámicamente a partir de project.
        """
        return self.project / "data" / self.raw_data_file_name

    @pydantic.computed_field
    @property
    def images(self) -> pathlib.Path:
        """Directorio de imágenes calculado dinámicamente a partir de project."""
        return self.project / "notebooks" / "images"


class DatasetColumns(pydantic.BaseModel):
    """Nombres de columnas del conjunto de datos principal.

    Mantiene una referencia única para las columnas utilizadas en el proceso
    de modelado.

    ## Attributes:
        id: Nombre de la columna identificadora del cliente.
        target: Nombre de la columna target del modelo.
        month: Nombre de la columna que indica el mes.
        first_product_date: Nombre de la columna con la fecha de registro
            del primer producto.
        last_product_date: Nombre de la columna con la fecha de registro
            del último producto.
        package_active: Nombre de la columna que indica si el cliente tiene
            paquete activo.
        credit_card_co_branding: Nombre de la columna que indica si el cliente
            tiene "credit card co branding".
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str = "client_id"
    target: str = "Target"
    month: str = "Month"
    first_product_date: str = "First_product_dt"
    last_product_date: str = "Last_product_dt"
    package_active: str = "Package_Active"
    credit_card_co_branding: str = "CreditCard_CoBranding"


class AnalysisColumns(pydantic.BaseModel):
    """Nombres de columnas genéricas para resultados de análisis.

    ## Attributes:
        feature: Nombre de la columna de feature.
        importance: Nombre de la columna de importancia de variable.
        bin: Nombre de la columna de bin.
        min: Nombre de la columna min.
        max: Nombre de la columna max.
        clients: Nombre de la columna de clientes.
        target_pct: Nombre de la columna de porcentaje de target.
        column_name: Nombre de la columna "Column name".
    """

    model_config = pydantic.ConfigDict(frozen=True)

    feature: str = "Feature"
    importance: str = "Importance"
    bin: str = "Bin"
    min: str = "Min"
    max: str = "Max"
    clients: str = "Clients"
    target_pct: str = "Target_pct"
    column_name: str = "Column name"


class Settings(pydantic_settings.BaseSettings):
    """Parámetros globales de configuración del proyecto.

    Define los valores por defecto utilizados en el proceso de modelado.
    Agrupa rutas y nombres de columnas para mantener una referencia única y
    consistente en todos los módulos.

    ## Attributes:
        debug: Indica si el modo de depuración está activado.
        random_state: Semilla utilizada para garantizar reproducibilidad.
        n_jobs: Cantidad de procesos paralelos para la búsqueda de
            hiperparámetros. El valor -1 utiliza todos los núcleos disponibles.
        paths: Rutas base de directorios del proyecto.
        dataset: Nombres de columnas del conjunto de datos principal.
        analysis: Nombres de columnas genéricas para resultados de análisis.
    """

    debug: bool = False
    random_state: int = 314
    n_jobs: int = -1

    paths: DirectoryPaths = pydantic.Field(default_factory=DirectoryPaths)
    dataset: DatasetColumns = pydantic.Field(default_factory=DatasetColumns)
    analysis: AnalysisColumns = pydantic.Field(default_factory=AnalysisColumns)

    # api_key: pydantic.SecretStr  # Keeps the secret hidden in logs and print statements.

    model_config = pydantic_settings.SettingsConfigDict(
        frozen=True
        # env_file=".env",
        # env_nested_delimiter="__",
        # env_file_encoding="utf-8",
        # extra="ignore",
    )


@functools.cache
def get_settings() -> Settings:
    """Obtiene la configuración global del proyecto.

    Devuelve una instancia única de `Settings` almacenada en caché (Singleton).

    ## Returns:
        Instancia compartida de la configuración global.

    ## Example:
        from bank_clients_ml import config

        settings = config.get_settings()
        print(settings.dataset.id)
        print(settings.dataset.target)
    """
    return Settings()
