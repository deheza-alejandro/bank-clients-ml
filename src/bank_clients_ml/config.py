from functools import cache

# from pydantic import SecretStr
from pydantic_settings import BaseSettings  # , SettingsConfigDict


class Settings(BaseSettings):
    debug: bool = False
    id: str = "client_id"
    target: str = "Target"
    # api_key: SecretStr  # Mantiene el secreto oculto en logs y prints

    # model_config = SettingsConfigDict(
    #     env_file=".env",
    #     env_file_encoding="utf-8",
    #     extra="ignore",
    # )


@cache
def get_settings() -> Settings:
    return Settings()
