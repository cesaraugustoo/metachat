import os
from functools import lru_cache
from .settings import BaseSettings, DevSettings, ProdSettings, DockerSettings

@lru_cache()
def get_settings() -> BaseSettings:
    """
    Factory function to get the current settings based on APP_ENV environment variable.
    Cached to avoid re-instantiation.
    """
    env = os.getenv("APP_ENV", "dev").lower()
    
    settings_map = {
        "dev": DevSettings,
        "development": DevSettings,
        "prod": ProdSettings,
        "production": ProdSettings,
        "docker": DockerSettings
    }
    
    settings_cls = settings_map.get(env, DevSettings)
    return settings_cls()
