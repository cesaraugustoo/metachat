from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings as PydanticBaseSettings, SettingsConfigDict

class PathSettings(BaseModel):
    data_dir: str = Field(default="./data", description="Base directory for data")
    checkpoint_dir: Optional[str] = Field(default=None, description="Directory for model checkpoints")
    results_dir: str = Field(default="./results", description="Directory for simulation results")
    log_dir: str = Field(default="./logs", description="Directory for logs")
    materials_db_path: str = Field(default="./tools/material_db/materials.db", description="Path to materials database")

class APISettings(BaseModel):
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API Key")
    together_api_key: Optional[str] = Field(default=None, description="Together AI API Key")
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API Key")
    openai_model_name: str = Field(default="gpt-5.2-2025-12-11", description="Default OpenAI model")
    openrouter_model_name: str = Field(default="google/gemini-2.0-flash-001", description="Default OpenRouter model")

class SolverSettings(BaseModel):
    batch_size: int = Field(default=64, description="Solver batch size")
    gpu_ids: List[int] = Field(default_factory=lambda: [0], description="List of GPU IDs to use")
    feature_size_meter: float = Field(default=4e-8, description="Feature size in meters")

class BaseSettings(PydanticBaseSettings):
    env: str = Field(default="dev", description="Current environment")
    paths: PathSettings = Field(default_factory=PathSettings)
    api: APISettings = Field(default_factory=APISettings)
    solver: SolverSettings = Field(default_factory=SolverSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # Allow overriding via PATHS__DATA_DIR etc.
        extra="ignore"
    )

class DevSettings(BaseSettings):
    env: str = "dev"

class ProdSettings(BaseSettings):
    env: str = "prod"
    # Production overrides could go here, e.g. enforcing safer defaults

class DockerSettings(BaseSettings):
    env: str = "docker"
    # Docker specific defaults, e.g. paths
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Default docker paths if not provided
        if self.paths.data_dir == "./data":
            self.paths.data_dir = "/app/data"
        if self.paths.results_dir == "./results":
            self.paths.results_dir = "/app/results"
