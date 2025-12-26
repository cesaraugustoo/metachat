# Configuration Management

MetaChat uses a centralized configuration system powered by `pydantic-settings`. This ensures that all components (AIM, Web-App, Core) share a consistent and validated set of settings.

## Environment Selection

The configuration is environment-aware. You can switch environments by setting the `APP_ENV` environment variable:

- `dev` (default): Local development settings.
- `prod`: Production/GPU-Server settings.
- `docker`: Settings optimized for running inside Docker containers.

```bash
export APP_ENV=prod
```

## Configuration Files

The system automatically loads variables from a `.env` file at the project root. Use `.env.example` as a template.

## Overriding Settings

Any setting can be overridden using environment variables. For nested settings, use double underscores (`__`) as a delimiter.

### Examples:

- **Top-level setting:** `export APP_ENV=prod`
- **Nested path setting:** `export PATHS__DATA_DIR=/custom/data/path`
- **Nested solver setting:** `export SOLVER__GPU_IDS=[0,1,2]`
- **API Key:** `export OPENAI_API_KEY=sk-...`

## Configuration Domains

Settings are grouped into logical domains:

- **`paths`**: Data, checkpoints, results, and logs directories.
- **`api`**: LLM provider API keys and default model names.
- **`solver`**: GPU configuration, batch sizes, and physical constants.

## Implementation Details

The configuration logic resides in `metachat_core/config/`. It uses Pydantic models for validation, ensuring that missing required keys or invalid types are caught at startup.
