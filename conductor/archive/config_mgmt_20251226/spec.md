# Track Spec: Centralized Configuration Management

## Overview
This track aims to replace hardcoded paths and inconsistent environment variable handling across the MetaChat monorepo with a robust, centralized configuration system using `pydantic-settings`. This will ensure environment-aware behavior (local dev vs. GPU server vs. Docker) and improve maintainability and deployment ease.

## Functional Requirements
- **Typed Configuration:** Define Pydantic models for all configuration parameters, providing validation and default values.
- **Environment Awareness:** Support multiple environments (Development, Production/GPU-Server, Docker) using class inheritance.
- **Domain-Specific Settings:** Group configuration into logical domains:
    - **Paths:** Centralize directories for data, checkpoints, results, and logs.
    - **API Keys:** Secure handling of LLM provider keys (OpenAI, Anthropic, Together).
    - **Solver/Hardware:** Configuration for GPU IDs, batch sizes, and physical constants.
- **Environment Variable Overrides:** Allow any setting to be overridden by a corresponding environment variable.
- **Automatic Loading:** Automatically load settings from `.env` files if present.

## Technical Implementation
- **Location:** Implement the configuration logic in `metachat_core/config/`.
- **Primary Tool:** `pydantic-settings` library.
- **Active Environment Selection:** Use an `APP_ENV` (or similar) environment variable to determine which settings subclass to instantiate.

## Acceptance Criteria
- [ ] Hardcoded paths in `web-app/` and `metachat-aim/` are removed and replaced with core configuration calls.
- [ ] Changing an environment variable (e.g., `DATA_DIR`) correctly updates the path used by all system components.
- [ ] Validation errors are raised at startup if required configuration (like API keys) is missing.
- [ ] Documentation is provided on how to add new settings and switch environments.

## Out of Scope
- Migrating non-Python configuration (e.g., frontend JavaScript config) unless directly impacted by backend changes.
- Implementation of a secret management service (e.g., HashiCorp Vault).
