# Track Plan: Centralized Configuration Management

## Phase 1: Core Configuration Implementation [checkpoint: 7ff8c9e]
- [x] Task: Configure `metachat_core` dependencies c4f798e
    - [ ] Sub-task: Add `pydantic-settings` to `metachat_core/pyproject.toml`
- [x] Task: Implement Settings Models in `metachat_core/config/` a82a80b
    - [ ] Sub-task: Write tests for Settings validation
    - [ ] Sub-task: Define `PathSettings`, `APISettings`, and `SolverSettings`
    - [ ] Sub-task: Implement `BaseSettings` and environment-specific subclasses (`DevSettings`, `ProdSettings`, `DockerSettings`)
- [x] Task: Implement Settings Factory ded1198
    - [ ] Sub-task: Write tests for environment selection logic
    - [ ] Sub-task: Create a factory function to instantiate the correct settings based on `APP_ENV`
- [ ] Task: Conductor - User Manual Verification 'Core Configuration Implementation' (Protocol in workflow.md)

## Phase 2: AIM Integration
- [ ] Task: Migrate `metachat-aim` to Centralized Config
    - [ ] Sub-task: Write integration tests for AIM config loading
    - [ ] Sub-task: Replace legacy environment variable reads and hardcoded paths in `metachat-aim/` with `metachat_core.config`
- [ ] Task: Conductor - User Manual Verification 'AIM Integration' (Protocol in workflow.md)

## Phase 3: Web-App Integration
- [ ] Task: Migrate `web-app` to Centralized Config
    - [ ] Sub-task: Write integration tests for Web-App config loading
    - [ ] Sub-task: Update `web-app/main.py` and backend tools to use `metachat_core.config`
    - [ ] Sub-task: Replace hardcoded paths in `NeuralDesignAPI` and `MaterialDatabaseCLI`
- [ ] Task: Conductor - User Manual Verification 'Web-App Integration' (Protocol in workflow.md)

## Phase 4: Finalization and Documentation
- [ ] Task: Unified Environment Setup
    - [ ] Sub-task: Create template `.env` files for different environments
    - [ ] Sub-task: Document the configuration system and how to switch environments in `README.md` or a new `docs/config.md`
- [ ] Task: Conductor - User Manual Verification 'Finalization and Documentation' (Protocol in workflow.md)
