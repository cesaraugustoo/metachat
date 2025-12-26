# Technical Debt and Refactoring Proposals

## 1. Identified Technical Debt

### Code Duplication (High Priority)
- **Problem:** The `NeuralDesignAPI` implementation is duplicated (with slight variations) in `metachat-aim/tools/design/api.py` and `web-app/backend/tools/design/api.py`.
- **Impact:** Divergence in functionality; maintenance overhead; bugs fixed in one version might persist in the other.
- **Problem:** The base agent and core model abstractions are duplicated between `metachat-aim/agent/` and `web-app/backend/agent/`.

### Hardcoded Paths & Environment Variables
- **Problem:** Many paths (e.g., `/media/tmp2/metachat/data`) are hardcoded or inconsistent across `config.yaml` and `.env` files.
- **Impact:** Difficult to deploy on new environments or for new users.

### Error Handling & Reliability
- **Problem:** Docker execution in `NeuralDesignAPI` relies on parsing shell output and checking for file existence. It lacks robust retry logic and granular error reporting from inside the container.
- **Impact:** Intermittent failures in long-running simulations can be hard to diagnose for the user.

### State Management
- **Problem:** The `web-app` uses an in-memory `ConversationManager`.
- **Impact:** All session history is lost if the server restarts.

### Missing Abstractions
- **Problem:** Token counting logic is partially implemented or missing (`#TODO` in `openai.py`).
- **Impact:** Inaccurate cost tracking and potential context window overflows.

## 2. Refactoring Proposals

### Proposal 1: Unified Core Library
- **Action:** Extract `agent`, `core/models`, and `tools` into a top-level shared package (e.g., `metachat_core`).
- **Goal:** Single source of truth for the agentic logic and tool interfaces.

### Proposal 2: Robust Docker Orchestration
- **Action:** Move from `subprocess`/`asyncio.create_subprocess_exec` to the official `docker` Python SDK.
- **Goal:** Better control over container lifecycle, volume management, and log streaming.

### Proposal 3: Configuration Management
- **Action:** Centralize all configuration using a tool like `Pydantic Settings` or a unified `config/` directory with hierarchical YAML files.
- **Goal:** Consistent environment-aware configuration for local dev, GPU servers, and Docker.

### Proposal 4: Persistent Session Store
- **Action:** Replace in-memory `ConversationManager` with a lightweight persistent store (e.g., SQLite or Redis).
- **Goal:** Session persistence across server restarts.

### Proposal 5: Centralized Logging & Observability
- **Action:** Standardize the JSON logging format between the standalone AIM agent and the Web App backend.
- **Goal:** Unified analysis of agent performance and design success rates.
