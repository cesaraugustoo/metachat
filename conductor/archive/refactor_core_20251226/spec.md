# Track Spec: Refactor Core Architecture

## Goal
To eliminate code duplication and architectural divergence by extracting the common agentic logic, model wrappers, and tool interfaces into a unified, shared Python package named `metachat_core`.

## Scope
- **Source Directories:** `metachat-aim/` and `web-app/backend/`.
- **Target Package:** Create a new top-level package `metachat_core/`.
- **Components to Migrate:**
    - **Agents:** `Agent` base class, `IterativeAgent`.
    - **Models:** `BaseModel`, `OpenAIModel`, etc.
    - **Tools:** `BaseTool`, `NeuralDesignAPI` (shared logic).
- **Refactoring:** Update both the standalone AIM experiments and the Web App backend to import from `metachat_core` instead of local copies.

## Deliverables
- A pip-installable `metachat_core` package (managed via Poetry).
- Updated `metachat-aim` utilizing the core package.
- Updated `web-app` utilizing the core package.
- Removal of duplicated code in `metachat-aim/agent`, `metachat-aim/core`, `web-app/backend/agent`, and `web-app/backend/core`.
