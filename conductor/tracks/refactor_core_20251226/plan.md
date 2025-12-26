# Track Plan: Refactor Core Architecture

## Phase 1: Package Scaffolding [checkpoint: 51dd3e3]
- [x] Task: Create `metachat_core` directory structure and `pyproject.toml` 3d1a964
- [x] Task: Configure shared dependencies in the new package b2dec94
- [ ] Task: Conductor - User Manual Verification 'Package Scaffolding' (Protocol in workflow.md)

## Phase 2: Component Migration [checkpoint: 04ba525]
- [x] Task: Migrate Core Models (`core/models/`) to `metachat_core` 80f92a0
- [x] Task: Migrate Base Agent and Tool interfaces to `metachat_core` 36d6051
- [x] Task: Migrate `NeuralDesignAPI` shared logic to `metachat_core` 2a0c78f
- [ ] Task: Conductor - User Manual Verification 'Component Migration' (Protocol in workflow.md)

## Phase 3: Consumer Integration & Cleanup [checkpoint: f54f22f]
- [x] Task: Refactor `metachat-aim` to use `metachat_core` e8be5ce
- [x] Task: Refactor `web-app` to use `metachat_core` 8a564e2
- [x] Task: Remove duplicated code from legacy directories 9f8fa7e
- [ ] Task: Conductor - User Manual Verification 'Integration and Cleanup' (Protocol in workflow.md)
