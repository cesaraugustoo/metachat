# Specification: Centralized User Guide Creation

## Overview
This track addresses the problem of fragmented and outdated project documentation by creating a proper, centralized `USER_GUIDE.md` at the project root. The goal is to provide a single, updated source of truth for installation, setup, and usage, making it easier for new users to deploy and interact with MetaChat.

## Functional Requirements
- **Centralized User Guide:** Create a comprehensive `USER_GUIDE.md` at the project root.
- **Consolidated Setup Instructions:** Migrate and consolidate installation and setup details from component-level READMEs (e.g., `metachat_core`, `web-app`) into the new guide.
- **Dependency Management:** Provide clear instructions for setting up the environment, including Python (Poetry), Node.js (if applicable), and environment variables.
- **Usage Walkthrough:** Include a step-by-step tutorial on running a metasurface design cycle and interacting with the web UI.
- **Component README Simplification:** Refactor existing component READMEs to remove redundant setup information and point users to the centralized guide.

## Non-Functional Requirements
- **Clarity and Conciseness:** The guide should be written in a direct, professional tone suitable for engineers and researchers.
- **Maintainability:** Structure the guide to be easily updatable as the project evolves.
- **Consistency:** Ensure instructions align with the current `tech-stack.md` and `product.md`.

## Acceptance Criteria
- [ ] `USER_GUIDE.md` exists at the project root.
- [ ] Setup instructions cover all necessary dependencies and configuration.
- [ ] A functional walkthrough of the design cycle is included.
- [ ] Component READMEs (`metachat_core/README.md`, `web-app/README.md`, etc.) are updated to point to the new guide.
- [ ] Redundant documentation is removed to prevent future fragmentation.

## Out of Scope
- Detailed API documentation for internal classes (should remain in code/docstrings).
- Architectural mapping (already tracked in `conductor/archive`).
- Rebranding or fundamental guideline changes.
