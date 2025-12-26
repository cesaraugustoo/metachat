# AIM Agentic Architecture Analysis

## Overview
The `metachat-aim` module implements a flexible agentic framework designed for iterative problem-solving in nanophotonics design. It is built around a core `Agent` abstraction that orchestrates interactions between Large Language Models (LLMs) and specialized tools.

## Core Components

### 1. Agents (`agent/`)
- **Base Agent (`agent/base.py`):**
    - Defines the abstract `Agent` class.
    - Manages model interaction (`_call_model`), tool registry, and tool execution (`_use_tool`).
    - Stores `tool_calls` history.
    - `solve(problem)` is the main entry point (abstract method).
- **Iterative Agent (`agent/cot_iterative.py`):**
    - Implements a Chain-of-Thought (CoT) loop.
    - **Loop Logic:** Continues until `<response>` tags are generated or `max_iterations` (default 20) is reached.
    - **Tooling:** Specifically integrates `NeuralDesignAPI` via XML-like tag parsing (`<tool>neural_design`).
    - **Logging:** Comprehensive JSON logging of every step (user input, thinking, tool calls, errors) to `experiments/logs/`.
    - **State Management:** Maintains conversation history in `messages` list, appending tool outputs as user messages.

### 2. Models (`core/models/`)
- **Abstraction (`core/models/base.py`):**
    - `BaseModel`: Abstract class for LLM backends.
    - `LLMResponse`: Standardized response object (content, tokens).
- **Implementations:**
    - `openai.py`, `anthropic.py`, `llama.py`: Wrappers for respective APIs.

### 3. Tools (`core/tools/`)
- **Abstraction (`core/tools/base.py`):**
    - `BaseTool`: Abstract base for tools.
- **Specific Tools:**
    - `NeuralDesignAPI` (referenced in `cot_iterative.py`): Interface to the surrogate solvers.

## Data Flow
1. **Initialization:** `IterativeAgent` is initialized with a model and tools.
2. **Problem Submission:** User submits a problem via `solve()`.
3. **Loop:**
    - Agent formats messages (System + User + History).
    - Model generates a response.
    - **If `<response>`:** Loop ends, answer returned.
    - **If `<tool>`:** Tool is executed, result is appended to history as a "user" message ("Tool output: ..."), loop continues.
    - **If text only:** Treated as "thinking", appended to history, loop continues.
4. **Logging:** All interactions are persisted to JSON files for audit and debugging.
