# Web App Architecture Analysis

## Overview
The `web-app` module provides a browser-based interface for interacting with the MetaChat system. It consists of a FastAPI backend that hosts the agentic stack and a lightweight HTML/JS frontend for real-time chat and visualization.

## Backend (`web-app/main.py`)

### Framework & Configuration
- **Server:** FastAPI with Uvicorn.
- **Protocol:** Uses Server-Sent Events (SSE) via `sse_starlette` for streaming agent responses to the client.
- **State Management:** In-memory `ConversationManager` handles session history.
- **Configuration:** Loads environment variables (API keys, paths) via `python-dotenv`.

### Agent Integration
- **Model:** Initializes `OpenAIModel` (default: `gpt-5.2-2025-12-11`).
- **Agent:** Uses `IterativeAgentToolsMaterials`, a specialized implementation of the iterative agent that includes access to:
    - **Neural Design Tools:** For running the FiLM WaveY-Net surrogate solver.
    - **Material Database:** `MaterialDatabaseCLI` for querying material properties.
    - **Scientific Solvers:** `ScientificCompute` and `SymbolicSolver`.

### API Structure
- **Endpoints:**
    - **Chat:** Accepts `ChatRequest` (message, conversation_id) and streams the agent's thought process and final response.
    - **Static Files:** Serves the frontend assets.
    - **Ping:** Health check endpoint (implied).

## Frontend (`web-app/frontend/`)

### Architecture
- **Type:** Single Page Application (SPA).
- **Tech:** Vanilla HTML/CSS/JavaScript.
- **Dependencies:**
    - **MathJax:** For rendering LaTeX equations.
    - **Config:** `config.js` for API endpoint configuration.

### Interaction Flow
1. **User Input:** User sends a message via the chat interface.
2. **Streaming:** The frontend opens an SSE connection (or streams the response).
3. **Visualization:**
    - Text responses are rendered in the chat window.
    - LaTeX is rendered by MathJax.
    - Tool outputs (e.g., design plots) are likely rendered as images or interactive elements (based on the `matplotlib` dependency in the backend).

## Directory Structure (`web-app/backend/`)
The backend contains a self-contained version of the agentic stack:
- `agent/`: Contains the specific agent implementations for the web app (`cot_iterative_tools_materials.py`).
- `core/`: Core model wrappers.
- `tools/`: Tool implementations, including the design API and material DB.
