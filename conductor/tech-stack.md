# Technology Stack: MetaChat

## Core Languages
- **Python (>=3.10):** The primary language for the agentic framework, scientific solvers, and backend services.
- **HTML/CSS/JavaScript:** Used for the interactive web-based design interface.

## Backend and API
- **FastAPI & Uvicorn:** Provides a high-performance asynchronous web server and API for the web application.
- **LLM Integrations:** Support for OpenAI, Anthropic, and Llama APIs to power the agentic iterative monologue.

## Scientific Computing and Simulation
- **PyTorch:** Used for building and executing the FiLM WaveY-Net surrogate solvers.
- **NumPy & SciPy:** Foundation for numerical computations and scientific algorithms.
- **SymPy:** Utilized for symbolic math operations.
- **Matplotlib:** Used for generating scientific plots and visualizations.

## Data Management
- **Pandas & PyArrow:** For efficient data manipulation and storage (e.g., Parquet).
- **SQLite (aiosqlite):** Used for managing persistent data, such as design histories and material databases.

## Infrastructure and Tools
- **Poetry:** Manages dependencies and packaging for the Python projects.
- **Git:** Version control system for the entire monorepo.
