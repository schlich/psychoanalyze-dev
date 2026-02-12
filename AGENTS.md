# AGENTS.md

## Project Overview
PsychoAnalyze is a Python library for interactive data simulation and analysis in psychophysics research. It models psychometric functions using logistic regression to estimate detection thresholds.

## Architecture
- **Three Pillars:** Data processing (deterministic transforms), Interactive dashboard (Marimo UI), and Python package (stable API/CLI).
- **Data Hierarchy:** Trials (raw table) -> Points (aggregated) -> Blocks (fitted curves) -> Sessions/Subjects.
- **Core Library:** `src/psychoanalyze/` (data manipulation, analysis, plotting).
- **App:** `app.py` (Marimo dashboard).
- **Models:** `models/` (dbt/Stan models).

## Development Environment
- **Shell:** Nushell is the default shell.
- **Package Management:** `uv` (do not use pip directly).
- **Tools:** Prefer MCP tools (`mcp_jj_*`, `mcp_marimo_*`, `runTests`) over shell commands.
- **Python Setup:** Always call `configure_python_environment` before using Python tools.

## Build and Test Commands
- **Install dependencies:** `uv sync`
- **Run tests:** `uv run pytest` (writes Allure results to `allure-results/`)
- **Run test watcher:** `uv run ptw . --now`
- **Lint and format:** `uv run ruff check --fix && uv run ruff format`
- **Type check:** `uv run ty check`
- **Run dashboard:** `uv run marimo edit app.py`

## Code Style & Conventions
- **Type Annotations:** Use broad input types, narrow output types. Use built-in generics (`list[str]`) and unions (`|`).
- **Data Handling:** Prefer Polars for core transforms; Pandas for presentation boundaries.
- **Column Names:** Standardize on `Result` (0/1), `Hits`, `Hit Rate`, `n trials`.
- **Plotting:** Use Plotly Express with global template `plot.template`.
- **Testing:** Mirror source structure in `tests/`. Use pytest fixtures for common data.

## Workflow & Agents
This project uses custom agents and workflows defined in `.github/`.

### TDD Workflow
- **Red Phase (`tdd-red`):** Write exactly ONE failing test for ONE behavior. Verify failure.
- **Green Phase (`tdd-green`):** Implement minimal code to pass the test. No refactoring.
- **Refactor Phase (`tdd-refactor`):** Improve code quality while keeping tests green.
- **Orchestrator (`tdd-cycle`):** Runs one full red/green cycle.

### Version Control (Jujutsu)
- **Tooling:** Use `mcp_jj_*` tools.
- **Commits:** Use conventional commit messages (`feat`, `fix`, `refactor`).
- **Revisions:** Use `jj absorb` for small fixes/typos in previous revisions. Use `jj-parallel-splitter` to split oversized revisions.
- **Review:** Use `jj evolog` to review revision evolution.

### Marimo Notebooks
- **Editing:** Use `mcp_marimo_*` tools.
- **Validation:** Check for lint errors (breaking, runtime, formatting) before finishing.

## Documentation
- **Gherkin Features:** Describe behavior, not UI steps. Keep steps reusable and atomic.
- **Specs:** See `COPILOT_SPECS.md` for detailed specifications of the custom copilot environment.
