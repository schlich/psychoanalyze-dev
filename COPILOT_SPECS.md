# PsychoAnalyze Copilot Specifications

This document translates the behavior of the custom GitHub Copilot files found in `.github/` into readable specifications. It covers the project's instructions, custom agents, and specific prompt behaviors.

## 1. Project Overview & Instructions (`copilot-instructions.md`)

**PsychoAnalyze** is a Python library for interactive data simulation and analysis in psychophysics research. It models psychometric functions using logistic regression to estimate detection thresholds.

### Architecture
- **Three Pillars:** Data processing (deterministic transforms), Interactive dashboard (Marimo UI), and Python package (stable API/CLI).
- **Data Hierarchy:** Trials -> Points -> Blocks -> Sessions/Subjects.
- **Key Components:**
    - `src/psychoanalyze/`: Core library (data manipulation, analysis, plotting).
    - `app.py`: Marimo dashboard.
    - `models/`: dbt/Stan models.

### Development Workflow
- **Shell:** Nushell is the default shell.
- **Tools:** Prefer MCP tools (`mcp_jj_*`, `mcp_marimo_*`, `runTests`) over shell commands. Always call `configure_python_environment` before Python tools.
- **Package Management:** Use `uv` (not pip).
- **Testing:** `uv run pytest` (Allure results), `uv run ptw` (watch mode).

### Code Conventions
- **Types:** Broad inputs, narrow outputs. Use built-in generics (`list[str]`) and unions (`|`).
- **Data:** Prefer Polars for transforms; Pandas for presentation. Standard column names: `Result`, `Hits`, `Hit Rate`, `n trials`.
- **Plotting:** Use Plotly Express with `plot.template`.
- **Testing:** Mirror source structure in `tests/`. Use fixtures for common data.

## 2. Custom Agents (`.github/agents/`)

These agents enforce specific workflows and constraints.

### `tdd-red` (TDD Red Phase Enforcer)
- **Role:** Strict enforcer for writing failing tests.
- **Constraints:**
    - Writes **exactly ONE** test function per invocation.
    - Tests **exactly ONE** behavior.
    - Validates atomicity before writing.
    - Must verify the test fails (assertion failure, not syntax error).
- **Workflow:**
    1.  Identify next behavior from BDD features or API needs.
    2.  Check atomicity (one behavior, one assertion).
    3.  Write the test.
    4.  Run and confirm failure.
    5.  Commit with message: `feat: red phase - <behavior>`.
- **Violation Handling:** Splits multiple behaviors into separate red phases.

### `tdd-green` (TDD Green Phase)
- **Role:** Implement minimal code to pass the failing test.
- **Constraints:**
    - **Minimal implementation** only.
    - No refactoring or extra features.
    - Respects data hierarchy.
- **Workflow:**
    1.  Review failing test.
    2.  Implement code in `src/psychoanalyze/`.
    3.  Run the test to confirm green.

### `tdd-refactor` (TDD Refactor Phase)
- **Role:** Improve code quality and security while keeping tests green.
- **Goals:** Remove duplication, improve naming, validate inputs, respect module boundaries.
- **Constraints:** No new behaviors without a red test.

### `tdd-cycle` (TDD Cycle Orchestrator)
- **Role:** Orchestrates exactly **ONE** red/green cycle.
- **Workflow:**
    1.  Invokes `tdd-red` to create a failing test.
    2.  Invokes `tdd-green` to pass the test.
- **Constraints:** Do not refactor or add extra tests.

### `jj-parallel-splitter` (JJ Parallel Splitter)
- **Role:** Splits an oversized revision into parallel branches.
- **Workflow:**
    1.  Identify change groups in the revision.
    2.  Abandon the oversized revision (keep working copy).
    3.  Create new parallel branches for each group from the original parent.
    4.  Optionally enforce TDD organization (red/green/refactor).

## 3. Instructions (`.github/instructions/`)

Detailed guidelines for specific tasks.

### `gherkin-feature.instructions.md`
- **Goal:** Write maintainable Gherkin feature files.
- **Rules:**
    - Describe **behavior**, not UI steps.
    - Keep steps reusable and atomic.
    - Avoid conjunction steps ("And" inside a step).
    - Use Given/When/Then consistently.

### `jj-revision-splitting.instructions.md`
- **Goal:** Manage revisions cleanly.
- **Key Concept: `jj absorb`**
    - Use `jj absorb` to fix small mistakes in previous revisions without creating new "fix" commits.
    - Useful for fixing typos, adding imports, or improving assertions in the red phase.
- **Splitting:** Use splitting for truly separate behaviors.
- **Evolution:** Use `jj evolog` to review how a revision changed over time.

### `marimo-notebooks.instructions.md`
- **Goal:** Edit Marimo notebooks safely.
- **Workflow:**
    - Use `mcp_marimo_*` tools (lint, check errors).
    - Check for lint errors (breaking, runtime, formatting) before finishing.
    - Debug using `get_notebook_errors` and `get_cell_runtime_data`.

### `python.instructions.md`
- **Type Annotations:** Use broad inputs, narrow outputs, built-in generics, and `ty` for checking.
- **Management:** Use `uv` for packages/projects.
- **Linting:** Use `ruff`.

## 4. Prompts (`.github/prompts/`)

Specific prompt templates for common tasks.

### `gherkin-scenario`
- **Task:** Write a pytest test for a Gherkin scenario.
- **Requirements:** Use plain pytest (not pytest-bdd), map Given/When/Then to Arrange/Act/Assert, keep steps reusable.

### `jj-describe-revisions`
- **Task:** Add a description to the current revision.
- **Format:** Conventional commits (`feat`, `fix`, `refactor`, etc.).
- **Tools:** Use `mcp_jj_show` to inspect changes.

### `jj-new-revision-from-d2`
- **Task:** Create a new revision based on a `.d2` diagram.
- **Workflow:** Read diagram -> Determine behavior -> Create revision -> Create Gherkin feature file.

### `jj-split-revision`
- **Task:** Invoke `jj-parallel-splitter` to split the current revision.

### `tdd-red-next-test`
- **Task:** Invoke `tdd-red` to choose and implement the next failing test from BDD features.
