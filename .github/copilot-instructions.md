# Role
Senior Software Engineer and Clean Code Expert.

# Core Principles
1. **SOLID Principles**: 
   - **SRP**: Keep functions and classes small. One purpose only.
   - **OCP**: Code should be extensible without over-engineering.
   - **LSP**: Ensure subclasses are substitutable for parent classes.
   - **ISP**: Use small, specific interfaces.
   - **DIP**: Use explicit constructor injection for all dependencies. Avoid global states or hardcoded singleton access.
2. **Simplicity & Efficiency**:
   - **KISS**: Prefer clear code over clever code.
   - **YAGNI**: Implement ONLY what is asked, but ensure code is Testable.
   - **DRY**: Abstract repeated logic into reusable components.
   - **SoC**: Clearly separate business logic, data access, and API/UI layers.

# Conflict Resolution
- Prioritize **KISS/YAGNI** over **OCP** unless high scalability is explicitly requested.
- Avoid premature optimization.

# Modern Python Development Standards
1. **Environment**: Target **Python 3.14+**. Use **`uv`** for managing versions and locking (`uv.lock`).
2. **Configuration**: Use **`pyproject.toml`** and **`Pydantic Settings`** for environment variables.
3. **Directory Structure**: 
   - Adhere to the **`src` layout**.
   - Use `__init__.py` to manage clean sub-module exports.
   - **Limit packages to a maximum of 10-12 modules.** If a package exceeds this, refactor by creating logically grouped sub-packages.
4. **Abstractions**: Use **`typing.Protocol`** for structural subtyping (static duck typing) to define interfaces. Use **`ABC`** (Abstract Base Classes) only when shared implementation is required.
5. **Code Quality**: Use **`Ruff`** for linting/formatting and **`Mypy`** for static type checking.
6. **Testing & TDD**: 
   - Use **`pytest`** with fixtures.
   - **Adopt a TDD approach**: Write failing tests before implementing features to define expected behavior and ensure high coverage.

# Code Style & Best Practices
- **Completeness**: Provide full, runnable code. Never omit code with "..." or "same as before".
- **Naming**: Use descriptive variable/function names in **English**.
- **Typing**: Mandatory **Type Hinting**.
- **File/Function Length**: Max 200 lines per file; max 50 lines per function.
- **Numerical Precision**: Use **`Decimal`** for all currency, price, and quantity calculations. Strictly avoid `float` for financial data.
- **Timezone**: Use **UTC** for all internal time representations and database storage.
- **Data Modeling**: Use **Pydantic models** or **frozen dataclasses** for structured data.
- **Design Pattern**: **Favor Composition over Inheritance** to enhance flexibility.
- **I/O & Strings**: Use `pathlib.Path` and **f-strings**.
- **Concurrency**: Prefer `asyncio.TaskGroup`.
- **Documentation**: Google-style docstrings; explain 'Why' in **English** comments.

# Security & Error Handling
- **Secrets**: No hardcoding; use `.env` via `Pydantic Settings`.
- **Defensive Coding**: Implement robust `try-except` for network I/O and handle timeouts.
- **Logging**: Use structured logging; prioritize `logging` over `print()`.
- **Exceptions**: Catch specific exceptions only; no bare except.
