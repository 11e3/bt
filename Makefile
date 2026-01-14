.PHONY: check lint format test security clean install-tools help

# Default target
all: check

# Comprehensive code quality check
check:
	@echo "🔍 Running comprehensive code quality checks..."
	./scripts/check.sh

# Install development tools
install-tools:
	@echo "📦 Installing development tools..."
	pip install ruff mypy bandit pre-commit pytest-cov
	pre-commit install

# Format code only
format:
	@echo "📝 Formatting code with ruff..."
	ruff format . --exclude=src/bt/core/simple_strategies.py

# Lint code only (without fixing)
lint:
	@echo "🔧 Running ruff linter..."
	ruff check . --exclude=src/bt/core/simple_strategies.py

# Fix linting issues
lint-fix:
	@echo "🔧 Linting and fixing with ruff..."
	ruff check . --fix --unsafe-fixes --exclude=src/bt/core/simple_strategies.py

# Type checking only
types:
	@echo "🔍 Type checking with mypy..."
	rm -rf .mypy_cache || true
	mypy src/bt --strict --exclude=src/bt/core/simple_strategies.py

# Security scan only
security:
	@echo "🛡️ Running security scan..."
	bandit -r src/bt/ --exclude=src/bt/core/simple_strategies.py -f json -o bandit-report.json || true
	@if [ -f "bandit-report.json" ]; then \
		echo "🛡️ Security Report: bandit-report.json"; \
	fi

# Run tests only
test:
	@echo "🧪 Running tests with coverage..."
	export PYTHONPATH="${PWD}/src:${PYTHONPATH}"; \
	pytest --cov=src/bt --cov-report=term-missing --cov-report=html

# Run tests without coverage
test-quick:
	@echo "🧪 Running tests (quick)..."
	export PYTHONPATH="${PWD}/src:${PYTHONPATH}"; \
	pytest -v

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -f bandit-report.json
	rm -rf .ruff_cache

# Run pre-commit on all files
pre-commit:
	@echo "📋 Running pre-commit on all files..."
	pre-commit run --all-files

# Quick check (format + lint + test)
quick-check: format test
	@echo "✅ Quick check completed"

# Development setup (install tools + pre-commit)
setup: install-tools
	@echo "🚀 Development environment setup complete!"

# Show help
help:
	@echo "Available targets:"
	@echo "  check         - Run comprehensive code quality checks"
	@echo "  install-tools  - Install development tools"
	@echo "  format        - Format code with ruff"
	@echo "  lint          - Run ruff linter (no fixes)"
	@echo "  lint-fix      - Run ruff linter with fixes"
	@echo "  types         - Run mypy type checking"
	@echo "  security      - Run bandit security scan"
	@echo "  test          - Run tests with coverage"
	@echo "  test-quick    - Run tests (no coverage)"
	@echo "  pre-commit    - Run pre-commit on all files"
	@echo "  quick-check   - Format + test (quick workflow)"
	@echo "  clean         - Clean generated files"
	@echo "  setup         - Development environment setup"
	@echo "  help          - Show this help message"
