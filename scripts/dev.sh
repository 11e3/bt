#!/bin/bash
# Development workflow script
# Usage: ./scripts/dev.sh [command]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    echo -e "${CYAN}Available commands:${NC}"
    echo "  check         - Run comprehensive code quality checks"
    echo "  install-tools  - Install development tools"
    echo "  format        - Format code with ruff"
    echo "  lint          - Run ruff linter (no fixes)"
    echo "  lint-fix      - Run ruff linter with fixes"
    echo "  types         - Run mypy type checking"
    echo "  security      - Run bandit security scan"
    echo "  test          - Run tests with coverage"
    echo "  test-quick    - Run tests (no coverage)"
    echo "  pre-commit    - Run pre-commit on all files"
    echo "  quick-check   - Format + test (quick workflow)"
    echo "  clean         - Clean generated files"
    echo "  setup         - Development environment setup"
    echo "  help          - Show this help message"
}

case "${1:-help}" in
    "check")
        ./scripts/check.sh
        ;;
    "install-tools")
        echo -e "${YELLOW}📦 Installing development tools...${NC}"
        pip install ruff mypy bandit pre-commit pytest-cov
        pre-commit install
        echo -e "${GREEN}✅ Development tools installed${NC}"
        ;;
    "format")
        echo -e "${YELLOW}📝 Formatting code with ruff...${NC}"
        ruff format . --exclude=src/bt/core/simple_strategies.py
        echo -e "${GREEN}✅ Code formatted${NC}"
        ;;
    "lint")
        echo -e "${YELLOW}🔧 Running ruff linter...${NC}"
        ruff check . --exclude=src/bt/core/simple_strategies.py
        echo -e "${GREEN}✅ Linting completed${NC}"
        ;;
    "lint-fix")
        echo -e "${YELLOW}🔧 Linting and fixing with ruff...${NC}"
        ruff check . --fix --unsafe-fixes --exclude=src/bt/core/simple_strategies.py
        echo -e "${GREEN}✅ Linting issues fixed${NC}"
        ;;
    "types")
        echo -e "${YELLOW}🔍 Type checking with mypy...${NC}"
        rm -rf .mypy_cache || true
        mypy src/bt --strict --exclude=src/bt/core/simple_strategies.py
        echo -e "${GREEN}✅ Type checking completed${NC}"
        ;;
    "security")
        echo -e "${YELLOW}🛡️ Running security scan...${NC}"
        bandit -r src/bt/ --exclude=src/bt/core/simple_strategies.py -f json -o bandit-report.json || true
        if [ -f "bandit-report.json" ]; then
            echo -e "${CYAN}🛡️ Security Report: bandit-report.json${NC}"
        fi
        echo -e "${GREEN}✅ Security scan completed${NC}"
        ;;
    "test")
        echo -e "${YELLOW}🧪 Running tests with coverage...${NC}"
        export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
        pytest --cov=src/bt --cov-report=term-missing --cov-report=html
        echo -e "${GREEN}✅ Tests completed${NC}"
        ;;
    "test-quick")
        echo -e "${YELLOW}🧪 Running tests (quick)...${NC}"
        export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
        pytest -v
        echo -e "${GREEN}✅ Quick tests completed${NC}"
        ;;
    "pre-commit")
        echo -e "${YELLOW}📋 Running pre-commit on all files...${NC}"
        pre-commit run --all-files
        echo -e "${GREEN}✅ Pre-commit checks completed${NC}"
        ;;
    "quick-check")
        echo -e "${CYAN}🚀 Running quick check (format + test)...${NC}"
        ruff format . --exclude=src/bt/core/simple_strategies.py
        export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
        pytest --cov=src/bt --cov-report=term-missing
        echo -e "${GREEN}✅ Quick check completed${NC}"
        ;;
    "clean")
        echo -e "${YELLOW}🧹 Cleaning up...${NC}"
        rm -rf .mypy_cache
        rm -rf .pytest_cache
        rm -rf htmlcov
        rm -rf .coverage
        rm -f bandit-report.json
        rm -rf .ruff_cache
        echo -e "${GREEN}✅ Clean completed${NC}"
        ;;
    "setup")
        echo -e "${CYAN}🚀 Development environment setup...${NC}"
        pip install ruff mypy bandit pre-commit pytest-cov
        pre-commit install
        echo -e "${GREEN}✅ Development environment setup complete!${NC}"
        ;;
    "help"|*)
        show_help
        ;;
esac
