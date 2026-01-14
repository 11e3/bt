#!/bin/bash
# Code quality check script for Linux
# Run: ./scripts/check.sh

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${CYAN}🔍 Running comprehensive code quality checks...${NC}"

echo -e "\n${YELLOW}📋 Installing/Updating pre-commit hooks...${NC}"
if ! command -v pre-commit &> /dev/null; then
    if [ -f ".venv/bin/pip" ]; then
        .venv/bin/pip install pre-commit
    elif command -v uv &> /dev/null; then
        uv pip install pre-commit
    else
        echo -e "${RED}❌ Neither virtual environment nor uv found${NC}"
        exit 1
    fi
fi
if [ -f ".venv/bin/pre-commit" ]; then
    .venv/bin/pre-commit install
elif command -v pre-commit &> /dev/null; then
    pre-commit install
else
    echo -e "${RED}❌ pre-commit command not found${NC}"
    exit 1
fi

echo -e "\n${YELLOW}📝 Formatting code with ruff...${NC}"
if [ -f ".venv/bin/ruff" ]; then
    .venv/bin/ruff format . --exclude=src/bt/core/simple_strategies.py
elif command -v ruff &> /dev/null; then
    ruff format . --exclude=src/bt/core/simple_strategies.py
else
    echo -e "${RED}❌ ruff not found${NC}"
    exit 1
fi
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Formatting failed${NC}"
    exit 1
fi

echo -e "\n${YELLOW}🔧 Linting and fixing with ruff...${NC}"
if [ -f ".venv/bin/ruff" ]; then
    .venv/bin/ruff check . --fix --unsafe-fixes --exclude=src/bt/core/simple_strategies.py
elif command -v ruff &> /dev/null; then
    ruff check . --fix --unsafe-fixes --exclude=src/bt/core/simple_strategies.py
else
    echo -e "${RED}❌ ruff not found${NC}"
    exit 1
fi
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Linting failed${NC}"
    exit 1
fi

echo -e "\n${YELLOW}🔍 Type checking with mypy...${NC}"
# Clean mypy cache to avoid deserialization errors
rm -rf .mypy_cache || true
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python -m mypy src/bt --strict --exclude="src/bt/core/simple_strategies.py"
elif command -v mypy &> /dev/null; then
    python -m mypy src/bt --strict --exclude="src/bt/core/simple_strategies.py"
else
    echo -e "${RED}❌ mypy not found${NC}"
    exit 1
fi
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Type checking failed${NC}"
    exit 1
fi

echo -e "\n${YELLOW}🛡️ Security scan with bandit...${NC}"
if [ -f ".venv/bin/bandit" ]; then
    .venv/bin/bandit -r src/bt/ --exclude="src/bt/core/simple_strategies.py" -f json -o bandit-report.json || true
elif command -v bandit &> /dev/null; then
    bandit -r src/bt/ --exclude="src/bt/core/simple_strategies.py" -f json -o bandit-report.json || true
else
    echo -e "${RED}❌ bandit not found${NC}"
    exit 1
fi
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Security scan failed${NC}"
    # Don't exit on security issues, just report them
fi

echo -e "\n${YELLOW}🧪 Running tests with coverage...${NC}"
# Ensure package is installed in editable mode
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
if [ -f ".venv/bin/pytest" ]; then
    .venv/bin/pytest --cov=src/bt --cov-report=term-missing --cov-report=html --cov-report=xml
elif command -v pytest &> /dev/null; then
    pytest --cov=src/bt --cov-report=term-missing --cov-report=html --cov-report=xml
else
    echo -e "${RED}❌ pytest not found${NC}"
    exit 1
fi
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Tests failed${NC}"
    exit 1
fi

echo -e "\n${YELLOW}📊 Running pre-commit on all files...${NC}"
if [ -f ".venv/bin/pre-commit" ]; then
    .venv/bin/pre-commit run --all-files
else
    echo -e "${RED}❌ pre-commit not found in virtual environment${NC}"
    exit 1
fi
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Pre-commit checks failed${NC}"
    exit 1
fi

# Show bandit summary if report exists
if [ -f "bandit-report.json" ]; then
    echo -e "\n${BLUE}🛡️ Security Scan Summary:${NC}"
    if command -v jq &> /dev/null; then
        # Use jq for pretty output if available
        ISSUES=$(jq '.results | length' bandit-report.json)
        HIGH=$(jq '.results | map(select(.issue_severity == "HIGH")) | length' bandit-report.json)
        MEDIUM=$(jq '.results | map(select(.issue_severity == "MEDIUM")) | length' bandit-report.json)
        LOW=$(jq '.results | map(select(.issue_severity == "LOW")) | length' bandit-report.json)

        echo -e "  Total Issues: ${YELLOW}$ISSUES${NC}"
        echo -e "  High Severity: ${RED}$HIGH${NC}"
        echo -e "  Medium Severity: ${YELLOW}$MEDIUM${NC}"
        echo -e "  Low Severity: ${BLUE}$LOW${NC}"

        if [ "$ISSUES" -eq 0 ]; then
            echo -e "${GREEN}✅ No security issues found${NC}"
        fi
    else
        # Fallback without jq
        ISSUES=$(grep -c '"issue_severity"' bandit-report.json 2>/dev/null || echo "0")
        echo -e "  Security Issues Found: ${YELLOW}$ISSUES${NC}"
        if [ "$ISSUES" -eq 0 ]; then
            echo -e "${GREEN}✅ No security issues found${NC}"
        fi
    fi
fi

# Show coverage summary
if [ -f "htmlcov/index.html" ]; then
    echo -e "\n${BLUE}📊 Coverage Report:${NC}"
    echo -e "  HTML report: ${CYAN}file://$(pwd)/htmlcov/index.html${NC}"
    if command -v python3 &> /dev/null; then
        COVERAGE=$(python3 -c "
import json
import re
with open('htmlcov/index.html', 'r') as f:
    content = f.read()
    match = re.search(r'([0-9]+)%', content)
    if match:
        print(match.group(1))
    else:
        print('N/A')
" 2>/dev/null || echo "N/A")
        echo -e "  Coverage: ${YELLOW}$COVERAGE%${NC}"
    fi
fi

echo -e "\n${GREEN}✅ All quality checks completed!${NC}"
echo -e "${BLUE}📋 Summary:${NC}"
echo -e "  - Code formatted with ruff"
echo -e "  - Linting issues fixed"
echo -e "  - Type checking completed"
echo -e "  - Security scan completed"
echo -e "  - Tests executed with coverage"
echo -e "  - Pre-commit hooks verified"

# Optional: Open coverage report in browser (uncomment if desired)
# if command -v xdg-open &> /dev/null; then
#     xdg-open htmlcov/index.html
# elif command -v open &> /dev/null; then
#     open htmlcov/index.html
# fi
