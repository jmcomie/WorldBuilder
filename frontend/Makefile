# Frontend Development Makefile
# Provides convenient shortcuts for common development tasks

# Default target - show help
.PHONY: help
help:
	@echo "Frontend Development Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make dev        - Start development server"
	@echo "  make build      - Build for production"
	@echo "  make preview    - Preview production build"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint       - Run linter"
	@echo "  make lint-fix   - Fix linting issues automatically"
	@echo "  make typecheck  - Run TypeScript type checking"
	@echo "  make format     - Format code with Prettier"
	@echo "  make format-check - Check code formatting"
	@echo ""
	@echo "Utilities:"
	@echo "  make install    - Install dependencies"
	@echo "  make clean      - Clean build artifacts"
	@echo ""
	@echo "Workflows:"
	@echo "  make check      - Run all checks (lint, typecheck, format-check)"
	@echo "  make fix        - Fix all auto-fixable issues"
	@echo "  make ci         - Complete CI pipeline (install, check, build)"

# Development commands
.PHONY: dev
dev:
	pnpm run dev

.PHONY: build  
build:
	pnpm run build

.PHONY: preview
preview:
	pnpm run preview

# Code quality commands
.PHONY: lint
lint:
	pnpm run lint

.PHONY: lint-fix
lint-fix:
	pnpm run lint:fix

.PHONY: typecheck
typecheck:
	pnpm run typecheck

.PHONY: format
format:
	pnpm run format

.PHONY: format-check
format-check:
	pnpm run format:check

# Utility commands
.PHONY: install
install:
	pnpm install

.PHONY: clean
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf dist
	@rm -rf node_modules/.cache
	@rm -rf node_modules/.tmp
	@echo "✅ Clean complete!"

# Compound workflow commands
.PHONY: check
check: typecheck lint format-check
	@echo "✅ All checks passed!"

.PHONY: fix
fix: format lint-fix
	@echo "🔧 Auto-fixes applied!"

# CI/CD pipeline
.PHONY: ci
ci: install check build
	@echo "🚀 CI pipeline completed successfully!"

# Additional shortcuts for common typos/variations
.PHONY: test
test: check
	@echo "ℹ️  No tests configured yet. Running code quality checks instead."

.PHONY: start
start: dev

.PHONY: serve
serve: preview