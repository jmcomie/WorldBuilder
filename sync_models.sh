#!/bin/bash

# sync_models.sh - Generate SQLModels, OpenAPI schema, and TypeScript SDK
# Usage: ./sync_models.sh [sqlmodels] [openapi] [sdk]
# Examples:
#   ./sync_models.sh sqlmodels          # Generate SQLModels only
#   ./sync_models.sh openapi            # Generate OpenAPI schema only
#   ./sync_models.sh sdk                # Generate TypeScript SDK only
#   ./sync_models.sh sqlmodels openapi  # Generate SQLModels and OpenAPI
#   ./sync_models.sh openapi sdk        # Generate OpenAPI and SDK
#   ./sync_models.sh sqlmodels openapi sdk  # Generate all

set -e  # Exit on error

# Get the script's directory (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="${SCRIPT_DIR}/backend"
FRONTEND_DIR="${SCRIPT_DIR}/frontend"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to generate SQLModels
generate_sqlmodels() {
    print_status "Generating SQLModels..."
    cd "${BACKEND_DIR}"
    
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install it first."
        return 1
    fi
    
    # Run the SQLModel generation script
    if uv run python scripts/generate_sqlmodels.py app/core/schema.py; then
        print_status "SQLModels generated successfully at backend/app/core/schema.py"
    else
        print_error "Failed to generate SQLModels"
        return 1
    fi
}

# Function to generate OpenAPI schema
generate_openapi() {
    print_status "Generating OpenAPI schema..."
    cd "${BACKEND_DIR}"
    
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install it first."
        return 1
    fi
    
    # Create the target directory if it doesn't exist
    mkdir -p "${FRONTEND_DIR}/src/shared/api"
    
    # Run the OpenAPI generation script
    if uv run python scripts/generate_openapi.py --output "${FRONTEND_DIR}/src/shared/api/openapi.json"; then
        print_status "OpenAPI schema generated successfully at frontend/src/shared/api/openapi.json"
    else
        print_error "Failed to generate OpenAPI schema"
        return 1
    fi
}

# Function to generate TypeScript SDK
generate_sdk() {
    print_status "Generating TypeScript SDK..."
    cd "${FRONTEND_DIR}"
    
    if ! command -v pnpm &> /dev/null; then
        print_error "pnpm is not installed. Please install it first."
        return 1
    fi
    
    # Check if OpenAPI schema exists
    if [ ! -f "src/shared/api/openapi.json" ]; then
        print_warning "OpenAPI schema not found. Generating it first..."
        generate_openapi
    fi
    
    # Run the SDK generation
    if pnpm run api:generate; then
        print_status "TypeScript SDK generated successfully at frontend/src/shared/api/sdk/"
    else
        print_error "Failed to generate TypeScript SDK"
        return 1
    fi
}

# Main script logic
main() {
    # If no arguments provided, show usage
    if [ $# -eq 0 ]; then
        echo "Usage: $0 [sqlmodels] [openapi] [sdk]"
        echo ""
        echo "Arguments:"
        echo "  sqlmodels  - Generate SQLModel classes from database schema"
        echo "  openapi    - Generate OpenAPI schema from FastAPI app"
        echo "  sdk        - Generate TypeScript SDK from OpenAPI schema"
        echo ""
        echo "Examples:"
        echo "  $0 sqlmodels           # Generate SQLModels only"
        echo "  $0 openapi             # Generate OpenAPI schema only"
        echo "  $0 sdk                 # Generate TypeScript SDK only"
        echo "  $0 sqlmodels openapi   # Generate SQLModels and OpenAPI"
        echo "  $0 openapi sdk         # Generate OpenAPI and SDK"
        echo "  $0 sqlmodels openapi sdk  # Generate all"
        exit 0
    fi
    
    # Track if any generation was requested
    local any_generated=false
    
    # Process each argument
    for arg in "$@"; do
        case "$arg" in
            sqlmodels)
                generate_sqlmodels
                any_generated=true
                ;;
            openapi)
                generate_openapi
                any_generated=true
                ;;
            sdk)
                generate_sdk
                any_generated=true
                ;;
            *)
                print_error "Unknown argument: $arg"
                echo "Valid arguments are: sqlmodels, openapi, sdk"
                exit 1
                ;;
        esac
    done
    
    if [ "$any_generated" = true ]; then
        print_status "All requested generations completed successfully!"
    fi
}

# Run the main function with all arguments
main "$@"
