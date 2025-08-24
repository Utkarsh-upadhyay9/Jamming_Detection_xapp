#!/bin/bash

# Setup script for Ensemble ML Jamming Detection xApp
# This script sets up the complete environment and dependencies

set -e  # Exit on error

echo "=================================="
echo "Jamming Detection xApp Setup"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if Python 3.8+ is available
check_python() {
    print_header "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
        print_status "Found Python $PYTHON_VERSION"
        
        # Check if version is 3.8 or higher
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_status "Python version is compatible (3.8+)"
        else
            print_error "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_header "Setting up virtual environment..."
    
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    else
        print_status "Virtual environment already exists"
    fi
    
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    print_status "Upgrading pip..."
    pip install --upgrade pip
}

# Install Python dependencies
install_dependencies() {
    print_header "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        print_status "Installing from requirements.txt..."
        pip install -r requirements.txt
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    print_status "All dependencies installed successfully"
}

# Clone dataset repository
setup_dataset() {
    print_header "Setting up dataset..."
    
    if [ ! -d "Ensemble_ML_Jamming_detection_dataset" ]; then
        print_status "Cloning dataset repository..."
        git clone https://github.com/Utkarsh-upadhyay9/Ensemble_ML_Jamming_detection_dataset.git
        
        if [ $? -eq 0 ]; then
            print_status "Dataset repository cloned successfully"
        else
            print_warning "Failed to clone dataset repository"
            print_warning "You can clone it manually later:"
            echo "git clone https://github.com/Utkarsh-upadhyay9/Ensemble_ML_Jamming_detection_dataset.git"
        fi
    else
        print_status "Dataset repository already exists"
    fi
}

# Create necessary directories
create_directories() {
    print_header "Creating project directories..."
    
    directories=("models" "logs" "plots" "results")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        else
            print_status "Directory already exists: $dir"
        fi
    done
}

# Verify installation
verify_installation() {
    print_header "Verifying installation..."
    
    print_status "Testing imports..."
    python3 -c "
import numpy as np
import pandas as pd
import scikit_learn as sklearn
import matplotlib
import seaborn
import plotly
print('✓ All core dependencies imported successfully')
"
    
    if [ $? -eq 0 ]; then
        print_status "Installation verification passed"
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Run quick test
run_quick_test() {
    print_header "Running quick functionality test..."
    
    print_status "Testing ensemble model initialization..."
    python3 -c "
import sys
import os
sys.path.append('.')
from src.ensemble_model import EnsembleJammingDetector
detector = EnsembleJammingDetector()
print('✓ Ensemble model initialized successfully')
"
    
    if [ $? -eq 0 ]; then
        print_status "Quick test passed"
    else
        print_error "Quick test failed"
        exit 1
    fi
}

# Main setup process
main() {
    print_header "Starting setup process..."
    
    # Change to script directory
    cd "$(dirname "$0")"
    
    # Run setup steps
    check_python
    create_venv
    install_dependencies
    setup_dataset
    create_directories
    verify_installation
    run_quick_test
    
    print_header "Setup completed successfully!"
    print_status "Virtual environment created: ./venv"
    print_status "Dependencies installed"
    print_status "Dataset repository cloned"
    print_status "Project directories created"
    
    echo ""
    print_header "Next steps:"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. Run the application: python main.py"
    echo "3. Or start with evaluation: python main.py evaluate --normal Ensemble_ML_Jamming_detection_dataset/dataset/normal_traffic.csv --jamming Ensemble_ML_Jamming_detection_dataset/dataset/jamming_attacks.csv"
    echo ""
    print_status "Setup complete! Ready to run the xApp."
}

# Run main function
main
