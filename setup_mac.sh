#!/bin/bash
# setup_mac.sh - Complete setup script for MediPredict on macOS

# Exit on error
set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo -e "${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         MediPredict Setup for macOS                  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"

# Detect system architecture
if [[ $(uname -m) == 'arm64' ]]; then
    print_info "Detected Apple Silicon Mac (M1/M2)"
    ARCH="arm64"
else
    print_info "Detected Intel Mac"
    ARCH="x86_64"
fi

# Step 1: Check Homebrew
print_info "Checking for Homebrew..."
if ! command -v brew &> /dev/null; then
    print_warning "Homebrew not found. It's recommended for macOS development."
    echo "Install Homebrew? (recommended) [y/N]"
    read -r INSTALL_BREW
    if [[ "$INSTALL_BREW" =~ ^[Yy]$ ]]; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
else
    print_success "Homebrew found"
fi

# Step 2: Install system dependencies
if command -v brew &> /dev/null; then
    print_info "Installing system dependencies..."
    brew install libomp || true  # OpenMP for XGBoost
    brew install cmake || true    # For building some packages
fi

# Step 3: Check Python version
print_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    print_error "Python $REQUIRED_VERSION or higher is required. Found: Python $PYTHON_VERSION"
    exit 1
fi
print_success "Python $PYTHON_VERSION detected"

# Step 4: Create virtual environment
if [ ! -d "medipredict_env" ]; then
    print_info "Creating Python virtual environment..."
    python3 -m venv medipredict_env
else
    print_info "Virtual environment already exists."
fi

# Step 5: Activate virtual environment
print_info "Activating virtual environment..."
source medipredict_env/bin/activate

# Verify activation
which python
python --version
print_success "Virtual environment activated"

# Step 6: Upgrade pip and install wheel
print_info "Upgrading pip and installing build tools..."
pip install --upgrade pip wheel setuptools

# Step 7: Install numpy first (required by many other packages)
print_info "Installing numpy (required by other packages)..."
pip install "numpy>=1.20.0,<1.24.0"

# Step 8: Install core packages one by one for better error handling
print_info "Installing core packages..."

# Install packages in order of dependencies
PACKAGES=(
    "pandas>=1.3.0,<2.0.0"
    "scikit-learn>=1.0.0,<1.3.0"
    "matplotlib>=3.4.0,<3.8.0"
    "seaborn>=0.11.0,<0.13.0"
    "flask>=2.0.0,<3.0.0"
    "joblib>=1.1.0,<1.3.0"
    "plotly>=5.0.0,<5.15.0"
    "python-dotenv>=0.19.0,<1.0.0"
)

for package in "${PACKAGES[@]}"; do
    print_info "Installing $package..."
    pip install "$package" || print_warning "Failed to install $package"
done

# Step 9: Install TensorFlow (special handling for M1/M2)
print_info "Installing TensorFlow..."
if [[ $ARCH == "arm64" ]]; then
    # For Apple Silicon
    pip install tensorflow-macos>=2.9.0 || print_warning "TensorFlow installation failed"
else
    # For Intel Macs
    pip install "tensorflow>=2.9.0,<2.13.0" || print_warning "TensorFlow installation failed"
fi

# Step 10: Handle XGBoost installation
print_info "Installing XGBoost with macOS fixes..."
if [[ $ARCH == "arm64" ]]; then
    # Set environment variables for Apple Silicon
    export CPATH="/opt/homebrew/include"
    export LIBRARY_PATH="/opt/homebrew/lib"
fi

# Try to install XGBoost
pip install xgboost --no-cache-dir || {
    print_warning "XGBoost installation failed. Trying alternative approach..."
    
    # Create symbolic link for libomp if on Apple Silicon
    if [[ $ARCH == "arm64" ]] && [ -f "/opt/homebrew/opt/libomp/lib/libomp.dylib" ]; then
        print_info "Creating symbolic link for libomp..."
        sudo mkdir -p /usr/local/lib 2>/dev/null || true
        sudo ln -sf /opt/homebrew/opt/libomp/lib/libomp.dylib /usr/local/lib/libomp.dylib 2>/dev/null || true
    fi
    
    # Try again
    pip install xgboost --no-cache-dir || print_warning "XGBoost installation failed. Continuing without it."
}

# Step 11: Install LightGBM (often problematic on M1/M2)
print_info "Installing LightGBM..."
pip install lightgbm || {
    print_warning "LightGBM installation failed. Trying from source..."
    if command -v brew &> /dev/null; then
        brew install cmake
        pip install lightgbm --no-binary lightgbm || print_warning "LightGBM installation failed. Continuing without it."
    fi
}

# Step 12: Install SHAP (can be problematic)
print_info "Installing SHAP..."
pip install shap || print_warning "SHAP installation failed. Continuing without it."

# Step 13: Create necessary directories
print_info "Creating project directories..."
mkdir -p data/{raw,processed}
mkdir -p models/{diabetes,heart_disease,meta,preprocessor}
mkdir -p visualizations
mkdir -p medipredict/api/static/img

# Step 14: Verify installation
print_info "Verifying installation..."
python -c "
import sys
print(f'Python: {sys.version}')
try:
    import numpy
    print(f'✓ NumPy: {numpy.__version__}')
except: print('✗ NumPy not installed')
try:
    import pandas
    print(f'✓ Pandas: {pandas.__version__}')
except: print('✗ Pandas not installed')
try:
    import sklearn
    print(f'✓ Scikit-learn: {sklearn.__version__}')
except: print('✗ Scikit-learn not installed')
try:
    import flask
    print(f'✓ Flask: {flask.__version__}')
except: print('✗ Flask not installed')
try:
    import xgboost
    print(f'✓ XGBoost: {xgboost.__version__}')
except: print('⚠ XGBoost not installed (optional)')
try:
    import lightgbm
    print(f'✓ LightGBM: {lightgbm.__version__}')
except: print('⚠ LightGBM not installed (optional)')
"

echo ""
print_success "✅ Setup complete!"
echo ""
print_info "Next steps:"
echo "1. To activate the environment: source medipredict_env/bin/activate"
echo "2. To run the application: ./run_medipredict.sh"
echo ""

# Offer to install the fixed app.py
print_info "Would you like to install the fixed app.py? [y/N]"
read -r INSTALL_FIX
if [[ "$INSTALL_FIX" =~ ^[Yy]$ ]]; then
    if [ -f "medipredict/api/app.py" ]; then
        cp medipredict/api/app.py medipredict/api/app.py.backup
        print_success "Original app.py backed up to app.py.backup"
    fi
    print_info "Please copy the fixed app.py content to medipredict/api/app.py"
fi