#!/bin/bash
set -e

# Default values
CONDA_ENV="viridot2_env"
SAM2_DIR="sam2_repo"
APP_DIR="."

# Function to display usage
usage() {
    echo "Usage: $0 [-e|--env CONDA_ENV] [-d|--dir SAM2_DIR]"
    echo "Options:"
    echo "  -e, --env     Name of the Conda environment (default: $CONDA_ENV)"
    echo "  -d, --dir     Directory for SAM2 repo (default: $SAM2_DIR)"
    echo "  -h, --help    Display this help message"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -d|--dir)
            SAM2_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate inputs
if [[ -z "$CONDA_ENV" ]]; then
    echo "Error: Conda environment name cannot be empty."
    usage
fi
if [[ -z "$SAM2_DIR" ]]; then
    echo "Error: SAM2 directory cannot be empty."
    usage
fi

# Check if module command is available
if ! command -v module >/dev/null 2>&1; then
    echo "Warning: 'module' command not found. Assuming standard Conda setup."
else
    # Load miniconda3 module
    module load miniconda3 || {
        echo "Error: Failed to load miniconda3 module. Is it available?"
        exit 1
    }
fi

# Check if conda is available
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: 'conda' command not found. Please ensure Miniconda/Anaconda is installed."
    exit 1
fi

# Initialize Conda
# Try modern hook method first, fall back to conda.sh
if eval "$(conda shell.bash hook 2>/dev/null)"; then
    echo "Conda initialized using shell hook."
else
    # Fallback to sourcing conda.sh
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [[ -z "$CONDA_BASE" ]] || [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        echo "Error: Cannot find conda.sh to initialize Conda."
        echo "Please ensure Conda is properly installed and initialized."
        exit 1
    fi
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

# Create or update Conda environment
if conda env list | grep -q "^${CONDA_ENV}\s"; then
    echo "Conda environment '$CONDA_ENV' already exists, updating it."
else
    echo "Creating Conda environment '$CONDA_ENV' with Python 3.11..."
    conda create -y -n "$CONDA_ENV" python=3.11 pyside6 qtimageformats
fi

# Activate Conda environment
conda activate "$CONDA_ENV" || {
    echo "Error: Failed to activate Conda environment '$CONDA_ENV'."
    exit 1
}

# Install dependencies
echo "Installing dependencies..."
# Install PySide6 and SAM2-related dependencies
pip install --upgrade pip
pip install natsort scikit-image pandas openpyxl|| {
    echo "Error: Failed to install core dependencies."
    exit 1
}
pip install histomicstk --find-links https://girder.github.io/large_image_wheels || {
    echo "Error: Failed to install core dependencies: histomicstk."
    exit 1
}
pip install torch torchvision torchaudio || {
    echo "Error: Failed to install core dependencies: pytorch."
    exit 1
}

# Install additional dependencies from requirements.txt if present
if [[ -f "${APP_DIR}/requirements.txt" ]]; then
    pip install -r "${APP_DIR}/requirements.txt" || {
        echo "Warning: Failed to install some dependencies from requirements.txt."
    }
else
    echo "No requirements.txt found, skipping additional dependencies."
fi

# Clone SAM2 repo if not already present
if [[ ! -d "$SAM2_DIR" ]]; then
    echo "Cloning SAM2 repository..."
    git clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR" || {
        echo "Error: Failed to clone SAM2 repository."
        exit 1
    }
else
    echo "SAM2 directory '$SAM2_DIR' already exists, skipping clone."
fi

# Install SAM2
echo "Installing SAM2..."
cd "$SAM2_DIR"
pip install -e . || {
    echo "Error: Failed to install SAM2."
    exit 1
}
cd ..

# Run weights download script
CHECKPOINTS_DIR="$SAM2_DIR/checkpoints"
WEIGHTS_SCRIPT="$CHECKPOINTS_DIR/download_ckpts.sh"
if [[ -f "$WEIGHTS_SCRIPT" ]]; then
    echo "Downloading SAM2 weights..."
    cd "$CHECKPOINTS_DIR" || {
        echo "Error: Cannot access checkpoints directory '$CHECKPOINTS_DIR'."
        exit 1
    }
    bash ./download_ckpts.sh || {
        echo "Warning: Weights download script failed. Check network or script."
    }
    cd - >/dev/null
else
    echo "Warning: Weights download script not found in '$CHECKPOINTS_DIR'."
fi

# Provide instructions for running the app
echo ""
echo "Setup complete for environment '$CONDA_ENV'!"
echo "To run the app:"
echo "1. Activate the Conda environment:"
echo "   module load miniconda3"
echo "   conda activate $CONDA_ENV"
echo "2. Run the app:"
echo "   cd $APP_DIR"
echo "   python viridot2.py"
echo ""
echo "If you encounter issues, ensure 'viridot2.py' is in the '$APP_DIR' directory."
