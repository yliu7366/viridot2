#
# HOW TO RUN THIS SCRIPT
# ======================
# Because of PowerShell's default security settings (Execution Policy), you cannot
# simply run this script by typing '.\install.ps1'.
#
# Instead, open a PowerShell terminal, navigate to the directory where you saved
# this file, and use the following command. This bypasses the policy for this
# single execution without changing any system settings.
#
# PowerShell -ExecutionPolicy Bypass -File .\install.ps1
#

# SCRIPT: install.ps1
#
# This script sets up the viridot2 environment, clones the SAM2 repository,
# and installs all necessary dependencies.

# --- Parameters for Command-Line Arguments ---
# To run with custom values: .\install.ps1 -CondaEnv "my_env" -Sam2Dir "my_sam_repo"
param(
    [string]$CondaEnv = "viridot2_env",
    [string]$Sam2Dir = "sam2_repo",
    [string]$AppDir = "."
)

# --- Main Logic ---
try {
    # --- Prerequisite Checks ---
    Write-Host "Checking for prerequisites..." -ForegroundColor Cyan
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        throw "Error: 'conda' command not found. Please ensure Miniconda/Anaconda is installed and configured for PowerShell."
    }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Error: 'git' command not found. Please install Git for Windows."
    }
    Write-Host "Prerequisites found."

    # --- Environment Setup ---
    Write-Host "Checking for Conda environment '$CondaEnv'..." -ForegroundColor Cyan
    # Get a list of conda environments as an array of lines. Do NOT use Out-String.
    $envList = @(conda env list)
    if ($envList -match ('^' + [regex]::Escape($CondaEnv) + '\s')) {
        Write-Host "Conda environment '$CondaEnv' already exists, skipping creation."
    } else {
        Write-Host "Creating Conda environment '$CondaEnv' with Python 3.11..."
        conda create -y -n $CondaEnv python=3.11 pyside6 qtimageformats
        if ($LASTEXITCODE -ne 0) {
            # We add a verification step here just in case `conda create` succeeded but returned a weird exit code
            $checkList = @(conda env list)
            if (-not ($checkList -match ('^' + [regex]::Escape($CondaEnv) + '\s'))) {
                 throw "Conda environment creation failed. Check the output above for errors."
            }
        }
        Write-Host "Environment created successfully."
    }

    # --- Activation and Installation ---
    $envPath = Join-Path (conda info --base) "envs\$CondaEnv"
    $pythonPath = Join-Path $envPath "python.exe"
    if (-not (Test-Path $pythonPath)) {
        throw "Could not find python.exe in the new environment at '$pythonPath'. Installation cannot continue."
    }

    Write-Host "Installing dependencies using pip..." -ForegroundColor Cyan
    & $pythonPath -m pip install --upgrade pip
    & $pythonPath -m pip install natsort scikit-image pandas openpyxl opencv-python torch torchvision torchaudio
    if ($LASTEXITCODE -ne 0) { throw "Failed to install one or more pip packages." }

    $requirementsFile = Join-Path $AppDir "requirements.txt"
    if (Test-Path $requirementsFile) {
        Write-Host "Installing dependencies from requirements.txt..."
        & $pythonPath -m pip install -r $requirementsFile
    } else {
        Write-Host "No requirements.txt found, skipping."
    }

    # --- SAM2 Installation ---
    if (-not (Test-Path $Sam2Dir)) {
        Write-Host "Cloning SAM2 repository..." -ForegroundColor Cyan
        git clone https://github.com/facebookresearch/sam2.git $Sam2Dir
        if ($LASTEXITCODE -ne 0) { throw "Failed to clone SAM2 repository." }
    } else {
        Write-Host "SAM2 directory '$Sam2Dir' already exists, skipping clone."
    }

    Write-Host "Installing SAM2..." -ForegroundColor Cyan
    Push-Location $Sam2Dir
    & $pythonPath -m pip install -e .
    if ($LASTEXITCODE -ne 0) { throw "Failed to install SAM2." }
    Pop-Location

    # --- Download Weights ---
    $weightsScript = Join-Path $Sam2Dir "checkpoints\download_ckpts.sh"
    if (Test-Path $weightsScript) {
        Write-Host "Downloading SAM2 weights..." -ForegroundColor Cyan
        $gitPath = (Get-Command git).Source
        $gitRootPath = (Get-Item (Split-Path $gitPath)).Parent.FullName
        $bashPath = Join-Path $gitRootPath "bin\bash.exe"
        if (-not (Test-Path $bashPath)) {
            throw "Could not find bash.exe at '$bashPath'. Your Git for Windows installation might be in an unexpected location."
        }
        Write-Host "Found bash at: $bashPath"
        Push-Location (Split-Path $weightsScript)
        & $bashPath ./download_ckpts.sh
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Weights download script failed. Check network or script."
        }
        Pop-Location
    } else {
        Write-Warning "Weights download script not found at '$weightsScript'."
    }

    # --- Final Instructions ---
    Write-Host "`nSetup complete for environment '$CondaEnv'!" -ForegroundColor Green
    Write-Host "To run the app:"
    Write-Host "1. Open a new PowerShell or Anaconda Prompt."
    Write-Host "2. Activate the Conda environment:"
    Write-Host "   conda activate $CondaEnv"
    Write-Host "3. Run the app:"
    Write-Host "   python viridot2.py"

}
catch {
    # This block runs if any 'throw' command is executed
    Write-Host "`nAn error occurred during installation:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
