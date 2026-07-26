# One-time setup for a fresh clone of this project: creates the virtual
# environment, installs dependencies, and initializes the local knowledge
# base. Safe to re-run - skips venv creation if it already exists, and
# ingest.py itself updates existing rows instead of duplicating them.
#
# Usage (from the project root):
#   .\setup.ps1
#
# If PowerShell blocks the script with an execution-policy error, run:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
#
# Note: this only sets up the venv for the REST of this script. To use the
# venv afterward in your own shell, activate it yourself:
#   .\venv\Scripts\Activate.ps1

$ErrorActionPreference = "Stop"

if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
} else {
    Write-Host "Virtual environment already exists, skipping creation."
}

Write-Host "Activating virtual environment for this script..."
. .\venv\Scripts\Activate.ps1

Write-Host "Installing dependencies from requirements.txt..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Initializing the knowledge base (knowledge.db)..."
Write-Host "This downloads the embedding model on first run - needs an internet"
Write-Host "connection once, then everything runs fully offline."
python ingest.py

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next steps (after activating the venv in your own shell):"
Write-Host "  python main.py          - CLI Q&A loop"
Write-Host "  streamlit run app.py    - web UI"
Write-Host "  python check-db.py      - inspect knowledge.db contents"
