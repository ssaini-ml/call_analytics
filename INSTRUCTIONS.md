# Call Analytics Project Setup and Run Instructions

## ⚠️ IMPORTANT: Directory Location
**CRITICAL**: You MUST be in the correct directory to run any Poetry commands. The project directory is:
```bash
/Users/sunilsaini/Desktop/Call_analytics/poc-call-analytics-data_retrieval-embeddings
```

You can verify you're in the correct directory by checking if `pyproject.toml` exists:
```bash
ls pyproject.toml
```

If you see an error like "Poetry could not find a pyproject.toml file", it means you're in the wrong directory!

## Project Overview
This project is a call analytics system that processes and analyzes call data using Python, with dependencies managed by Poetry. The project includes several Jupyter notebooks for data analysis, feature extraction, and machine learning tasks.

## Setup Process

### 1. Environment Setup
1. **Python Version Requirement**
   - The project requires Python 3.10 (specifically ^3.10,<3.11)
   - We initially faced issues with Python 3.12, which was incompatible

2. **Poetry Installation**
   ```bash
   # Install Poetry using pip
   pip install poetry
   ```

3. **Project Directory Structure**
   ```
   poc-call-analytics-data_retrieval-embeddings/  # ⚠️ YOU MUST BE IN THIS DIRECTORY
   ├── notebooks/          # Jupyter notebooks
   ├── scripts/           # Python scripts
   ├── src/              # Source code
   │   └── call_analytics/
   ├── data/             # Data directory
   ├── pyproject.toml    # Poetry configuration (⚠️ This file must exist in your current directory)
   └── poetry.lock       # Locked dependencies
   ```

### 2. Installation Steps
1. **Navigate to Project Directory** (⚠️ THIS STEP IS CRITICAL)
   ```bash
   # You MUST run this command first
   cd /Users/sunilsaini/Desktop/Call_analytics/poc-call-analytics-data_retrieval-embeddings
   
   # Verify you're in the correct directory
   ls pyproject.toml  # Should show pyproject.toml
   ```

2. **Install Dependencies**
   ```bash
   # Only run this after you're in the correct directory
   poetry install
   ```

3. **Install Jupyter Kernel**
   ```bash
   # Only run this after you're in the correct directory
   poetry run python -m ipykernel install --user --name call-analytics --display-name "Python (call-analytics)"
   ```

## Running the Notebooks

### 1. Starting Jupyter Notebook (⚠️ READ THIS CAREFULLY)
```bash
# ⚠️ FIRST, make sure you're in the correct directory
cd /Users/sunilsaini/Desktop/Call_analytics/poc-call-analytics-data_retrieval-embeddings

# ⚠️ THEN start Jupyter Notebook
poetry run jupyter notebook
```

If you see this error:
```
Poetry could not find a pyproject.toml file in /Users/sunilsaini/Desktop/Call_analytics or its parents
```
It means you're in the wrong directory! Go back to step 1 and make sure you're in the correct directory.

### 2. Important Notes for Running Notebooks
1. **Kernel Selection**
   - Always select "Python (call-analytics)" kernel
   - Go to Kernel → Change kernel → Python (call-analytics)

2. **Notebook Order**
   Run notebooks in sequence:
   1. "1 - DATA RETRIEVAL - Aircall.ipynb"
   2. "2 - EDA - Aircall.ipynb"
   3. "3 - OPENAI - Get additional features.ipynb"
   4. "4 - OPENAI - Get embeddings.ipynb"
   5. "5 - CLUSTERING - OpenAI embeddings.ipynb"
   6. "6 - LABELED DATA.ipynb"
   7. "7 - EDA - Labels.ipynb"
   8. "7_1 - CLASSIFICATION - Multi-label.ipynb"
   And so on...

## Issues Faced and Solutions

### 1. Python Version Mismatch
- **Issue**: Project required Python 3.10 but system had Python 3.12
- **Solution**: Used Poetry to create a virtual environment with Python 3.10

### 2. Poetry Environment Issues
- **Issue**: "Poetry could not find a pyproject.toml file"
- **Solution**: Always run Poetry commands from the project root directory where pyproject.toml exists

### 3. Jupyter Kernel Issues
- **Issue**: Kernel not found or incorrect kernel being used
- **Solution**: 
  1. Removed old kernels
  2. Installed new kernel in Poetry environment
  3. Selected correct kernel in notebooks

### 4. Import Issues
- **Issue**: ModuleNotFoundError for src.settings
- **Solution**: Changed import from `src.settings` to `call_analytics.settings`

### 5. Jupyter Notebook Server Issues
- **Issue**: Port 8888 already in use
- **Solution**: Jupyter automatically switched to port 8889

## Common Commands

### Poetry Commands
```bash
# Activate Poetry environment
poetry shell

# Run a specific script
poetry run python scripts/your_script.py

# Update dependencies
poetry update

# Show environment info
poetry env info
```

### Jupyter Commands
```bash
# List available kernels
jupyter kernelspec list

# Remove a kernel
jupyter kernelspec remove kernel_name -f

# Start Jupyter Notebook
poetry run jupyter notebook
```

## Troubleshooting

1. **If Poetry commands fail**:
   - Verify you're in the correct directory (where pyproject.toml exists)
   - Check Python version: `poetry env info`

2. **If imports fail**:
   - Make sure you're using the correct kernel
   - Verify the package is installed: `poetry show`

3. **If Jupyter Notebook doesn't start**:
   - Check if another instance is running
   - Try a different port: `poetry run jupyter notebook --port 8889`

4. **If kernel connection fails**:
   - Restart the kernel
   - Verify kernel installation
   - Check Poetry environment

## Additional Notes
- Always run notebooks in sequence as they depend on each other
- Keep the Poetry environment activated while working
- Make sure to use the correct kernel for all notebooks
- The project uses Poetry for dependency management, so avoid using pip directly 