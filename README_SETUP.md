# Call Analytics Embeddings Project Setup Guide

This guide will help you set up and run the notebooks in the embeddings project using a virtual environment.

## Prerequisites
- Python 3.12 or higher
- pip (Python package installer)
- Git (for version control)

## Setup Instructions

### 1. Create and Activate Virtual Environment

#### On macOS/Linux:
```bash
# Navigate to the project directory
cd poc-call-analytics-data_retrieval-embeddings

# Create virtual environment
python3 -m venv embeddings_env

# Activate virtual environment
source embeddings_env/bin/activate
```

#### On Windows:
```bash
# Navigate to the project directory
cd poc-call-analytics-data_retrieval-embeddings

# Create virtual environment
python -m venv embeddings_env

# Activate virtual environment
.\embeddings_env\Scripts\activate
```

### 2. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Install Jupyter and ipykernel
pip install jupyter ipykernel

# Register the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=embeddings_env --display-name="Embeddings (Python 3.12)"
```

### 3. Running the Notebooks
```bash
# Make sure your virtual environment is activated
# Start Jupyter Notebook
jupyter notebook
```

### 4. Using the Notebooks
1. Open Jupyter Notebook in your browser
2. Navigate to the `notebooks` directory
3. When opening a notebook, select the kernel "Embeddings (Python 3.12)"
4. Run the notebooks in sequence:
   - `1 - DATA RETRIEVAL - Aircall.ipynb`
   - `2 - EDA - Aircall.ipynb`
   - `3 - OPENAI - Get additional features.ipynb`
   - `4 - OPENAI - Get embeddings.ipynb`
   - And so on...

### 5. Deactivating the Virtual Environment
When you're done working:
```bash
deactivate
```

## Troubleshooting

### Common Issues and Solutions

1. **ModuleNotFoundError**
   - Make sure you're using the correct kernel in Jupyter
   - Verify the virtual environment is activated
   - Try reinstalling requirements: `pip install -r requirements.txt`

2. **CUDA/GPU Issues**
   - The project uses PyTorch with CUDA support
   - If you encounter GPU-related issues, you may need to install the CPU-only version:
     ```bash
     pip uninstall torch torchvision torchaudio
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     ```

3. **Memory Issues**
   - Some notebooks might require significant memory
   - Consider running with reduced batch sizes if needed
   - Close other memory-intensive applications

### Environment Variables
Make sure to set up your environment variables in a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
AIR_CALL_API_ID=your_aircall_id_here
AIR_CALL_API_TOKEN=your_aircall_token_here
```

## Project Structure
```
poc-call-analytics-data_retrieval-embeddings/
├── embeddings_env/           # Virtual environment
├── notebooks/               # Jupyter notebooks
├── src/                    # Source code
├── data/                   # Data directory
│   ├── aircall/           # Aircall data
│   ├── openai/            # OpenAI data
│   ├── labeled/           # Labeled data
│   └── prompts/           # Prompt templates
├── requirements.txt        # Project dependencies
└── README_SETUP.md        # This setup guide
```

## Support
If you encounter any issues not covered in this guide, please:
1. Check the error message carefully
2. Review the project's documentation
3. Contact the project maintainers 