# Call Analytics - Conversational Intelligence
# Setup and Usage Instructions

## Prerequisites

1. Python 3.10 or higher
2. Poetry (Python package manager)
3. Git (for version control)
4. Aircall API credentials
5. Azure Language Service credentials

## Initial Setup

1. **Navigate to the Project Directory**
   ```bash
   cd /Users/sunilsaini/Desktop/Call_analytics/poc-call-analytics-data_retrieval-conversational_intelligence
   ```
   > **Important**: Always run commands from this directory. The `pyproject.toml` file must be present.

2. **Install Poetry** (if not already installed)
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Set Up Environment Variables**
   ```bash
   # Step 1: Copy the template environment file
   cp template.env .env
   
   # Step 2: Edit .env with your credentials
   # Open .env in your preferred editor and replace the placeholder values:
   # - AIRCALL_API_ID: Your Aircall API ID
   # - AIRCALL_API_TOKEN: Your Aircall API token
   # - AZURE_LANGUAGE_ENDPOINT: Your Azure Language Service endpoint URL
   # - AZURE_LANGUAGE_KEY: Your Azure Language Service key
   
   # Example .env file:
   # AIRCALL_API_ID=abc123
   # AIRCALL_API_TOKEN=xyz789
   # AZURE_LANGUAGE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
   # AZURE_LANGUAGE_KEY=your-azure-key-here
   ```

4. **Verify Environment Setup**
   ```bash
   # Check if .env file exists and has the correct permissions
   ls -l .env
   
   # Should show something like:
   # -rw-r--r--  1 user  group  1234 Mar 8 10:00 .env
   ```

5. **Install Dependencies**
   ```bash
   # Install all required packages
   poetry install
   
   # Install Jupyter kernel for notebooks
   poetry run python -m ipykernel install --user --name call-analytics --display-name "Python (call-analytics)"
   ```

6. **Verify Installation**
   ```bash
   # Check if poetry environment is properly set up
   poetry env info
   
   # Should show your Python version and environment path
   ```

## Running the Scripts

### 1. Data Retrieval

To retrieve call data from Aircall:

```bash
# Make sure you're in the project directory
cd /Users/sunilsaini/Desktop/Call_analytics/poc-call-analytics-data_retrieval-conversational_intelligence

# Run the data retrieval script
poetry run python scripts/data_retrieval.py --business CS --start-date 2025-03-08 --end-date 2025-03-09
```

Common options:
- `--business`: Choose from 'CS' (Customer Service), 'PS' (Pharma Service), or 'OTHER'
- `--start-date`: Start date in YYYY-MM-DD format
- `--end-date`: End date in YYYY-MM-DD format
- `--sample-size`: Number of calls to download (default: 2)
- `--custom-numbers`: Required if business is 'OTHER'

Example for custom numbers:
```bash
poetry run python scripts/data_retrieval.py --business OTHER --start-date 2025-03-08 --end-date 2025-03-09 --custom-numbers "+31 85 888 1579" "+31 85 888 1529"
```

### 2. Exploratory Data Analysis (EDA)

To analyze the retrieved data:

```bash
# Make sure you're in the project directory
cd /Users/sunilsaini/Desktop/Call_analytics/poc-call-analytics-data_retrieval-conversational_intelligence

# Run EDA with interactive plots
poetry run python scripts/eda.py --input-file 20250113_20250212_CS.csv

# Or save plots to a directory
poetry run python scripts/eda.py --input-file 20250113_20250212_CS.csv --output-dir analysis_plots --save-plots
```

Common options:
- `--input-file`: Name of the CSV file in the processed data directory
- `--output-dir`: Directory to save plots (default: 'plots')
- `--save-plots`: Flag to save plots instead of displaying them

## Directory Structure

```
poc-call-analytics-data_retrieval-conversational_intelligence/
├── data/
│   └── aircall/
│       ├── calls/          # Raw call data
│       ├── processed/      # Processed data
│       ├── sentiments/     # Sentiment analysis
│       ├── summaries/      # Call summaries
│       ├── topics/         # Call topics
│       └── transcriptions/ # Call transcriptions
├── notebooks/             # Jupyter notebooks
├── scripts/              # Python scripts
│   ├── data_retrieval.py # Data retrieval script
│   └── eda.py           # EDA script
├── src/                  # Source code
├── .env                  # Environment variables (create from template.env)
├── pyproject.toml        # Poetry dependencies
└── INSTRUCTIONS.md       # This file
```

## Common Issues and Solutions

1. **"Poetry could not find pyproject.toml"**
   - Make sure you're in the correct directory
   - Verify you're in `/Users/sunilsaini/Desktop/Call_analytics/poc-call-analytics-data_retrieval-conversational_intelligence`
   - Check that `pyproject.toml` exists in the current directory

2. **Authentication Errors**
   - Verify your credentials in `.env`
   - Check API rate limits
   - Ensure all required environment variables are set

3. **Directory Errors**
   - Ensure you have write permissions
   - Scripts will create necessary directories automatically
   - Check available disk space

4. **Data Processing Errors**
   - Verify input date format (YYYY-MM-DD)
   - Check phone number format
   - Ensure CSV files are properly formatted

## Development

1. **Code Formatting**
   ```bash
   # Format code
   poetry run black .
   
   # Sort imports
   poetry run isort .
   ```

2. **Type Checking**
   ```bash
   poetry run mypy .
   ```

3. **Running Tests**
   ```bash
   poetry run pytest
   ```

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your environment setup
3. Check the API documentation:
   - [Aircall API Reference](https://developer.aircall.io/api-references)
   - [Azure Language Service](https://learn.microsoft.com/en-us/azure/cognitive-services/language-service/) 