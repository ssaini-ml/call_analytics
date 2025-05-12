# Call Analytics - Conversational Intelligence Scripts

This directory contains Python scripts for retrieving and processing call data from Aircall, with a focus on conversation intelligence features.

## Setup

1. Make sure you have Poetry installed:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Set up your environment variables:
```bash
cp ../template.env .env
# Edit .env with your credentials
```

## Available Scripts

### 1. data_retrieval.py

Retrieves call data from Aircall API and processes it for analysis.

#### Usage

```bash
poetry run python data_retrieval.py --business CS --start-date 2025-03-08 --end-date 2025-03-09
```

#### Arguments

- `--business`: Business unit to analyze (required)
  - Choices: 'CS' (Customer Service), 'PS' (Pharma Service), 'OTHER'
- `--start-date`: Start date in YYYY-MM-DD format (required)
- `--end-date`: End date in YYYY-MM-DD format (required)
- `--sample-size`: Number of calls to download CI features for (default: 2)
- `--custom-numbers`: Custom phone numbers to analyze (required if business is OTHER)
- `--ci-features`: Conversational Intelligence features to retrieve
  - Default: ["sentiments", "summary", "topics", "transcription"]

#### Examples

1. Analyze Customer Service calls:
```bash
poetry run python data_retrieval.py --business CS --start-date 2025-03-08 --end-date 2025-03-09
```

2. Analyze Pharma Service calls:
```bash
poetry run python data_retrieval.py --business PS --start-date 2025-03-08 --end-date 2025-03-09
```

3. Analyze custom phone numbers:
```bash
poetry run python data_retrieval.py --business OTHER --start-date 2025-03-08 --end-date 2025-03-09 --custom-numbers "+31 85 888 1579" "+31 85 888 1529"
```

### 2. eda.py

Performs exploratory data analysis on the processed call data, generating statistics and visualizations.

#### Usage

```bash
poetry run python eda.py --input-file 20250113_20250212_CS.csv
```

#### Arguments

- `--input-file`: Input CSV file path (required)
  - Path should be relative to PATH_AIRCALL_PROCESSED
- `--output-dir`: Directory to save plots (default: 'plots')
- `--save-plots`: Flag to save plots to output directory
  - If not set, plots will be displayed interactively

#### Examples

1. Basic analysis with interactive plots:
```bash
poetry run python eda.py --input-file 20250113_20250212_CS.csv
```

2. Save plots to a specific directory:
```bash
poetry run python eda.py --input-file 20250113_20250212_CS.csv --output-dir analysis_plots --save-plots
```

#### Output

The EDA script generates:

1. **Summary Statistics**:
   - Call volume by phone number
   - Recording, transcription, summary, topics, and sentiment counts
   - Call duration statistics
   - Token estimates for CI features

2. **Visualizations**:
   - Calls per day (bar plot)
   - Call duration distribution (histogram)
   - Call duration by day of week (box plot)

## Data Storage

- Raw call data: `data/aircall/calls/`
- Processed data: `data/aircall/processed/`
- CI features:
  - Sentiments: `data/aircall/sentiments/`
  - Summaries: `data/aircall/summaries/`
  - Topics: `data/aircall/topics/`
  - Transcriptions: `data/aircall/transcriptions/`
- Analysis plots: `plots/` (when using --save-plots)

## Troubleshooting

1. **Authentication Errors**:
   - Verify your credentials in `.env`
   - Check API rate limits

2. **Directory Errors**:
   - Ensure you have write permissions
   - Scripts will create necessary directories

3. **Data Processing Errors**:
   - Check input date format
   - Verify phone number format
   - Ensure sufficient disk space

4. **Plot Generation Errors**:
   - Check if matplotlib backend is properly configured
   - Verify input data format
   - Ensure output directory is writable

## Development

- Use Poetry for dependency management
- Follow PEP 8 style guide
- Run tests with `poetry run pytest`
- Format code with `poetry run black .`
- Sort imports with `poetry run isort .`
- Type check with `poetry run mypy .` 