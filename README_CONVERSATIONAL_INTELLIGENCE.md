# Call Analytics - Conversational Intelligence

This project provides tools for analyzing call data using conversational intelligence features, including sentiment analysis, topic extraction, and transcription analysis.

## Project Structure

```
poc-call-analytics-data_retrieval-conversational_intelligence/
├── src/
│   └── call_analytics_conversational/
│       ├── __init__.py          # Package initialization
│       ├── api_utils.py         # Aircall API utilities
│       ├── conversation.py      # Conversation analysis
│       ├── pii_utility.py       # PII redaction utilities
│       ├── settings.py          # Configuration settings
│       └── utils.py             # Helper functions
├── scripts/
│   ├── data_retrieval.py       # Data retrieval script
│   └── eda.py                  # Exploratory data analysis
└── data/
    └── aircall/
        ├── calls/              # Raw call data
        ├── processed/          # Processed data
        ├── sentiments/         # Sentiment analysis
        ├── summaries/          # Call summaries
        ├── topics/             # Topic extraction
        └── transcriptions/     # Call transcriptions
```

## Core Components

### 1. Data Retrieval (`scripts/data_retrieval.py`)

This script retrieves and processes call data from the Aircall API. Key features:

- **Command Line Interface**:
  ```bash
  poetry run python scripts/data_retrieval.py --business CS --start-date 2025-03-08 --end-date 2025-03-09
  ```

- **Functionality**:
  - Retrieves call data for specified business units (CS, PS, or custom numbers)
  - Handles API pagination and rate limiting
  - Processes and stores call metadata
  - Downloads and processes conversational intelligence features:
    - Sentiment analysis
    - Call summaries
    - Topic extraction
    - Transcriptions

- **Key Functions**:
  - `retrieve_calls()`: Fetches call data from Aircall API
  - `process_and_save_calls()`: Processes and saves call metadata
  - `setup_directories()`: Creates necessary data directories

### 2. Exploratory Data Analysis (`scripts/eda.py`)

Performs exploratory data analysis on the processed call data. Key features:

- **Command Line Interface**:
  ```bash
  poetry run python scripts/eda.py --input-file 20250308_20250309_CS.csv --save-plots
  ```

- **Functionality**:
  - Loads and preprocesses call data
  - Generates summary statistics
  - Creates visualizations:
    - Calls per day
    - Call duration distribution
    - Call duration by day of week
  - Calculates token estimates for CI features

- **Key Functions**:
  - `load_and_preprocess_data()`: Loads and prepares data for analysis
  - `plot_calls_per_day()`: Visualizes call volume over time
  - `plot_call_duration_histogram()`: Shows call duration distribution
  - `plot_call_duration_by_day()`: Analyzes call patterns by day
  - `generate_summary_statistics()`: Creates comprehensive statistics

### 3. Core Modules

#### API Utilities (`api_utils.py`)
- Handles Aircall API interactions
- Manages rate limiting and pagination
- Provides request utilities

#### Conversation Analysis (`conversation.py`)
- `Conversation` class for managing call data
- Handles loading and formatting of:
  - Call summaries
  - Topics
  - Transcriptions
  - Sentiment analysis
- Formats conversation data for analysis

#### PII Utility (`pii_utility.py`)
- Redacts Personally Identifiable Information
- Uses Azure Language Service
- Handles text chunking and batching
- Async processing for efficiency

#### Settings (`settings.py`)
- Configuration management
- Environment variables
- API credentials
- File paths
- Azure service settings

#### Utilities (`utils.py`)
- Helper functions
- Date handling
- Data processing utilities

## Data Flow

1. **Data Retrieval**:
   - Script fetches call data from Aircall API
   - Processes and stores metadata
   - Downloads CI features
   - Stores in structured directories

2. **Data Processing**:
   - PII redaction
   - Text formatting
   - Feature extraction
   - Data validation

3. **Analysis**:
   - Statistical analysis
   - Visualization
   - Pattern identification
   - Token estimation

## Usage Examples

1. **Retrieve Call Data**:
   ```bash
   # Customer Service calls
   poetry run python scripts/data_retrieval.py --business CS --start-date 2025-03-08 --end-date 2025-03-09
   
   # Pharma Service calls
   poetry run python scripts/data_retrieval.py --business PS --start-date 2025-03-08 --end-date 2025-03-09
   
   # Custom numbers
   poetry run python scripts/data_retrieval.py --business OTHER --start-date 2025-03-08 --end-date 2025-03-09 --custom-numbers "+31 85 888 1579"
   ```

2. **Analyze Data**:
   ```bash
   # Basic analysis with interactive plots
   poetry run python scripts/eda.py --input-file 20250308_20250309_CS.csv
   
   # Save plots to directory
   poetry run python scripts/eda.py --input-file 20250308_20250309_CS.csv --output-dir analysis_plots --save-plots
   ```

## Dependencies

- Python 3.8+
- Poetry for dependency management
- Azure Language Service
- Aircall API access
- Required Python packages (see `pyproject.toml`)

## Environment Setup

1. Copy template environment file:
   ```bash
   cp template.env .env
   ```

2. Configure environment variables:
   - Aircall API credentials
   - Azure Language Service credentials
   - File paths
   - Other settings

3. Install dependencies:
   ```bash
   poetry install
   ```

## Contributing

1. Follow PEP 8 style guide
2. Use Poetry for dependency management
3. Add tests for new features
4. Update documentation
5. Use type hints
6. Run linting tools:
   ```bash
   poetry run black .
   poetry run isort .
   poetry run mypy .
   ``` 