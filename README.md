# Call Analytics - Embeddings Analysis

This project provides tools for analyzing call data using embeddings and machine learning techniques, including clustering, classification, and topic modeling.

## Project Structure

```
poc-call-analytics-data_retrieval-embeddings/
├── src/
│   └── call_analytics_embeddings/
│       ├── __init__.py          # Package initialization
│       ├── api_utils.py         # Aircall API utilities
│       ├── embeddings.py        # Embedding generation and analysis
│       ├── classification.py    # Classification models
│       ├── clustering.py        # Clustering analysis
│       ├── pii_utility.py       # PII redaction utilities
│       ├── settings.py          # Configuration settings
│       └── utils.py             # Helper functions
├── scripts/
│   ├── 1 - DATA RETRIEVAL - Aircall.py
│   ├── 2 - EDA - Aircall.py
│   ├── 3 - OPENAI - Get additional features.py
│   ├── 4 - OPENAI - Get embeddings.py
│   ├── 5 - CLUSTERING - OpenAI embeddings.py
│   ├── 6 - LABELED DATA.py
│   ├── 7 - EDA - Labels.py
│   ├── 7_1 - CLASSIFICATION - Multi-label.py
│   ├── 7_1_1 - CLASSIFICATION - Multi-class - Flat.py
│   ├── 7_1_2 - CLASSIFICATION - Multi-class - Flat.py
│   ├── 7_2_1 - CLASSIFICATION - Multi-class - Hierarchical.py
│   ├── 7_2_2 - CLASSIFICATION - Multi-class - Hierarchical.py
│   ├── 8_1 - TAGGING - Binary classification.py
│   └── 999 - PII Check.py
└── data/
    └── aircall/
        ├── calls/              # Raw call data
        ├── processed/          # Processed data
        ├── embeddings/         # Generated embeddings
        ├── labels/            # Classification labels
        └── models/            # Trained models
```

## Core Components

### 1. Data Retrieval and Processing

#### Script: `1 - DATA RETRIEVAL - Aircall.py`
- Retrieves call data from Aircall API
- Processes and stores call metadata
- Downloads transcriptions and summaries
- Handles PII redaction

#### Script: `2 - EDA - Aircall.py`
- Performs exploratory data analysis
- Generates call statistics
- Creates visualizations
- Analyzes call patterns

### 2. Feature Engineering

#### Script: `3 - OPENAI - Get additional features.py`
- Uses OpenAI to extract additional features
- Generates call summaries
- Extracts key topics
- Performs sentiment analysis

#### Script: `4 - OPENAI - Get embeddings.py`
- Generates embeddings using OpenAI
- Processes call transcriptions
- Creates vector representations
- Stores embeddings for analysis

### 3. Clustering Analysis

#### Script: `5 - CLUSTERING - OpenAI embeddings.py`
- Performs clustering on embeddings
- Uses various algorithms:
  - K-means
  - DBSCAN
  - Hierarchical clustering
- Visualizes clusters
- Analyzes cluster characteristics

### 4. Classification Pipeline

#### Script: `6 - LABELED DATA.py`
- Prepares labeled data for classification
- Handles data splitting
- Creates training/validation sets
- Manages label encoding

#### Script: `7 - EDA - Labels.py`
- Analyzes label distribution
- Visualizes class balance
- Examines label correlations
- Prepares for classification

#### Multi-label Classification
- Script: `7_1 - CLASSIFICATION - Multi-label.py`
- Handles multiple labels per call
- Uses binary classification approach
- Evaluates model performance
- Generates predictions

#### Multi-class Classification (Flat)
- Scripts: `7_1_1` and `7_1_2 - CLASSIFICATION - Multi-class - Flat.py`
- Single-label classification
- Direct class prediction
- Performance evaluation
- Model comparison

#### Multi-class Classification (Hierarchical)
- Scripts: `7_2_1` and `7_2_2 - CLASSIFICATION - Multi-class - Hierarchical.py`
- Hierarchical classification
- Parent-child relationships
- Cascading predictions
- Tree-based evaluation

#### Binary Classification
- Script: `8_1 - TAGGING - Binary classification.py`
- Binary classification tasks
- Tag prediction
- Threshold optimization
- Performance metrics

### 5. Utility Scripts

#### Script: `999 - PII Check.py`
- Validates PII redaction
- Checks for sensitive information
- Ensures data privacy
- Generates compliance reports

## Data Flow

1. **Data Collection**:
   - Retrieve call data from Aircall
   - Process and clean data
   - Redact PII
   - Store structured data

2. **Feature Generation**:
   - Generate OpenAI embeddings
   - Extract additional features
   - Create vector representations
   - Store processed features

3. **Analysis Pipeline**:
   - Perform clustering
   - Generate labels
   - Train classification models
   - Evaluate performance

4. **Model Deployment**:
   - Save trained models
   - Generate predictions
   - Create analysis reports
   - Monitor performance

## Usage Examples

1. **Data Retrieval and Processing**:
   ```bash
   poetry run python "scripts/1 - DATA RETRIEVAL - Aircall.py"
   poetry run python "scripts/2 - EDA - Aircall.py"
   ```

2. **Feature Engineering**:
   ```bash
   poetry run python "scripts/3 - OPENAI - Get additional features.py"
   poetry run python "scripts/4 - OPENAI - Get embeddings.py"
   ```

3. **Analysis**:
   ```bash
   poetry run python "scripts/5 - CLUSTERING - OpenAI embeddings.py"
   poetry run python "scripts/7_1 - CLASSIFICATION - Multi-label.py"
   ```

## Dependencies

- Python 3.8+
- Poetry for dependency management
- OpenAI API access
- Azure Language Service
- Required Python packages (see `pyproject.toml`)

## Environment Setup

1. Copy template environment file:
   ```bash
   cp template.env .env
   ```

2. Configure environment variables:
   - OpenAI API credentials
   - Azure Language Service credentials
   - Aircall API credentials
   - File paths
   - Model parameters

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

## Model Performance

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

### Clustering Metrics
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index
- Cluster Coherence

## Best Practices

1. **Data Privacy**:
   - Always use PII redaction
   - Validate redaction effectiveness
   - Follow data protection guidelines

2. **Model Development**:
   - Use cross-validation
   - Monitor for bias
   - Regular model updates
   - Performance tracking

3. **Code Quality**:
   - Type hints
   - Documentation
   - Unit tests
   - Code review

4. **Resource Management**:
   - API rate limiting
   - Batch processing
   - Efficient storage
   - Cache management