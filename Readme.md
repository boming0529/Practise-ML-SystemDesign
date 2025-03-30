# Text Processing Environment

This project sets up a Python environment for text processing tasks, including tokenization, lemmatization, and stemming. It uses `nltk` for traditional NLP tasks and `transformers` for modern tokenization (e.g., BERT).

## Requirements

- Python 3.8 or higher
- CPU (GPU optional for model inference with `transformers`)

## Installation

- **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # Linux/macOS
   ```


- **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

- **Download NLTK resources: Run the following Python script to download required NLTK data:**
    ```bash
    python -m download_workNet.py
    ```

## Usage
```bash
python -m main.py
```