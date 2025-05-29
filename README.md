# PDF Page Ordering Agents

Advanced PDF processing with AI-powered page ordering capabilities.

## Features

- OCR processing for scanned PDFs
- Intelligent page ordering for unorganized documents
- FastAPI web service with simple endpoints
- Support for both text-based and scanned PDFs

## Installation

```bash
pip install -e .
```

## API Endpoints

- `/process-pdf`: Process PDF and return ordered pages
- `/analyze-pdf`: Analyze PDF structure without ordering
- `/health`: Health check endpoint

## Usage

Start the API server:

```bash
python -m src.api.main
```

The API will be available at http://localhost:8000

## License

MIT 