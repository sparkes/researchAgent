# Research Agent API

A RESTful API version of the Research Agent that analyzes PDF research papers using AI.

## Features

- RESTful endpoints for paper analysis
- PDF file upload and processing
- Human-readable summary generation
- Health check endpoint
- Interactive documentation page
- Configurable host and port
- Debug mode support
- Advanced logging system:
  - Automatic log rotation
  - 5 files maximum
  - 1MB per file limit
  - Timestamped log files
  - Detailed operation tracking

## Requirements

- Python 3.x
- FastAPI
- uvicorn
- python-multipart
- python-dotenv
- jinja2
- pydantic>=2.5.3
- starlette>=0.27.0
- research_agent module

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env` (optional if using command line arguments):
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL_NAME=your_model_name_here
OPENAI_API_BASE=your_api_base_here
```

## Usage

### Starting the Server

Basic start (localhost:9696):
```bash
python research_api.py
```

Custom host and port:
```bash
python research_api.py --host 0.0.0.0 --port 8000
```

With debug mode:
```bash
python research_api.py --debug
```

With OpenAI settings:
```bash
python research_api.py --api-key YOUR_KEY --model MODEL_NAME --api-base API_BASE
```

### API Endpoints

- `GET /`: Interactive documentation page
- `GET /health`: Health check endpoint
- `POST /analyze`: Analyze PDF paper
  - Accepts PDF file upload
  - Returns detailed analysis report

### Documentation

The API includes a built-in documentation page at the root URL (`http://localhost:9696/`). This page provides:
- Interactive API documentation
- Example requests and responses
- Error handling information
- Command line usage examples

### Error Handling

The API includes comprehensive error handling:
- 400 Bad Request: Invalid file format
- 503 Service Unavailable: Agent not initialized
- 500 Internal Server Error: Analysis failures

### Logging

- Logs are written to `research_api_YYYYMMDD_HHMMSS.log`
- Automatic rotation after 1MB
- Keeps last 5 log files
- Debug mode enables verbose logging
- Production mode shows INFO and above
- Third-party library logging is suppressed
