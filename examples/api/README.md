# Research Agent API

A RESTful API version of the Research Agent that analyzes PDF research papers using AI.

## Features

- PDF research paper analysis endpoint
- Human-readable summary generation
- Health check endpoint
- Configurable host and port
- Debug mode support
- Comprehensive logging

## Requirements

- Python 3.x
- FastAPI
- uvicorn
- python-dotenv
- research_agent module
- jinja2

## Installation

1. Install the required packages using requirements.txt:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env` (optional if using command line arguments):
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL_NAME=your_model_name_here
OPENAI_API_BASE=your_api_base_here (optional)
```

## Usage

### Starting the Server

Run the server with default settings (localhost:9696):
```bash
python research_api.py
```

Run with custom host and port:
```bash
python research_api.py --host 0.0.0.0 --port 8000
```

Run with OpenAI settings:
```bash
python research_api.py --api-key your_key --model gpt-4 --api-base custom_url
```

Enable debug mode:
```bash
python research_api.py --debug
```

### Command Line Arguments

- `--host`: Host to run the server on (default: 127.0.0.1)
- `--port`: Port to run the server on (default: 9696)
- `--debug`: Enable debug mode
- `--api-key`: OpenAI API key
- `--model`: OpenAI model name
- `--api-base`: OpenAI API base URL

### API Endpoints

1. **Analyze Paper** (POST `/analyze`)
   - Endpoint: `http://localhost:9696/analyze`
   - Method: POST
   - Content-Type: multipart/form-data
   - Body: PDF file
   - Response: JSON containing raw analysis and human-readable summary

   Example using curl:
   ```bash
   curl -X POST "http://localhost:9696/analyze" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@path/to/your/paper.pdf"
   ```

2. **Health Check** (GET `/health`)
   - Endpoint: `http://localhost:9696/health`
   - Method: GET
   - Response: JSON with API health status

   Example using curl:
   ```bash
   curl "http://localhost:9696/health"
   ```

### Response Format

The `/analyze` endpoint returns a JSON response with two main sections:

1. `raw_analysis`: The complete analysis data
2. `summary`: A structured, human-readable summary containing:
   - Research Objectives
   - Methodology
   - Key Findings
   - Conclusions
   - Validity Assessment
   - Credibility Assessment
   - Limitations
   - Counter Arguments

Example response:
```json
{
  "raw_analysis": {
    "objectives": "...",
    "methodology": "...",
    "key_findings": "...",
    ...
  },
  "summary": {
    "title": "Research Paper Analysis Summary",
    "sections": {
      "objectives": {
        "title": "Research Objectives",
        "content": "..."
      },
      ...
    }
  }
}
```

### Documentation

The API includes a built-in documentation page available at the root URL (`http://localhost:9696/`). This page provides:
- Interactive API documentation
- Example requests and responses
- Error handling information
- Command line usage examples

## Error Handling

The API includes comprehensive error handling:

- 400 Bad Request: Invalid file format (non-PDF)
- 503 Service Unavailable: ResearchAgent not initialized
- 500 Internal Server Error: Analysis failures

## Logging

Logs are written to both console and a file named `research_api_YYYYMMDD_HHMMSS.log`.
