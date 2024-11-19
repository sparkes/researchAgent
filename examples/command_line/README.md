# Command Line Example

This example demonstrates how to use the ResearchAgent from the command line to analyze PDF papers.

## Features

- Simple command-line interface
- Support for PDF file analysis
- Debug mode for detailed logging
- Automatic log rotation (5 files, 1MB each)
- Suppressed verbose library logging
- Comprehensive error handling

## Usage

```bash
python example.py [--debug] path/to/paper.pdf
```

### Arguments

- `path/to/paper.pdf`: (Required) Path to the PDF file to analyze
- `--debug`: (Optional) Enable debug mode for detailed logging

### Examples

```bash
# Analyze a paper with normal logging
python example.py paper.pdf

# Analyze a paper with debug logging enabled
python example.py --debug paper.pdf
```

### Logging

- Logs are written to 'research_agent.log'
- Automatic rotation after 1MB
- Keeps last 5 log files
- Debug mode enables verbose logging
- Production mode shows INFO and above
- Third-party library logging is suppressed

### Environment Variables

Make sure you have the following environment variables set:
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL_NAME`: The name of the OpenAI model to use (e.g., "gpt-4-1106-preview")
- `OPENAI_API_BASE`: The base URL for OpenAI API requests