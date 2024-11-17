# Command Line Example

This example demonstrates how to use the ResearchAgent from the command line to analyze PDF papers.

## Usage

```bash
python example.py [--debug] path/to/paper.pdf
```

### Arguments

- `path/to/paper.pdf`: (Required) Path to the PDF file you want to analyze
- `--debug`: (Optional) Enable debug mode for more detailed logging

### Examples

```bash
# Analyze a paper with normal logging
python example.py paper.pdf

# Analyze a paper with debug logging enabled
python example.py --debug paper.pdf
```

### Environment Variables

Make sure you have the following environment variables set:
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL_NAME`: The name of the OpenAI model to use (e.g., "gpt-4-1106-preview")
- `OPENAI_API_BASE` : the base URL for OpenAI API requests