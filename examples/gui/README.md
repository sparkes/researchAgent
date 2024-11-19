# Research Agent GUI Example

A graphical user interface for the ResearchAgent that allows users to analyze PDF papers with a friendly interface.

## Features

- Modern graphical interface
- File selection dialog for PDF files
- Debug mode toggle
- Real-time progress indication
- Tabbed results display:
  - Metadata overview
  - Validity assessment
  - Credibility analysis
  - Retraction status
  - Counter arguments
  - Summary
- Advanced logging system:
  - Automatic log rotation
  - 5 files maximum
  - 1MB per file limit
  - Timestamped log files
  - Real-time log display
- Comprehensive error handling
- Queue-based log processing

## Requirements

- Python 3.x
- tkinter (usually comes with Python)
- All ResearchAgent dependencies

## Usage

1. Make sure you have all required environment variables set in your `.env` file:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `OPENAI_MODEL_NAME`: The name of the OpenAI model to use
   - `OPENAI_API_BASE`: The base URL for OpenAI API requests

2. Run the GUI application:
   ```bash
   python research_gui.py [--debug]
   ```

3. Using the application:
   - Click "Browse" to select a PDF file
   - Toggle "Debug Mode" for detailed logging
   - Click "Analyze Paper" to start analysis
   - View results in organized tabs
   - Monitor progress in real-time
   - Check logs in the logging panel

### Logging

- Logs are written to `research_gui_YYYYMMDD_HHMMSS.log`
- Automatic rotation after 1MB
- Keeps last 5 log files
- Debug mode enables verbose logging
- Production mode shows INFO and above
- Real-time log display in GUI
- Third-party library logging is suppressed

### Error Handling

The application includes comprehensive error handling for:
- Missing environment variables
- Invalid PDF files
- File not found errors
- API errors
- Analysis failures
- Queue processing errors

All errors are:
- Logged to rotating log files
- Displayed in the GUI log panel
- Shown in user-friendly message boxes
