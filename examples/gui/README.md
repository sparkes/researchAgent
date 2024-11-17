# Research Agent GUI Example

A graphical user interface for the ResearchAgent that allows users to analyze PDF papers with a friendly interface.

## Features

- File selection dialog for choosing PDF files
- Debug mode toggle
- Tabbed results display for easy reading
- Progress indication during analysis
- Error handling with user-friendly messages
- Detailed logging

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
   python research_gui.py
   ```

3. Using the application:
   - Click "Browse" to select a PDF file
   - Toggle "Debug Mode" if you want detailed logging
   - Click "Analyze Paper" to start the analysis
   - Results will appear in tabs below
   - Check the log file (research_gui_[timestamp].log) for detailed operation logs

## Error Handling

The application includes comprehensive error handling for:
- Missing environment variables
- Invalid PDF files
- File not found errors
- API errors
- Analysis failures

All errors are logged and displayed to the user in friendly message boxes.
