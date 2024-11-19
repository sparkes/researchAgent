import os
import sys
import logging
import argparse
from pathlib import Path

# Add the parent directory to sys.path to allow importing research_agent
sys.path.append(str(Path(__file__).resolve().parents[2]))

from research_agent import ResearchAgent
from dotenv import load_dotenv

def setup_logging(level=logging.INFO):
    """Set up logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.handlers.RotatingFileHandler('research_agent.log', maxBytes=1000000, backupCount=5)  # Output to file with rotation
        ]
    )
    # Set logging levels for specific modules
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    
    # Suppress verbose PDF library logging
    logging.getLogger('pdfminer').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    logging.getLogger('pdfplumber').setLevel(logging.WARNING)
    logging.getLogger('PyPDF2').setLevel(logging.WARNING)
    
    # Also suppress other verbose libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('scholarly').setLevel(logging.WARNING)
    logging.getLogger('bs4').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

def main(debug=False, pdf_path=None):
    # Set up logging first
    if debug:
        setup_logging(logging.DEBUG)
    else:
        setup_logging(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info("Starting research agent example")

    try:
        # Load environment variables
        load_dotenv()
        
        # Get required environment variables
        api_key = os.getenv('OPENAI_API_KEY')
        model_name = os.getenv('OPENAI_MODEL_NAME')
        
        # Validate required environment variables
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if not model_name:
            raise ValueError("OPENAI_MODEL_NAME not found in environment variables")
            
        logger.info("Environment variables loaded successfully")
        logger.debug(f"Using model: {model_name}")

        # Initialize the research agent with debug mode
        logger.info("Initializing ResearchAgent")
        agent = ResearchAgent(
            api_key=api_key,
            model_name=model_name,
            debug=debug
        )
        
        # Validate PDF path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")
        logger.info(f"Found PDF at {pdf_path}")

        # Analyze the paper
        logger.info(f"Starting analysis of {pdf_path}")
        report = agent.analyze_paper(pdf_path)
        
        # Log and print the results
        logger.info("Analysis completed successfully")
        logger.debug(f"Analysis report: {report}")
        print("\nAnalysis Report:")
        print("=" * 50)
        for section, content in report.items():
            print(f"\n{section.replace('_', ' ').title()}:")
            print("-" * 40)
            print(content)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Research Agent Paper Analysis')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('pdf', type=str, help='Path to the PDF file to analyze')
    
    args = parser.parse_args()
    main(debug=args.debug, pdf_path=args.pdf)
