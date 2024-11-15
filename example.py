from research_agent import ResearchAgent
import os
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    try:
        # Initialize the research agent with custom API configuration and debug mode off by default
        agent = ResearchAgent(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            model_name=os.getenv("OPENAI_MODEL_NAME"),
            debug=False  # Set to True to enable debug logging
        )
        
        # Example with a PDF file
        pdf_path = "./test.pdf"
        print(f"\nAnalyzing paper: {pdf_path}")
        print("=" * 50)
        
        # Analyze the paper
        report = agent.analyze_paper(pdf_path)
        
        # Print the report
        print("\nAnalysis Report:")
        print("-" * 50)
        print(report)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
