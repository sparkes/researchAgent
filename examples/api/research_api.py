import os
import sys
import argparse 
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Optional
import tempfile
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Add the parent directory to sys.path to allow importing research_agent
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parents[1]
sys.path.append(str(root_dir))

from research_agent import ResearchAgent
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI(
    title="Research Agent API",
    description="API for analyzing research papers using AI",
    version="1.0.0"
)

# Create and mount static and templates directories
static_dir = current_dir / "static"
templates_dir = current_dir / "templates"
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Configure logging
def setup_logging(debug: bool = False):
    """Configure logging for the application"""
    log_file = f'research_api_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.handlers.RotatingFileHandler(log_file, maxBytes=1000000, backupCount=5),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Global variables
agent = None
logger = None

class AnalysisResponse(BaseModel):
    """Model for the analysis response"""
    raw_analysis: dict
    summary: dict

@app.on_event("startup")
async def startup_event():
    """Initialize the ResearchAgent on startup"""
    global agent, logger
    
    # Load environment variables
    load_dotenv()
    
    # Get debug setting from environment
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    # Setup logging
    logger = setup_logging(debug)
    logger.info("Starting Research Agent API")
    
    try:
        # Get required environment variables
        api_key = os.getenv('OPENAI_API_KEY')
        model_name = os.getenv('OPENAI_MODEL_NAME')
        api_base = os.getenv('OPENAI_API_BASE')
        
        if not all([api_key, model_name]):
            raise ValueError("Missing required environment variables")
        
        # Initialize the agent
        agent = ResearchAgent(
            api_key=api_key,
            model_name=model_name,
            api_base=api_base,
            debug=debug
        )
        logger.info("ResearchAgent initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize ResearchAgent: {str(e)}")
        raise

def create_summary(analysis_result: dict) -> dict:
    """Create a human-readable summary of the analysis results"""
    summary = {
        "title": "Research Paper Analysis Summary",
        "sections": {}
    }
    
    # Basic information
    if 'objectives' in analysis_result:
        summary["sections"]["objectives"] = {
            "title": "Research Objectives",
            "content": analysis_result['objectives']
        }
    
    if 'methodology' in analysis_result:
        summary["sections"]["methodology"] = {
            "title": "Methodology",
            "content": analysis_result['methodology']
        }
    
    if 'key_findings' in analysis_result:
        summary["sections"]["findings"] = {
            "title": "Key Findings",
            "content": analysis_result['key_findings']
        }
    
    if 'conclusions' in analysis_result:
        summary["sections"]["conclusions"] = {
            "title": "Conclusions",
            "content": analysis_result['conclusions']
        }
    
    # Validity Assessment
    if 'validity_assessment' in analysis_result:
        va = analysis_result['validity_assessment']
        if isinstance(va, dict):
            summary["sections"]["validity"] = {
                "title": "Research Validity Assessment",
                "confidence_score": va.get('confidence_score'),
                "methodology_analysis": va.get('methodology_analysis'),
                "conclusion_validity": va.get('conclusion_validity')
            }
    
    # Credibility Assessment
    if 'credibility_assessment' in analysis_result:
        ca = analysis_result['credibility_assessment']
        if isinstance(ca, dict):
            summary["sections"]["credibility"] = {
                "title": "Research Credibility Assessment",
                "credibility_score": ca.get('credibility_score'),
                "journal_reputation": ca.get('journal_reputation'),
                "peer_review_status": ca.get('peer_review_status')
            }
    
    # Additional information
    if 'limitations' in analysis_result:
        summary["sections"]["limitations"] = {
            "title": "Limitations",
            "content": analysis_result['limitations']
        }
    
    if 'counter_arguments' in analysis_result:
        summary["sections"]["counter_arguments"] = {
            "title": "Counter Arguments",
            "content": analysis_result['counter_arguments']
        }
    
    return summary

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_paper(file: UploadFile = File(...)):
    """Analyze a PDF research paper"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Create a temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name
        
        try:
            # Analyze the paper
            logger.info(f"Starting analysis of {file.filename}")
            analysis_result = agent.analyze_paper(temp_path)
            
            # Create the summary
            summary = create_summary(analysis_result)
            
            logger.info("Analysis completed successfully")
            return AnalysisResponse(
                raw_analysis=analysis_result,
                summary=summary
            )
            
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Error analyzing paper: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the API is running and the agent is initialized"""
    if agent is None:
        raise HTTPException(status_code=503, detail="ResearchAgent not initialized")
    return {"status": "healthy", "agent_initialized": True}

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Serve the documentation page"""
    try:
        return templates.TemplateResponse(
            name="index.html",
            context={"request": request}
        )
    except Exception as e:
        logger.error(f"Error serving home page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Run the API server"""
    parser = argparse.ArgumentParser(description='Research Agent API Server')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=9696, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # OpenAI settings
    parser.add_argument('--api-key', type=str, help='OpenAI API key')
    parser.add_argument('--model', type=str, help='OpenAI model name')
    parser.add_argument('--api-base', type=str, help='OpenAI API base URL')
    
    args = parser.parse_args()
    
    # Set environment variables from command line arguments if provided
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
    if args.model:
        os.environ['OPENAI_MODEL_NAME'] = args.model
    if args.api_base:
        os.environ['OPENAI_API_BASE'] = args.api_base
    
    # Set debug environment variable
    os.environ['DEBUG'] = str(args.debug).lower()
    
    # Run the server
    uvicorn.run(
        "research_api:app",
        host=args.host,
        port=args.port,
        reload=args.debug
    )

if __name__ == "__main__":
    main()
