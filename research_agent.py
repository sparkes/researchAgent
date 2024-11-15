import os
from typing import Dict, List, Optional, Union
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import PyPDF2
import requests
from scholarly import scholarly
import habanero
import nltk
from bs4 import BeautifulSoup
import pdfplumber
import pandas as pd
from dotenv import load_dotenv
import logging
import logging.handlers
import sys
import csv
from pathlib import Path

# Load environment variables
load_dotenv()

def setup_logging(debug_mode: bool = False):
    """Configure logging based on debug mode."""
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.handlers.RotatingFileHandler('research_agent.log', maxBytes=1000000, backupCount=5)
        ]
    )
    
    # Set all pdfminer loggers to ERROR level
    for logger_name in ['pdfminer', 'pdfminer.psparser', 'pdfminer.pdfparser', 'pdfminer.pdfdocument', 
                       'pdfminer.pdfpage', 'pdfminer.pdfinterp', 'pdfminer.converter', 'pdfminer.cmapdb', 
                       'pdfminer.layout']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # Set other verbose loggers to WARNING
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)

# Create logger for this module
logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, model_name: Optional[str] = None, debug: bool = False):
        """Initialize the Research Agent with necessary APIs and models."""
        setup_logging(debug)
        logger.info("Initializing ResearchAgent")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        
        # Initialize usage tracking
        self.usage_stats = {
            'api_calls': 0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0,
            'has_token_tracking': False,
            'estimated_cost_usd': 0.0
        }
        
        # Load token rates from CSV
        self.token_rates = None
        try:
            rates_file = Path(__file__).parent / 'token_rates.csv'
            rates_df = pd.read_csv(rates_file)
            
            # Try to find the model in the rates
            model_rates = rates_df[rates_df['model'] == self.model_name]
            if not model_rates.empty:
                self.token_rates = {
                    'prompt': float(model_rates.iloc[0]['prompt']),
                    'completion': float(model_rates.iloc[0]['completion'])
                }
                logger.debug(f"Loaded token rates for model {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load token rates: {str(e)}")
        
        if not self.api_key:
            logger.error("API key not found")
            raise ValueError("API key is required")
            
        if self.api_base:
            logger.debug(f"Using API base: {self.api_base}")
        
        self.llm = ChatOpenAI(
            temperature=0.3,
            openai_api_key=self.api_key,
            model_name=self.model_name,
            openai_api_base=self.api_base if self.api_base else None
        )
        
        logger.info("Initializing Crossref client")
        self.crossref_client = habanero.Crossref()
        
        logger.debug("Downloading NLTK data")
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        logger.info("ResearchAgent initialization complete")

    def _track_usage(self, response) -> None:
        """Track API usage from response."""
        self.usage_stats['api_calls'] += 1
        
        # Try to get token usage from response
        try:
            usage = response.llm_output.get('token_usage', {})
            if usage:
                self.usage_stats['has_token_tracking'] = True
                self.usage_stats['total_prompt_tokens'] += usage.get('prompt_tokens', 0)
                self.usage_stats['total_completion_tokens'] += usage.get('completion_tokens', 0)
                self.usage_stats['total_tokens'] += usage.get('total_tokens', 0)
                
                # Only calculate cost if we have both token rates and token counts
                if self.token_rates and self.usage_stats['has_token_tracking']:
                    prompt_cost = (usage.get('prompt_tokens', 0) / 1000) * self.token_rates['prompt']
                    completion_cost = (usage.get('completion_tokens', 0) / 1000) * self.token_rates['completion']
                    self.usage_stats['estimated_cost_usd'] += prompt_cost + completion_cost
            
        except (AttributeError, KeyError) as e:
            logger.debug(f"Could not track token usage: {str(e)}")

    def _get_usage_report(self) -> dict:
        """Generate a usage report."""
        usage_report = {
            'api_usage': {
                'total_api_calls': self.usage_stats['api_calls']
            }
        }
        
        # Only include token counts if we have them
        if self.usage_stats['has_token_tracking']:
            usage_report['api_usage'].update({
                'total_tokens': self.usage_stats['total_tokens'],
                'prompt_tokens': self.usage_stats['total_prompt_tokens'],
                'completion_tokens': self.usage_stats['total_completion_tokens']
            })
            
            # Only include cost if we have both token rates and token counts
            if self.token_rates:
                usage_report['api_usage']['estimated_cost_usd'] = round(self.usage_stats['estimated_cost_usd'], 4)
        
        return usage_report

    def process_input(self, input_source: str) -> Dict:
        """Process input source (PDF or URL) and extract paper content."""
        logger.info(f"Processing input source: {input_source}")
        try:
            if input_source.startswith(('http://', 'https://')):
                logger.debug("Detected URL input")
                return self._process_url(input_source)
            else:
                logger.debug("Detected file input")
                # Convert to absolute path if relative
                if not os.path.isabs(input_source):
                    input_source = os.path.abspath(input_source)
                    logger.debug(f"Converted to absolute path: {input_source}")
                return self._process_pdf(input_source)
        except Exception as e:
            logger.error(f"Error processing input {input_source}: {str(e)}", exc_info=True)
            raise Exception(f"Error processing input {input_source}: {str(e)}")

    def _process_pdf(self, pdf_path: str) -> Dict:
        """Extract text and metadata from PDF file."""
        logger.info(f"Processing PDF: {pdf_path}")
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            text = ""
            metadata = {}
            
            logger.debug("Attempting to extract text with pdfplumber")
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        logger.debug(f"Processing page {i+1}")
                        extracted_text = page.extract_text()
                        if extracted_text:
                            text += extracted_text + "\n"
                logger.info(f"Successfully extracted {len(text)} characters with pdfplumber")
            except Exception as e:
                logger.warning(f"pdfplumber failed, trying PyPDF2: {str(e)}")
                
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    metadata = reader.metadata
                    logger.debug(f"Extracted metadata: {metadata}")
                    
                    # Extract text from each page
                    for i, page in enumerate(reader.pages):
                        logger.debug(f"Processing page {i+1} with PyPDF2")
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                logger.info(f"Successfully extracted {len(text)} characters with PyPDF2")

            if not metadata:
                logger.debug("Attempting to extract metadata with PyPDF2")
                try:
                    with open(pdf_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        metadata = reader.metadata
                        logger.debug(f"Successfully extracted metadata: {metadata}")
                except Exception as e:
                    logger.error(f"Failed to extract metadata: {str(e)}")
                    metadata = {}

            if not text.strip():
                logger.error("No text could be extracted from the PDF")
                raise ValueError("No text could be extracted from the PDF")

            result = {
                'text': text,
                'metadata': metadata,
                'source_type': 'pdf',
                'path': pdf_path
            }
            logger.info("Successfully processed PDF")
            logger.debug(f"Result metadata: {metadata}")
            return result
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}", exc_info=True)
            raise Exception(f"Error processing PDF {pdf_path}: {str(e)}")

    def _process_url(self, url: str) -> Dict:
        """Extract paper content from URL."""
        logger.info(f"Processing URL: {url}")
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            text = soup.get_text()
            
            logger.info(f"Successfully extracted {len(text)} characters from URL")
            return {
                'text': text,
                'metadata': {
                    'url': url,
                },
                'source_type': 'url'
            }
        except Exception as e:
            logger.error(f"Error processing URL: {str(e)}", exc_info=True)
            raise Exception(f"Error processing URL: {str(e)}")

    def assess_validity(self, paper_data: Dict) -> Dict:
        """Assess the validity of the research paper."""
        logger.info("Assessing validity")
        validity_prompt = PromptTemplate.from_template("""
            Analyze the following research paper for validity. Focus on:
            1. Research methodology
            2. Sample size and selection
            3. Statistical methods
            4. Potential biases
            5. Data handling and processing
            6. Conclusion validity
            
            Paper text: {paper_text}
            
            Provide a structured analysis addressing each point.
            """)
        
        # Create a runnable chain
        chain = validity_prompt | self.llm | StrOutputParser()
        
        # Invoke the chain
        response = chain.invoke({"paper_text": paper_data['text'][:4000]})  # Limit text length
        
        # Track API usage
        self._track_usage(response)
        
        logger.info("Validity assessment complete")
        return {
            'validity_assessment': response,
            'confidence_score': self._calculate_confidence_score(response)
        }
        
    def check_credibility(self, paper_data: Dict) -> Dict:
        """Check the credibility of the paper and authors."""
        logger.info("Checking credibility")
        # Query Google Scholar
        search_query = scholarly.search_pubs(paper_data['metadata'].get('title', ''))
        publication = next(search_query, None)
        
        # Query CrossRef
        try:
            if 'doi' in paper_data['metadata']:
                crossref_data = self.crossref_client.works(ids=paper_data['metadata']['doi'])
            else:
                crossref_data = None
        except:
            crossref_data = None
            
        logger.info("Credibility check complete")
        return {
            'publication_info': publication,
            'crossref_data': crossref_data,
            'citation_count': publication.citedby if publication else None
        }
        
    def check_retractions(self, paper_data: Dict) -> Dict:
        """Check for retractions and corrections."""
        logger.info("Checking retractions")
        retraction_prompt = PromptTemplate.from_template("""
            Based on the following paper information, identify any signs of retraction or major corrections:
            
            {paper_info}
            
            Provide a detailed analysis of the paper's current status and any concerns.
            """)
        
        # Create a runnable chain
        chain = retraction_prompt | self.llm | StrOutputParser()
        
        # Invoke the chain
        response = chain.invoke({"paper_info": str(paper_data['metadata'])})
        
        # Track API usage
        self._track_usage(response)
        
        logger.info("Retraction check complete")
        return {
            'retraction_status': response
        }
        
    def find_counter_arguments(self, paper_data: Dict) -> Dict:
        """Search for counter-arguments and contradicting papers."""
        logger.info("Finding counter-arguments")
        counter_prompt = PromptTemplate.from_template("""
            Analyze the following paper and identify potential counter-arguments and limitations:
            
            {paper_text}
            
            Provide a detailed analysis of:
            1. Potential weaknesses in methodology
            2. Alternative interpretations of results
            3. Conflicting evidence from other research
            """)
        
        # Create a runnable chain
        chain = counter_prompt | self.llm | StrOutputParser()
        
        # Invoke the chain
        response = chain.invoke({"paper_text": paper_data['text'][:4000]})
        
        # Track API usage
        self._track_usage(response)
        
        logger.info("Counter-arguments found")
        return {
            'counter_arguments': response
        }
        
    def create_summary(self, paper_data: Dict) -> Dict:
        """Create a comprehensive summary of the paper."""
        logger.info("Creating summary")
        summary_prompt = PromptTemplate.from_template("""
            Create a comprehensive summary of the following research paper:
            
            {paper_text}
            
            Include:
            1. Main objectives
            2. Methodology
            3. Key findings
            4. Conclusions
            5. Limitations
            """)
        
        # Create a runnable chain
        chain = summary_prompt | self.llm | StrOutputParser()
        
        # Invoke the chain
        response = chain.invoke({"paper_text": paper_data['text'][:4000]})
        
        # Track API usage
        self._track_usage(response)
        
        logger.info("Summary created")
        return {
            'summary': response
        }
        
    def generate_report(self, paper_data: Dict, analyses: Dict) -> Dict:
        """Generate a final analysis report."""
        logger.info("Generating report")
        report_prompt = PromptTemplate.from_template("""
            Generate a comprehensive research paper analysis report based on the following analyses:
            
            {analyses}
            
            Format the report with the following sections:
            1. Basic Information
            2. Validity Assessment
            3. Credibility Analysis
            4. Retraction Status
            5. Scientific Context
            6. Paper Summary
            7. Agent's Conclusion
            8. Sources
            """)
        
        # Create a runnable chain
        chain = report_prompt | self.llm | StrOutputParser()
        
        # Invoke the chain
        response = chain.invoke({"analyses": str(analyses)})
        
        # Track API usage
        self._track_usage(response)
        
        logger.info("Report generated")
        return {
            'report': response,
            'metadata': paper_data['metadata'],
            'analyses': analyses
        }
        
    def _calculate_confidence_score(self, analysis: str) -> float:
        """Calculate a confidence score for the analysis."""
        logger.info("Calculating confidence score")
        # Implement confidence scoring logic
        # This is a placeholder implementation
        return 0.8
        
    def analyze_paper(self, input_source: str) -> Dict:
        """Main method to analyze a research paper."""
        logger.info(f"Starting full paper analysis for: {input_source}")
        try:
            logger.debug("Step 1: Processing input")
            paper_data = self.process_input(input_source)
            
            logger.debug("Step 2: Assessing validity")
            validity = self.assess_validity(paper_data)
            
            logger.debug("Step 3: Checking credibility")
            credibility = self.check_credibility(paper_data)
            
            logger.debug("Step 4: Checking retractions")
            retractions = self.check_retractions(paper_data)
            
            logger.debug("Step 5: Finding counter arguments")
            counter_arguments = self.find_counter_arguments(paper_data)
            
            logger.debug("Step 6: Creating summary")
            summary = self.create_summary(paper_data)
            
            # Combine all analyses
            analyses = {
                'validity': validity,
                'credibility': credibility,
                'retractions': retractions,
                'counter_arguments': counter_arguments,
                'summary': summary
            }
            
            logger.debug("Step 7: Generating final report")
            report = self.generate_report(paper_data, analyses)
            
            # Add usage report
            report.update(self._get_usage_report())
            
            logger.info("Paper analysis completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error during paper analysis: {str(e)}", exc_info=True)
            raise Exception(f"Error analyzing paper: {str(e)}")
