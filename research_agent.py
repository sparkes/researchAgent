import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Optional, List, Any
from pathlib import Path
from datetime import datetime
import nltk
import pandas as pd
import habanero
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, validator
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from scholarly import scholarly
import pdfplumber
import PyPDF2
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import logging.handlers
import sys
import csv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

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

# Define Pydantic models for structured outputs
class ValidityAssessment(BaseModel):
    methodology_analysis: str = Field(description="Analysis of research methodology")
    sample_analysis: str = Field(description="Analysis of sample size and selection")
    statistical_analysis: str = Field(description="Analysis of statistical methods")
    bias_analysis: str = Field(description="Analysis of potential biases")
    data_handling_analysis: str = Field(description="Analysis of data handling and processing")
    conclusion_validity: str = Field(description="Analysis of conclusion validity")
    confidence_score: float = Field(description="Confidence score between 0 and 1")
    
    class Config:
        arbitrary_types_allowed = True

class PaperSummary(BaseModel):
    objectives: str = Field(description="Main objectives of the paper")
    methodology: str = Field(description="Research methodology used")
    key_findings: str = Field(description="Key findings and results")
    conclusions: str = Field(description="Main conclusions")
    limitations: str = Field(description="Study limitations")
    
    class Config:
        arbitrary_types_allowed = True

class CredibilityAssessment(BaseModel):
    """Model for research paper credibility assessment."""
    author_credentials: str = Field(..., description="Analysis of authors' credentials and affiliations")
    journal_reputation: str = Field(..., description="Analysis of the publishing journal's reputation")
    citation_analysis: str = Field(..., description="Analysis of citations and references")
    funding_transparency: str = Field(..., description="Analysis of funding sources and disclosures")
    peer_review_status: str = Field(..., description="Status of peer review")
    methodology_robustness: str = Field(..., description="Assessment of methodology robustness")
    credibility_score: float = Field(..., ge=0.0, le=1.0, description="Credibility score between 0 and 1")

    @validator('credibility_score')
    def validate_score(cls, v):
        """Ensure credibility score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Credibility score must be between 0 and 1')
        return v

class RetractionInfo(BaseModel):
    """Model for paper retraction information."""
    is_retracted: bool = Field(False, description="Whether the paper has been retracted")
    retraction_status: str = Field("No retraction found", description="Current status of any retraction")
    retraction_date: Optional[str] = Field(None, description="Date of retraction if applicable")
    retraction_reason: Optional[str] = Field(None, description="Reason for retraction if applicable")
    related_retractions: List[str] = Field(default_factory=list, description="Related retractions or integrity issues")
    verification_sources: List[str] = Field(default_factory=list, description="Sources used to verify retraction status")
    last_checked_date: Optional[str] = Field(None, description="Date when retraction status was last checked")
    confidence_score: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in the retraction assessment")

class ResearchAgent:
    """Agent for analyzing research papers using LLM."""

    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None,
                model_name: Optional[str] = None, debug: bool = False):
        """Initialize the research agent.
        
        Args:
            api_key: Optional OpenAI API key. If not provided, will look for OPENAI_API_KEY in environment
            api_base: Optional API base URL. If not provided, will look for OPENAI_API_BASE in environment
            model_name: Optional model name. If not provided, will look for OPENAI_MODEL_NAME in environment
            debug: Whether to enable debug logging
        """
        try:
            # Configure logging
            if debug:
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(logging.INFO)
            
            logger.info("Initializing ResearchAgent")
            
            # Validate API key
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not provided and not found in environment variables")
            logger.debug("API key validated")
            
            # Get model name from environment if not provided
            self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME")
            if not self.model_name:
                raise ValueError("Model name not provided and OPENAI_MODEL_NAME not found in environment variables")
            logger.debug(f"Using model: {self.model_name}")
            
            # Configure API base if provided
            self.api_base = api_base or os.getenv("OPENAI_API_BASE")
            if self.api_base:
                logger.debug(f"Using custom API base: {self.api_base}")
            
            # Initialize OpenAI client
            logger.debug(f"Initializing OpenAI client with model: {self.model_name}")
            try:
                client_args = {"api_key": self.api_key}
                if self.api_base:
                    client_args["base_url"] = self.api_base
                self.client = OpenAI(**client_args)
                logger.debug("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                raise
            
            # Initialize LLM
            try:
                logger.debug("Initializing ChatOpenAI")
                self.llm = ChatOpenAI(
                    model_name=self.model_name,
                    openai_api_key=self.api_key,
                    openai_api_base=self.api_base,
                    temperature=0,
                    request_timeout=60
                )
                # Test the LLM connection
                test_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a test assistant."),
                    ("user", "Respond with the word 'OK'")
                ])
                test_response = self._invoke_with_retry(test_prompt.format_messages())
                logger.debug(f"LLM test response received: {test_response.content}")
            except Exception as e:
                logger.error(f"Failed to initialize or test LLM: {str(e)}")
                raise
            
            logger.info("ResearchAgent initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize ResearchAgent: {str(e)}")
            raise

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

    def _handle_rate_limit(self, attempt: int, max_attempts: int = 5) -> None:
        """Handle rate limit with exponential backoff."""
        if attempt >= max_attempts:
            raise Exception("Max retry attempts reached for rate limit")
        
        wait_time = min(2 ** attempt, 32)  # Exponential backoff capped at 32 seconds
        logger.info(f"Rate limit hit. Waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_attempts})")
        time.sleep(wait_time)

    def _invoke_with_retry(self, messages: List[Dict], max_retries: int = 3, initial_delay: float = 1.0) -> Optional[Any]:
        """Invoke the LLM with retry logic.
        
        Args:
            messages: List of message dictionaries or Message objects to send to the LLM
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            
        Returns:
            LLM response or None if all retries fail
        """
        delay = initial_delay
        
        # Initialize usage tracking if not already done
        if not hasattr(self, "_total_tokens"):
            self._total_tokens = 0
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._total_cost = 0.0
            self._num_api_calls = 0
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1} of {max_retries}")
                
                # Convert messages to the format expected by OpenAI
                formatted_messages = []
                for msg in messages:
                    if hasattr(msg, 'type') and hasattr(msg, 'content'):  # Message object
                        formatted_messages.append({
                            "role": msg.type if msg.type != "human" else "user",
                            "content": msg.content
                        })
                    elif isinstance(msg, dict):  # Already formatted dict
                        formatted_messages.append(msg)
                    else:
                        logger.warning(f"Unexpected message format: {type(msg)}")
                        continue
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=formatted_messages
                )
                
                # Track usage statistics
                self._num_api_calls += 1
                if hasattr(response, "usage"):
                    usage = response.usage
                    self._total_tokens += usage.total_tokens
                    self._total_prompt_tokens += usage.prompt_tokens
                    self._total_completion_tokens += usage.completion_tokens
                    
                    # Estimate cost based on model and tokens
                    # These rates are approximate and may need adjustment
                    if "gpt-4" in self.model_name.lower():
                        prompt_cost = 0.03 * (usage.prompt_tokens / 1000)
                        completion_cost = 0.06 * (usage.completion_tokens / 1000)
                    else:  # Assume GPT-3.5 rates
                        prompt_cost = 0.0015 * (usage.prompt_tokens / 1000)
                        completion_cost = 0.002 * (usage.completion_tokens / 1000)
                    self._total_cost += prompt_cost + completion_cost
                
                return response.choices[0].message
                
            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error("All retry attempts failed")
                    return None

    def _parse_json_response(self, response_text: str, max_retries: int = 3) -> Dict:
        """Try to extract JSON from the response text, with multiple attempts."""
        import json
        import re

        logger.debug(f"Attempting to parse response text: {response_text}")

        # First try: direct JSON parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed: {str(e)}")

        # Second try: find JSON-like structure
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                json_str = match.group()
                logger.debug(f"Found JSON-like structure: {json_str}")
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.debug(f"JSON-like structure parsing failed: {str(e)}")

        # Third try: clean up common formatting issues
        try:
            # Remove markdown code block syntax
            cleaned = re.sub(r'```json\s*|\s*```', '', response_text)
            # Remove any text before the first {
            cleaned = re.sub(r'^[^{]*', '', cleaned)
            # Remove any text after the last }
            cleaned = re.sub(r'}[^}]*$', '}', cleaned)
            logger.debug(f"Cleaned text: {cleaned}")
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.debug(f"Cleaned text parsing failed: {str(e)}")
            raise ValueError(f"Could not parse JSON from response: {response_text}")

    def _adaptive_chunk_text(self, text: str, initial_max_size: int = 12000, min_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Adaptively split text into chunks, trying larger chunks first and splitting if needed."""
        def split_chunk(text: str, num_parts: int) -> List[str]:
            """Split text into approximately equal parts at natural break points."""
            avg_size = len(text) // num_parts
            chunks = []
            start = 0
            
            for i in range(num_parts - 1):
                # Look for a natural break point around the target position
                target_pos = start + avg_size
                # Search for natural break points (period or newline) within a window
                window_size = min(200, avg_size // 2)
                search_start = max(0, target_pos - window_size)
                search_end = min(len(text), target_pos + window_size)
                
                # Find the last period or newline in the search window
                period_pos = text.rfind('.', search_start, search_end)
                newline_pos = text.rfind('\n', search_start, search_end)
                break_pos = max(period_pos, newline_pos)
                
                if break_pos == -1 or break_pos <= start:
                    # If no natural break found, just split at target position
                    break_pos = target_pos
                
                chunks.append(text[start:break_pos + 1])
                start = break_pos + 1
            
            # Add the last chunk
            chunks.append(text[start:])
            return chunks

        def attempt_chunk_processing(chunk: str, messages_func) -> Optional[Dict]:
            """Try to process a chunk, return None if it fails due to size."""
            try:
                messages = messages_func(chunk)
                response = self._invoke_with_retry(messages)
                if response is None:
                    return None
                return self._parse_json_response(response.content)
            except Exception as e:
                error_str = str(e).lower()
                # Check for common size-related error messages
                size_error_indicators = [
                    "too long", "maximum context", "token limit",
                    "context window", "input too large", "content too long"
                ]
                if any(indicator in error_str for indicator in size_error_indicators):
                    logger.info(f"Chunk size {len(chunk)} characters was too large, will split further")
                    return None
                raise  # Re-raise if it's not a size-related error

        def process_chunks(chunks: List[str], messages_func) -> List[Dict]:
            """Process a list of chunks, splitting further if needed."""
            results = []
            for i, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {i+1} of {len(chunks)}")
                
                if len(chunk) < min_chunk_size:
                    logger.warning(f"Chunk {i+1} is smaller than minimum size ({len(chunk)} < {min_chunk_size})")
                    continue
                
                result = attempt_chunk_processing(chunk, messages_func)
                if result is None and len(chunk) > min_chunk_size * 2:
                    # Split this chunk in half and process recursively
                    logger.info(f"Splitting chunk {i+1} of size {len(chunk)} into smaller pieces")
                    subchunks = split_chunk(chunk, 2)
                    subresults = process_chunks(subchunks, messages_func)
                    results.extend(subresults)
                elif result is not None:
                    results.append(result)
                else:
                    logger.warning(f"Chunk {i+1} could not be processed and is too small to split further")
            
            return results

        # First try processing the entire text
        logger.info(f"Attempting to process entire text of size {len(text)}")
        if len(text) <= initial_max_size:
            return [text]  # Return as single chunk if under initial max size

        # If that fails or text is too large, split into quarters initially
        logger.info("Splitting text into quarters for initial attempt")
        initial_chunks = split_chunk(text, 4)
        return initial_chunks

    def assess_validity(self, paper_data: Dict) -> Dict:
        """Assess the validity of the research paper using structured output."""
        logger.info("Assessing validity")
        
        # Define the expected JSON structure
        example_json = {
            "methodology_analysis": "Analysis of research methodology",
            "sample_analysis": "Analysis of sample size and selection",
            "statistical_analysis": "Analysis of statistical methods",
            "bias_analysis": "Analysis of potential biases",
            "data_handling_analysis": "Analysis of data handling and processing",
            "conclusion_validity": "Analysis of conclusion validity",
            "confidence_score": 0.0
        }

        def create_messages_for_chunk(chunk: str):
            return [
                SystemMessage(
                    content="You are a research paper analyzer. Analyze the given text and output ONLY a valid JSON object matching the provided structure. No other text."
                ),
                HumanMessage(
                    content=f"""Analyze this section of a research paper.
Output a JSON object with this exact structure:

{json.dumps(example_json, indent=2)}

Text to analyze:
{chunk}"""
                )
            ]

        # Get text chunks using adaptive chunking
        text = paper_data.get('text', '')
        if not text:
            raise ValueError("No text content found in paper data")

        chunks = self._adaptive_chunk_text(text)
        chunk_results = []

        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1} of {len(chunks)}")
            messages = create_messages_for_chunk(chunk)
            
            try:
                response = self._invoke_with_retry(messages)
                if response is None:
                    raise ValueError("No response received from LLM")
                
                json_data = self._parse_json_response(response.content)
                parsed_response = ValidityAssessment.parse_obj(json_data)
                chunk_results.append(parsed_response.dict())
                
            except Exception as e:
                logger.warning(f"Error processing chunk {i+1}: {str(e)}")
                continue

        if not chunk_results:
            raise Exception("Failed to process any chunks successfully")

        # Combine results from all chunks
        combined_result = {
            "methodology_analysis": "",
            "sample_analysis": "",
            "statistical_analysis": "",
            "bias_analysis": "",
            "data_handling_analysis": "",
            "conclusion_validity": "",
            "confidence_score": 0.0
        }

        # Combine text fields and average confidence score
        for result in chunk_results:
            for key in combined_result:
                if key == "confidence_score":
                    combined_result[key] += result[key] / len(chunk_results)
                else:
                    if combined_result[key]:
                        combined_result[key] += " "
                    combined_result[key] += result[key]

        logger.info("Validity assessment complete")
        return combined_result

    def create_summary(self, paper_data: Dict) -> Dict:
        """Create a structured summary of the paper."""
        logger.info("Creating summary")
        max_retries = 3
        last_response = None
        last_error = None
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research paper summarizer. Output a JSON object with these fields: objectives (main objectives), methodology (research methods), key_findings (results), conclusions (main conclusions), and limitations (study limitations). Output ONLY the JSON object without any additional text or formatting."),
            ("user", "Summarize this paper as a JSON object with the specified fields. Paper text: {paper_text}")
        ])
        
        # Format the prompt
        formatted_prompt = prompt.format_messages(
            paper_text=paper_data.get('text', '')[:4000]  # Safely get text with fallback
        )
        
        # Get response from LLM
        logger.debug("Calling LLM for response")
        last_response = self._invoke_with_retry(formatted_prompt)
        logger.debug(f"Raw LLM response: {last_response.content}")
        
        # Clean and parse the JSON response
        try:
            # Try to extract JSON from the response
            response_text = last_response.content.strip()
            logger.debug(f"Initial text: {response_text}")
            
            # Remove any text before the first {
            if '{' in response_text:
                response_text = response_text[response_text.find('{'):]
            
            # Remove any text after the last }
            if '}' in response_text:
                response_text = response_text[:response_text.rfind('}')+1]
            
            # Remove any markdown code block markers
            response_text = response_text.replace('```json', '').replace('```', '')
            
            # Clean up any remaining whitespace
            response_text = response_text.strip()
            logger.debug(f"Cleaned text: {response_text}")
            
            if not response_text.startswith('{') or not response_text.endswith('}'):
                raise json.JSONDecodeError("Invalid JSON structure", response_text, 0)
            
            json_data = json.loads(response_text)
            logger.debug(f"Successfully parsed JSON: {json_data}")
            
            # Ensure all required fields are present
            required_fields = ["objectives", "methodology", "key_findings", "conclusions", "limitations"]
            for field in required_fields:
                if field not in json_data:
                    json_data[field] = "Not specified in the paper"
            
            # Parse into Pydantic model
            parsed_response = PaperSummary.parse_obj(json_data)
            logger.debug("Successfully parsed into PaperSummary model")
            
            logger.info("Summary created")
            return parsed_response.dict()
            
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parsing failed: {str(e)}")
            raise Exception(f"Could not parse JSON from response: {response_text}")
        except Exception as e:
            raise Exception(f"Error processing response: {str(e)}")

    def check_credibility(self, paper_data: Dict) -> Dict:
        """Check the credibility of the research paper using structured output."""
        logger.info("Checking credibility")
        max_retries = 3
        last_response = None
        last_error = None

        # Define the expected JSON structure
        example_json = {
            "author_credentials": "Analysis of authors' credentials and affiliations",
            "journal_reputation": "Analysis of the publishing journal's reputation",
            "citation_analysis": "Analysis of citations and references",
            "funding_transparency": "Analysis of funding sources and disclosures",
            "peer_review_status": "Status of peer review",
            "methodology_robustness": "Assessment of methodology robustness",
            "credibility_score": 0.0
        }

        try:
            # Create messages with proper types
            messages = [
                SystemMessage(
                    content="You are a research credibility analyzer. Analyze the given text and output ONLY a valid JSON object matching the provided structure. No other text."
                ),
                HumanMessage(
                    content=f"""Analyze this research paper's credibility.
Output a JSON object with this exact structure:

{json.dumps(example_json, indent=2)}

Text to analyze:
{paper_data.get('text', '')}"""
                )
            ]

            # Get response from LLM
            logger.debug("Calling LLM for credibility check")
            last_response = self._invoke_with_retry(messages)
            if last_response is None:
                raise ValueError("No response received from LLM")
            logger.debug(f"Raw LLM response: {last_response.content}")
            
            # Try to parse the JSON response
            json_data = self._parse_json_response(last_response.content)
            logger.debug(f"Successfully parsed JSON: {json_data}")

            # Parse into Pydantic model
            parsed_response = CredibilityAssessment.parse_obj(json_data)
            logger.debug("Successfully parsed into CredibilityAssessment model")

            logger.info("Credibility check complete")
            return parsed_response.dict()

        except Exception as e:
            error_msg = f"Failed to check credibility.\n"
            error_msg += f"Error: {str(e)}\n"
            if last_response:
                error_msg += f"Last response content: {last_response.content}"
            raise Exception(error_msg)

    def check_retractions(self, paper_data: Dict) -> Dict:
        """Check if the paper has been retracted or has related retractions."""
        logger.info("Checking for retractions")

        # Define the expected JSON structure
        example_json = {
            "is_retracted": False,
            "retraction_status": "No retraction found",
            "retraction_date": None,
            "retraction_reason": None,
            "related_retractions": [],
            "verification_sources": ["CrossRef", "Publisher Database"],
            "last_checked_date": None,
            "confidence_score": 1.0
        }

        try:
            # Extract DOI if available
            doi = paper_data.get('doi')
            title = paper_data.get('title', '')
            authors = paper_data.get('authors', [])

            retraction_info = example_json.copy()
            retraction_info['last_checked_date'] = datetime.now().isoformat()

            # Check CrossRef if DOI is available
            if doi:
                try:
                    cr = habanero.Crossref()
                    work = cr.works(ids=doi)
                    if work:
                        # Check if paper is marked as retracted
                        if 'is-retracted' in work[0]:
                            retraction_info['is_retracted'] = work[0]['is-retracted']
                            retraction_info['retraction_status'] = "Confirmed retraction"
                            if 'retraction-date' in work[0]:
                                retraction_info['retraction_date'] = work[0]['retraction-date']
                except Exception as e:
                    logger.warning(f"Error checking CrossRef: {str(e)}")

            # Use LLM to analyze the paper content for retraction indicators
            messages = [
                SystemMessage(
                    content="You are a research paper analyzer specializing in identifying retractions and integrity issues. Analyze the given text and output ONLY a valid JSON object matching the provided structure. No other text."
                ),
                HumanMessage(
                    content=f"""Analyze this research paper for any indicators of retraction or integrity issues.
Output a JSON object with this exact structure:

{json.dumps(example_json, indent=2)}

Paper Title: {title}
Authors: {', '.join(authors) if authors else 'Not provided'}
DOI: {doi if doi else 'Not provided'}

Text to analyze:
{paper_data.get('text', '')}"""
                )
            ]

            # Get LLM analysis
            response = self._invoke_with_retry(messages)
            if response:
                llm_analysis = self._parse_json_response(response.content)
                
                # Update retraction info with LLM findings if more severe
                if llm_analysis.get('is_retracted', False) and not retraction_info['is_retracted']:
                    retraction_info.update(llm_analysis)
                    retraction_info['verification_sources'].append("AI Analysis")
                    # Reduce confidence score since it's based on AI analysis only
                    retraction_info['confidence_score'] = 0.7

            logger.info("Retraction check complete")
            return retraction_info

        except Exception as e:
            error_msg = f"Failed to check retractions.\n"
            error_msg += f"Error: {str(e)}"
            raise Exception(error_msg)

    def find_counter_arguments(self, paper_data: Dict) -> Dict:
        """Find counter arguments to the research paper."""
        logger.info("Finding counter arguments")
        max_retries = 3
        last_response = None
        last_error = None

        try:
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a research paper analyzer. Your task is to find counter arguments to research papers. Output a valid JSON object with counter arguments."),
                ("user", "Find counter arguments to this paper and output a JSON object with counter arguments as a list. Paper text: {paper_text}")
            ])

            # Format the prompt
            formatted_prompt = prompt.format_messages(
                paper_text=paper_data.get('text', '')[:4000]  # Safely get text with fallback
            )

            # Get response from LLM
            logger.debug("Calling LLM for counter arguments")
            last_response = self._invoke_with_retry(formatted_prompt)
            logger.debug(f"Raw LLM response: {last_response.content}")

            # Try to parse the JSON response
            json_data = self._parse_json_response(last_response.content)
            logger.debug(f"Successfully parsed JSON: {json_data}")

            # Parse into Pydantic model
            parsed_response = {"counter_arguments": json_data.get("counter_arguments", [])}
            logger.debug("Successfully parsed into counter arguments")

            logger.info("Counter arguments found")
            return parsed_response

        except Exception as e:
            error_msg = f"Failed to find counter arguments.\n"
            error_msg += f"Error: {str(e)}\n"
            if last_response:
                error_msg += f"Last response content: {last_response.content}"
            raise Exception(error_msg)

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

    def analyze_paper(self, input_source: str) -> Dict:
        """Main method to analyze a research paper with structured outputs."""
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

    def generate_report(self, paper_data: Dict, analyses: Dict) -> Dict:
        """Generate a comprehensive final report combining all analyses.
        
        Args:
            paper_data: Dictionary containing paper text and metadata
            analyses: Dictionary containing results of all analyses
            
        Returns:
            Dictionary containing the complete analysis report
        """
        logger.info("Generating final report")
        
        try:
            # Compile all analyses into a structured report
            report = {
                "metadata": {
                    "title": paper_data.get("title", "Unknown Title"),
                    "authors": paper_data.get("authors", []),
                    "date_analyzed": datetime.now().isoformat(),
                },
                "summary": analyses.get("summary", {}),
                "validity_assessment": analyses.get("validity", {}),
                "credibility_assessment": analyses.get("credibility", {}),
                "retraction_check": analyses.get("retractions", {}),
                "counter_arguments": analyses.get("counter_arguments", {"counter_arguments": []}),
                "analysis_status": "complete",
                "analysis_metadata": {
                    "model_used": self.model_name,
                    "api_base": self.api_base if self.api_base else "default",
                }
            }
            
            # Add error information if any analysis failed
            if any(not value for value in analyses.values()):
                report["analysis_status"] = "partial"
                report["failed_analyses"] = [
                    key for key, value in analyses.items() if not value
                ]
            
            logger.info("Report generation complete")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise Exception(f"Failed to generate report: {str(e)}")

    def _get_usage_report(self) -> Dict:
        """Generate a report of API usage statistics.
        
        Returns:
            Dictionary containing usage statistics
        """
        try:
            usage_stats = {
                "api_usage": {
                    "total_tokens": getattr(self, "_total_tokens", 0),
                    "total_prompt_tokens": getattr(self, "_total_prompt_tokens", 0),
                    "total_completion_tokens": getattr(self, "_total_completion_tokens", 0),
                    "total_cost": getattr(self, "_total_cost", 0.0),
                    "num_api_calls": getattr(self, "_num_api_calls", 0)
                }
            }
            return usage_stats
            
        except Exception as e:
            logger.warning(f"Error generating usage report: {str(e)}")
            return {
                "api_usage": {
                    "total_tokens": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_cost": 0.0,
                    "num_api_calls": 0
                }
            }