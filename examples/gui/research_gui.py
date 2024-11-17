import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from tkinter.messagebox import showerror, showinfo
from pathlib import Path
import json
import logging
from datetime import datetime
import queue
import threading
import argparse

# Add the parent directory to sys.path to allow importing research_agent
sys.path.append(str(Path(__file__).resolve().parents[2]))

from research_agent import ResearchAgent
from dotenv import load_dotenv

class QueueHandler(logging.Handler):
    """Handler that puts logs into a queue for the GUI"""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

class ResearchGUI:
    def __init__(self, root, debug=False):
        self.root = root
        self.root.title("Research Paper Analyzer")
        self.root.geometry("1200x800")
        self.debug = debug
        
        # Create a queue for logging
        self.log_queue = queue.Queue()
        
        # Configure logging
        self.setup_logging()
        
        # Load environment variables
        load_dotenv()
        
        # Create and configure the main container
        self.create_gui()
        
        # Initialize the research agent
        self.initialize_agent()
        
        # Start the log queue checker
        self.check_log_queue()

    def setup_logging(self):
        """Configure logging for the application"""
        log_file = f'research_gui_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        # Create handlers
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setFormatter(console_formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            handlers=[file_handler, queue_handler]
        )
        
        self.logger = logging.getLogger(__name__)

    def create_gui(self):
        """Create the GUI elements"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File selection frame
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=80).grid(row=0, column=0, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Analyze Paper", command=self.analyze_paper).grid(row=0, column=2, padx=5)
        
        # Output area with notebook
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="5")
        output_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Create notebook for output
        self.notebook = ttk.Notebook(output_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create log tab
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="Log Output")
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=80, height=30)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        file_frame.columnconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

    def check_log_queue(self):
        """Check for new log records"""
        while True:
            try:
                record = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, record + '\n')
                self.log_text.see(tk.END)
                self.log_text.update_idletasks()
            except queue.Empty:
                break
        self.root.after(100, self.check_log_queue)

    def initialize_agent(self):
        """Initialize the ResearchAgent with environment variables"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            model_name = os.getenv('OPENAI_MODEL_NAME')
            api_base = os.getenv('OPENAI_API_BASE')
            
            if not all([api_key, model_name]):
                raise ValueError("Missing required environment variables")
            
            self.agent = ResearchAgent(
                api_key=api_key,
                model_name=model_name,
                api_base=api_base,
                debug=self.debug
            )
            self.logger.info("ResearchAgent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ResearchAgent: {str(e)}")
            showerror("Initialization Error", 
                     "Failed to initialize ResearchAgent. Please check your environment variables.")

    def browse_file(self):
        """Open file dialog to select a PDF file"""
        filename = filedialog.askopenfilename(
            title="Select a PDF file",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            self.file_path.set(filename)

    def create_result_tab(self, title, content):
        """Create a new tab in the notebook with scrolled text"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        
        text_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=80, height=30)
        text_area.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Pretty print if content is JSON
        if isinstance(content, (dict, list)):
            text_area.insert(tk.END, json.dumps(content, indent=2))
        else:
            text_area.insert(tk.END, content)
            
        text_area.config(state=tk.DISABLED)
        
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

    def create_summary_tab(self, analysis_result):
        """Create a human-readable summary of the paper analysis"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Summary")
        
        text_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=80, height=30)
        text_area.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Build the summary
        summary = []
        summary.append("Research Paper Analysis Summary")
        summary.append("=" * 40 + "\n")
        
        # Objectives and Methods
        if 'objectives' in analysis_result:
            summary.append("Research Objectives:")
            summary.append("-" * 20)
            summary.append(f"{analysis_result['objectives']}\n")
        
        if 'methodology' in analysis_result:
            summary.append("Methodology:")
            summary.append("-" * 20)
            summary.append(f"{analysis_result['methodology']}\n")
        
        # Key Findings and Conclusions
        if 'key_findings' in analysis_result:
            summary.append("Key Findings:")
            summary.append("-" * 20)
            summary.append(f"{analysis_result['key_findings']}\n")
        
        if 'conclusions' in analysis_result:
            summary.append("Conclusions:")
            summary.append("-" * 20)
            summary.append(f"{analysis_result['conclusions']}\n")
        
        # Validity Assessment
        if 'validity_assessment' in analysis_result:
            va = analysis_result['validity_assessment']
            if isinstance(va, dict):
                summary.append("Research Validity Assessment:")
                summary.append("-" * 20)
                if 'confidence_score' in va:
                    summary.append(f"Confidence Score: {va['confidence_score']:.2f}/1.00\n")
                if 'methodology_analysis' in va:
                    summary.append(f"Methodology Analysis: {va['methodology_analysis']}\n")
                if 'conclusion_validity' in va:
                    summary.append(f"Conclusion Validity: {va['conclusion_validity']}\n")
        
        # Credibility Assessment
        if 'credibility_assessment' in analysis_result:
            ca = analysis_result['credibility_assessment']
            if isinstance(ca, dict):
                summary.append("Research Credibility Assessment:")
                summary.append("-" * 20)
                if 'credibility_score' in ca:
                    summary.append(f"Credibility Score: {ca['credibility_score']:.2f}/1.00\n")
                if 'journal_reputation' in ca:
                    summary.append(f"Journal Reputation: {ca['journal_reputation']}\n")
                if 'peer_review_status' in ca:
                    summary.append(f"Peer Review Status: {ca['peer_review_status']}\n")
        
        # Limitations and Counter Arguments
        if 'limitations' in analysis_result:
            summary.append("Limitations:")
            summary.append("-" * 20)
            summary.append(f"{analysis_result['limitations']}\n")
        
        if 'counter_arguments' in analysis_result:
            summary.append("Counter Arguments:")
            summary.append("-" * 20)
            summary.append(f"{analysis_result['counter_arguments']}\n")
        
        # Join all parts with proper spacing
        summary_text = '\n'.join(summary)
        
        text_area.insert(tk.END, summary_text)
        text_area.config(state=tk.DISABLED)
        
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        
        # Switch to the summary tab
        self.notebook.select(frame)

    def clear_results(self):
        """Clear all result tabs except the log tab"""
        for tab in self.notebook.tabs()[1:]:  # Skip the first (log) tab
            self.notebook.forget(tab)

    def analyze_paper(self):
        """Analyze the selected paper and display results"""
        pdf_path = self.file_path.get()
        
        if not pdf_path:
            showerror("Error", "Please select a PDF file first")
            return
            
        if not os.path.exists(pdf_path):
            showerror("Error", f"File not found: {pdf_path}")
            return
            
        # Disable the analyze button during processing
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.configure(state='disabled')
        
        # Show analysis in progress
        self.root.config(cursor="wait")
        self.clear_results()
        self.logger.info("Starting paper analysis...")
        
        # Create and start analysis thread
        analysis_thread = threading.Thread(
            target=self._run_analysis,
            args=(pdf_path,),
            daemon=True
        )
        analysis_thread.start()
        
        # Start progress checking
        self.root.after(100, lambda: self._check_analysis_complete(analysis_thread))
    
    def _run_analysis(self, pdf_path):
        """Run the analysis in a separate thread"""
        try:
            self.analysis_result = self.agent.analyze_paper(pdf_path)
            self.analysis_error = None
        except Exception as e:
            self.logger.error(f"Error analyzing paper: {str(e)}")
            self.analysis_error = str(e)
            self.analysis_result = None
    
    def _check_analysis_complete(self, analysis_thread):
        """Check if analysis is complete and update GUI accordingly"""
        if analysis_thread.is_alive():
            # Analysis still running, check again in 100ms
            self.root.after(100, lambda: self._check_analysis_complete(analysis_thread))
            return
        
        # Re-enable buttons
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.configure(state='normal')
        
        # Reset cursor
        self.root.config(cursor="")
        
        if self.analysis_error:
            showerror("Analysis Error", f"Failed to analyze paper: {self.analysis_error}")
            return
        
        try:
            # First create all the raw data tabs
            for section, content in self.analysis_result.items():
                title = section.replace('_', ' ').title()
                self.create_result_tab(title, content)
            
            # Create and switch to the summary tab
            self.create_summary_tab(self.analysis_result)
            
            self.logger.info("Analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error displaying results: {str(e)}")
            showerror("Display Error", f"Failed to display results: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Research Paper Analyzer GUI')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    root = tk.Tk()
    app = ResearchGUI(root, debug=args.debug)
    root.mainloop()

if __name__ == "__main__":
    main()
