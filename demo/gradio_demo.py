"""
Gradio Demo Interface for PDF Page Ordering System
"""
import demo.gradio_demo as gr
import tempfile
import os
import time
import json
from typing import Dict, Any, Tuple, Optional
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our system components
try:
    from src.core.orchestrator import create_orchestrator
    from src.config.settings import get_settings
    from src.utils.helpers import format_duration, format_file_size
    from src.utils.validation import validate_pdf_file, validate_file_size
except ImportError as e:
    logger.error(f"Failed to import system components: {e}")
    logger.info("Make sure you're running from the project root directory")
    raise

class PDFOrderingDemo:
    """Main demo class for PDF ordering system"""
    
    def __init__(self):
        self.settings = get_settings()
        self.orchestrators = {}
        logger.info("PDF Ordering Demo initialized")
    
    def get_orchestrator(self, workflow_type: str):
        """Get or create orchestrator for workflow type"""
        if workflow_type not in self.orchestrators:
            self.orchestrators[workflow_type] = create_orchestrator(workflow_type)
        return self.orchestrators[workflow_type]
    
    def process_pdf(
        self, 
        pdf_file, 
        shuffle_enabled: bool = False,
        workflow_type: str = "standard",
        show_debug_info: bool = False
    ) -> Tuple[str, str, str, str]:
        """
        Process PDF file and return results
        
        Returns:
            - Main result (HTML)
            - Processing details (HTML) 
            - Debug info (JSON)
            - Error messages (if any)
        """
        if pdf_file is None:
            return "❌ No file uploaded", "", "", "Please upload a PDF file"
        
        start_time = time.time()
        temp_file_path = None
        
        try:
            # Validate file
            if not pdf_file.name.lower().endswith('.pdf'):
                return "❌ Invalid file type", "", "", "Please upload a PDF file"
            
            # Get file size
            file_size = os.path.getsize(pdf_file.name)
            
            # Validate file size
            size_validation = validate_file_size(file_size, self.settings.max_file_size_mb)
            if not size_validation["valid"]:
                return "❌ File too large", "", "", size_validation["error"]
            
            # Create temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                with open(pdf_file.name, 'rb') as src:
                    tmp_file.write(src.read())
                temp_file_path = tmp_file.name
            
            logger.info(f"Processing PDF: {pdf_file.name} (shuffle: {shuffle_enabled}, workflow: {workflow_type})")
            
            # Get orchestrator and process
            orchestrator = self.get_orchestrator(workflow_type)
            
            result = orchestrator.process_pdf(
                pdf_path=temp_file_path,
                filename=os.path.basename(pdf_file.name),
                shuffle_for_testing=shuffle_enabled,
                custom_config={
                    "show_debug_info": show_debug_info
                }
            )
            
            processing_time = time.time() - start_time
            
            # Format results
            main_result = self._format_main_result(result, processing_time, file_size)
            processing_details = self._format_processing_details(result, workflow_type, shuffle_enabled)
            debug_info = self._format_debug_info(result) if show_debug_info else "Debug info disabled"
            
            return main_result, processing_details, debug_info, ""
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(error_msg)
            return "❌ Processing failed", "", "", error_msg
        
        finally:
            # Cleanup temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
    
    def _format_main_result(self, result, processing_time: float, file_size: int) -> str:
        """Format main result as HTML"""
        
        success = result.confidence_score > 0
        status_emoji = "✅" if success else "❌"
        status_text = "Success" if success else "Failed"
        
        # Confidence color coding
        confidence = result.confidence_score
        if confidence >= 0.8:
            confidence_color = "green"
        elif confidence >= 0.6:
            confidence_color = "orange"
        else:
            confidence_color = "red"
        
        html = f"""
        <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <h2 style="margin: 0 0 15px 0;">{status_emoji} PDF Page Ordering Result</h2>
            
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h3 style="margin: 0 0 10px 0;">📊 Processing Summary</h3>
                <p><strong>Status:</strong> {status_text}</p>
                <p><strong>Method:</strong> {result.method_used.value.replace('_', ' ').title()}</p>
                <p><strong>Confidence:</strong> <span style="color: {confidence_color}; font-weight: bold;">{confidence:.1%}</span></p>
                <p><strong>Fragments Ordered:</strong> {len(result.ordered_fragment_ids)}</p>
                <p><strong>Processing Time:</strong> {format_duration(processing_time)}</p>
                <p><strong>File Size:</strong> {format_file_size(file_size)}</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                <h3 style="margin: 0 0 10px 0;">📄 Fragment Order</h3>
                <div style="max-height: 200px; overflow-y: auto; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px;">
                    <code style="color: white;">
                        {' → '.join([f"Fragment_{i+1}" for i in range(len(result.ordered_fragment_ids))])}
                    </code>
                </div>
            </div>
        </div>
        """
        
        return html
    
    def _format_processing_details(self, result, workflow_type: str, shuffle_enabled: bool) -> str:
        """Format processing details as HTML"""
        
        quality_metrics = result.quality_metrics
        
        html = f"""
        <div style="padding: 15px; border: 1px solid #ddd; border-radius: 8px; background: #f9f9f9;">
            <h3 style="color: #333; margin-bottom: 15px;">🔍 Processing Details</h3>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div>
                    <h4 style="color: #666; margin-bottom: 8px;">Configuration</h4>
                    <p><strong>Workflow Type:</strong> {workflow_type.title()}</p>
                    <p><strong>Shuffling:</strong> {'✅ Enabled' if shuffle_enabled else '❌ Disabled'}</p>
                    <p><strong>Page Hints Found:</strong> {result.page_hints_found}</p>
                    <p><strong>Section Analysis:</strong> {'✅ Success' if result.section_analysis_success else '❌ Failed'}</p>
                    <p><strong>Continuity Analysis:</strong> {'✅ Success' if result.continuity_analysis_success else '❌ Failed'}</p>
                </div>
                
                <div>
                    <h4 style="color: #666; margin-bottom: 8px;">Quality Metrics</h4>
        """
        
        for metric_name, value in quality_metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.1%}" if value <= 1.0 else f"{value:.2f}"
                html += f"<p><strong>{metric_name.replace('_', ' ').title()}:</strong> {formatted_value}</p>"
        
        html += """
                </div>
            </div>
        </div>
        """
        
        return html
    
    def _format_debug_info(self, result) -> str:
        """Format debug info as JSON"""
        debug_data = {
            "document_id": result.document_id,
            "method_used": result.method_used.value,
            "confidence_score": result.confidence_score,
            "processing_time": result.processing_time_seconds,
            "page_hints_found": result.page_hints_found,
            "quality_metrics": result.quality_metrics,
            "debug_info": result.debug_info
        }
        
        return json.dumps(debug_data, indent=2, ensure_ascii=False)
    
    def analyze_pdf_structure(self, pdf_file) -> Tuple[str, str]:
        """Analyze PDF structure without ordering"""
        if pdf_file is None:
            return "❌ No file uploaded", "Please upload a PDF file"
        
        temp_file_path = None
        
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                with open(pdf_file.name, 'rb') as src:
                    tmp_file.write(src.read())
                temp_file_path = tmp_file.name
            
            logger.info(f"Analyzing PDF structure: {pdf_file.name}")
            
            # Analyze structure
            orchestrator = self.get_orchestrator("standard")
            analysis = orchestrator.analyze_pdf_structure(
                pdf_path=temp_file_path,
                filename=os.path.basename(pdf_file.name)
            )
            
            # Format analysis result
            html_result = f"""
            <div style="padding: 15px; border-radius: 8px; background: #f0f8ff; border: 1px solid #4CAF50;">
                <h3 style="color: #2E7D32;">📋 Document Structure Analysis</h3>
                
                <div style="margin-bottom: 15px;">
                    <h4>📄 Document Info</h4>
                    <p><strong>PDF Type:</strong> {analysis.get('pdf_type', 'Unknown').replace('_', ' ').title()}</p>
                    <p><strong>Total Fragments:</strong> {analysis.get('total_fragments', 0)}</p>
                    <p><strong>Processing Time:</strong> {format_duration(analysis.get('processing_time', 0))}</p>
                </div>
                
                <div style="margin-bottom: 15px;">
                    <h4>🔢 Page Analysis</h4>
                    <p><strong>Page Hints Found:</strong> {analysis.get('page_hints_found', 0)}</p>
                    <p><strong>Coverage:</strong> {(analysis.get('page_hints_found', 0) / max(analysis.get('total_fragments', 1), 1) * 100):.1f}%</p>
                </div>
                
                <div style="margin-bottom: 15px;">
                    <h4>📊 Section Analysis</h4>
                    <p><strong>Hierarchy Detected:</strong> {analysis.get('section_analysis', {}).get('hierarchy_detected', False)}</p>
                    <p><strong>Numbering System:</strong> {analysis.get('section_analysis', {}).get('numbering_system', 'none')}</p>
                </div>
                
                <div>
                    <h4>🔗 Continuity Analysis</h4>
                    <p><strong>Total Pairs:</strong> {analysis.get('continuity_analysis', {}).get('total_pairs', 0)}</p>
                    <p><strong>Average Score:</strong> {analysis.get('continuity_analysis', {}).get('average_score', 0):.3f}</p>
                </div>
            </div>
            """
            
            # Detailed JSON
            json_result = json.dumps(analysis, indent=2, ensure_ascii=False)
            
            return html_result, json_result
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            return f"❌ Analysis failed: {error_msg}", ""
        
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

def create_demo_interface():
    """Create Gradio interface"""
    
    demo_app = PDFOrderingDemo()
    
    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
    }
    .gr-button-primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    """
    
    with gr.Blocks(css=css, title="PDF Page Ordering Demo", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.5em;">🚀 PDF Page Ordering System</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">AI-powered document page ordering using LangGraph workflows</p>
        </div>
        """)
        
        with gr.Tabs():
            # Main Processing Tab
            with gr.TabItem("📄 Process PDF"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h3>⚙️ Configuration</h3>")
                        
                        pdf_input = gr.File(
                            label="Upload PDF Document",
                            file_types=[".pdf"],
                            type="file"
                        )
                        
                        workflow_type = gr.Dropdown(
                            choices=["standard", "testing", "production"],
                            value="standard",
                            label="Workflow Type",
                            info="Standard: balanced, Testing: with debug info, Production: optimized"
                        )
                        
                        shuffle_enabled = gr.Checkbox(
                            label="Enable Shuffling (for accuracy testing)",
                            value=False,
                            info="Randomly shuffle pages before ordering to test accuracy"
                        )
                        
                        show_debug = gr.Checkbox(
                            label="Show Debug Information",
                            value=False
                        )
                        
                        process_btn = gr.Button(
                            "🚀 Process PDF",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        gr.HTML("<h3>📊 Results</h3>")
                        
                        main_output = gr.HTML(
                            label="Processing Result",
                            value="Upload a PDF file and click 'Process PDF' to get started"
                        )
                        
                        details_output = gr.HTML(
                            label="Processing Details"
                        )
                        
                        with gr.Accordion("🐛 Debug Information", open=False):
                            debug_output = gr.Code(
                                label="Debug JSON",
                                language="json"
                            )
                        
                        error_output = gr.Textbox(
                            label="Error Messages",
                            visible=False
                        )
            
            # Analysis Tab
            with gr.TabItem("🔍 Analyze Structure"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h3>📋 Document Analysis</h3>")
                        
                        analysis_pdf = gr.File(
                            label="Upload PDF for Analysis",
                            file_types=[".pdf"],
                            type="file"
                        )
                        
                        analyze_btn = gr.Button(
                            "🔍 Analyze Structure",
                            variant="primary"
                        )
                    
                    with gr.Column(scale=2):
                        analysis_output = gr.HTML(
                            label="Analysis Result",
                            value="Upload a PDF file to analyze its structure"
                        )
                        
                        with gr.Accordion("📄 Detailed Analysis (JSON)", open=False):
                            analysis_json = gr.Code(
                                label="Full Analysis Data",
                                language="json"
                            )
            
            # Info Tab
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## 🚀 PDF Page Ordering System
                
                This system uses advanced AI techniques to automatically order shuffled or disorganized PDF pages.
                
                ### ✨ Features
                
                - **Multi-format Support**: Works with text-based, scanned, and mixed PDFs
                - **Advanced Analysis**: 
                  - Page number extraction with pattern recognition
                  - Document section and hierarchy analysis
                  - Content continuity analysis between pages
                - **Multiple Ordering Strategies**:
                  - Direct page number matching
                  - Section-based ordering
                  - Content continuity analysis
                  - Hybrid approaches
                - **Quality Metrics**: Confidence scoring and processing quality assessment
                - **Testing Support**: Built-in shuffling to test ordering accuracy
                
                ### 🔧 Workflow Types
                
                - **Standard**: Balanced approach suitable for most documents
                - **Testing**: Includes detailed debug information and logging
                - **Production**: Optimized for speed and reliability
                
                ### 📋 How to Use
                
                1. **Upload** your PDF document
                2. **Configure** processing options (workflow type, shuffling, etc.)
                3. **Process** the document to get ordered pages
                4. **Review** results including confidence scores and quality metrics
                
                ### 🧪 Testing Accuracy
                
                Enable the "Shuffling" option to test the system's accuracy:
                - The system will randomly shuffle pages
                - Then attempt to restore the correct order
                - Compare results to assess ordering quality
                
                ### 🔍 Structure Analysis
                
                Use the "Analyze Structure" tab to:
                - Examine document type and structure
                - Check page numbering patterns
                - Analyze section hierarchy
                - Assess content quality without ordering
                """)
        
        # Event handlers
        process_btn.click(
            fn=demo_app.process_pdf,
            inputs=[pdf_input, shuffle_enabled, workflow_type, show_debug],
            outputs=[main_output, details_output, debug_output, error_output]
        )
        
        analyze_btn.click(
            fn=demo_app.analyze_pdf_structure,
            inputs=[analysis_pdf],
            outputs=[analysis_output, analysis_json]
        )
        
        # Example section
        gr.HTML("""
        <div style="margin-top: 30px; padding: 15px; background: #f0f8ff; border-radius: 8px; border-left: 4px solid #2196F3;">
            <h4 style="color: #1976D2; margin: 0 0 10px 0;">💡 Tips for Best Results</h4>
            <ul style="margin: 0; color: #666;">
                <li><strong>File Size:</strong> Keep PDFs under 50MB for optimal processing speed</li>
                <li><strong>Quality:</strong> Clear, high-resolution scans work better for OCR</li>
                <li><strong>Testing:</strong> Use shuffling mode to verify ordering accuracy</li>
                <li><strong>Languages:</strong> System supports Vietnamese and English text</li>
                <li><strong>Page Numbers:</strong> Documents with visible page numbers get higher confidence scores</li>
            </ul>
        </div>
        """)
    
    return demo

# Main execution
if __name__ == "__main__":
    # Create and launch demo
    demo = create_demo_interface()
    
    # Launch configuration
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create public link
        debug=True,             # Enable debug mode
        show_error=True,        # Show detailed errors
        quiet=False,            # Show startup logs
        inbrowser=True,         # Open browser automatically
        show_tips=True,         # Show usage tips
        height=800,             # Interface height
        favicon_path=None,      # Custom favicon
        auth=None,              # No authentication required
        max_threads=10          # Max concurrent users
    )