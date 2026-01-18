"""
Flask Web Application for Intelligent Document Understanding Platform
Main web interface for document upload, processing, and analysis.
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
import os
import uuid
import json
from werkzeug.utils import secure_filename
from datetime import datetime

# Try importing our document processing modules
try:
    from document_processor import DocumentIngestion, SemanticUnderstanding
    from summarizer import DocumentSummarizer
    print("✓ Document processing modules imported")
except ImportError:
    # Fallback classes for testing if modules aren't available
    class DocumentIngestion:
        def extract_text_from_pdf(self, path): return "Sample text"
        def segment_document(self, text): return {"section": "content"}
        def chunk_document(self, text): return ["chunk1", "chunk2"]
    
    class SemanticUnderstanding:
        def extract_entities(self, text): return {"ORG": ["Test"]}
        def extract_topics(self, chunks): return {"topic": {"freq": 1}}
        def extract_actionable_insights(self, text): return {"risk": ["test"]}
    
    class DocumentSummarizer:
        def generate_executive_summary(self, text): return "Executive summary"
        def generate_detailed_summary(self, text): return "Detailed summary"
        def generate_section_summaries(self, sections): return {"section": "summary"}

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'document-platform-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB file size limit
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'data'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'doc'}

# Create required directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_analysis_result(document_id, result):
    """Save analysis results to JSON file for later retrieval"""
    result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{document_id}_analysis.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result_file

def load_analysis_result(document_id):
    """Load previously saved analysis results"""
    result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{document_id}_analysis.json")
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

@app.route('/')
def index():
    """Home page - shows document upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle document upload from web form"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique ID for this document
        document_id = str(uuid.uuid4())[:8]
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{document_id}_{filename}")
        file.save(filepath)
        
        # Store document info in session
        session['document_id'] = document_id
        session['filename'] = filename
        session['filepath'] = filepath
        
        return jsonify({
            'success': True,
            'document_id': document_id,
            'filename': filename,
            'redirect': url_for('process', document_id=document_id)
        })
    
    return jsonify({'error': 'Please upload PDF or DOCX files only'}), 400

@app.route('/process/<document_id>')
def process(document_id):
    """Display processing page while document is being analyzed"""
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return redirect(url_for('index'))
    
    return render_template('processing.html', 
                         document_id=document_id, 
                         filename=session.get('filename', 'Unknown'))

@app.route('/api/analyze/<document_id>', methods=['POST'])
def analyze_document(document_id):
    """Main analysis endpoint - processes document using our pipeline"""
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        print(f"Starting analysis for document: {document_id}")
        
        # Step 1: Document Ingestion & Segmentation
        print("Step 1: Document Ingestion")
        ingestion = DocumentIngestion()
        text = ingestion.extract_text_from_pdf(filepath)
        
        if not text:
            return jsonify({'error': 'Could not extract text from document'}), 400
        
        sections = ingestion.segment_document(text)
        chunks = ingestion.chunk_document(text, chunk_size=768, overlap=100)
        
        # Step 2: Semantic Understanding
        print("Step 2: Semantic Understanding")
        semantic = SemanticUnderstanding()
        
        # Extract entities and insights from first chunk
        if chunks:
            sample_chunk = chunks[0]
            entities = semantic.extract_entities(sample_chunk)
            insights = semantic.extract_actionable_insights(sample_chunk)
        else:
            entities = {}
            insights = {}
        
        topics = semantic.extract_topics(chunks)
        
        # Step 3: Summarization
        print("Step 3: Summarization")
        summarizer = DocumentSummarizer()
        exec_summary = summarizer.generate_executive_summary(text[:2000])
        detailed_summary = summarizer.generate_detailed_summary(text[:3000])
        section_summaries = summarizer.generate_section_summaries(sections)
        
        # Prepare final results
        results = {
            'document_id': document_id,
            'filename': session.get('filename'),
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'total_characters': len(text),
                'total_words': len(text.split()),
                'sections_identified': len(sections),
                'chunks_created': len(chunks)
            },
            'sections': {k: v[:500] + "..." for k, v in sections.items()},
            'summaries': {
                'executive': exec_summary,
                'detailed': detailed_summary,
                'section_summaries': section_summaries
            },
            'entities': entities,
            'topics': topics,
            'insights': insights,
            'chunks': chunks[:10]  # Show first 10 chunks only
        }
        
        # Save results for later access
        save_analysis_result(document_id, results)
        
        print(f"Analysis complete for {document_id}")
        
        return jsonify({
            'success': True,
            'results': results,
            'redirect': url_for('results', document_id=document_id)
        })
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/results/<document_id>')
def results(document_id):
    """Display analysis results page"""
    results = load_analysis_result(document_id)
    if not results:
        return redirect(url_for('index'))
    
    return render_template('results.html', 
                         results=results, 
                         document_id=document_id)

@app.route('/api/search/<document_id>', methods=['POST'])
def search_document(document_id):
    """Search within analyzed document content"""
    query = request.json.get('query', '')
    if not query:
        return jsonify({'error': 'Please enter a search query'}), 400
    
    results = load_analysis_result(document_id)
    if not results:
        return jsonify({'error': 'Document not found'}), 404
    
    # Simple text search implementation
    matches = []
    query_lower = query.lower()
    
    # Search in document chunks
    if 'chunks' in results:
        for i, chunk in enumerate(results['chunks']):
            if query_lower in chunk.lower():
                matches.append({
                    'type': 'chunk',
                    'index': i,
                    'content': chunk[:200] + "...",
                    'relevance': chunk.lower().count(query_lower)
                })
    
    # Search in document sections
    if 'sections' in results:
        for section_name, content in results['sections'].items():
            if query_lower in content.lower():
                matches.append({
                    'type': 'section',
                    'name': section_name,
                    'content': content[:200] + "...",
                    'relevance': content.lower().count(query_lower)
                })
    
    # Sort matches by relevance (most occurrences first)
    matches.sort(key=lambda x: x['relevance'], reverse=True)
    
    return jsonify({
        'query': query,
        'matches': matches[:10],
        'total_matches': len(matches)
    })

@app.route('/download/<document_id>/<format_type>')
def download_results(document_id, format_type):
    """Download analysis results in different formats"""
    results = load_analysis_result(document_id)
    if not results:
        return "Document not found", 404
    
    if format_type == 'json':
        # Download complete JSON data
        filename = f"{document_id}_analysis.json"
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        return send_file(filepath, as_attachment=True, download_name=filename)
    
    elif format_type == 'report':
        # Generate human-readable text report
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("=" * 60 + "\n")
            f.write(f"DOCUMENT ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Document: {results.get('filename', 'Unknown')}\n")
            f.write(f"Analyzed on: {results.get('timestamp', 'Unknown')}\n")
            f.write(f"Total words: {results.get('statistics', {}).get('total_words', 0)}\n\n")
            
            f.write("EXECUTIVE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(results.get('summaries', {}).get('executive', '') + "\n\n")
            
            f.write("KEY INSIGHTS:\n")
            f.write("-" * 40 + "\n")
            insights = results.get('insights', {})
            for insight_type, items in insights.items():
                if items:
                    f.write(f"\n{insight_type}:\n")
                    for i, item in enumerate(items, 1):
                        f.write(f"  {i}. {item}\n")
            
            f.flush()
            return send_file(f.name, as_attachment=True, download_name=f"{document_id}_report.txt")
    
    return "Invalid format requested", 400

@app.route('/clear/<document_id>')
def clear_analysis(document_id):
    """Clear analysis results for a document"""
    result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{document_id}_analysis.json")
    if os.path.exists(result_file):
        os.remove(result_file)
    
    # Clear session data
    session.pop('document_id', None)
    session.pop('filename', None)
    session.pop('filepath', None)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Run the Flask application
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)