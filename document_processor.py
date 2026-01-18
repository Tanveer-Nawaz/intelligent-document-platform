"""
Complete Document Analysis Pipeline
Implements Methodology 3.a (Ingestion & Segmentation) and 3.b (Semantic Understanding)
"""

import pdfplumber
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import re
import json
from typing import List, Dict, Tuple
import os
from collections import Counter

print("=" * 60)
print("Intelligent Document Understanding Platform")
print("Document Processing Module")
print("=" * 60)

# ============================================================================
# Part 1: Document Ingestion & Segmentation (Methodology 3.a)
# ============================================================================

class DocumentIngestion:
    """Handles document ingestion and segmentation as per Methodology 3.a"""
    
    def __init__(self):
        print("\nInitializing Document Ingestion Module...")
        # Load tokenizer for text chunking
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        print("✓ Tokenizer loaded")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from PDF files"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    def segment_document(self, text: str) -> Dict[str, str]:
        """Split document into logical sections based on headings"""
        sections = {}
        current_section = "Document Start"
        section_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            is_header = False
            
            # Pattern 1: Numbered sections like "1. Introduction"
            if re.match(r'^\d+\.\s+[A-Z]', line):
                is_header = True
            
            # Pattern 2: All caps headings
            if line.isupper() and len(line) > 3 and len(line.split()) < 6:
                is_header = True
            
            # Pattern 3: Common section keywords
            section_keywords = ['abstract', 'introduction', 'methodology', 'conclusion', 
                               'references', 'results', 'discussion', 'summary']
            if any(keyword in line.lower() for keyword in section_keywords):
                is_header = True
            
            if is_header:
                # Save previous section
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content)
                
                # Start new section
                current_section = line[:50]
                section_content = [line]
            else:
                section_content.append(line)
        
        # Add final section
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content)
        
        return sections
    
    def chunk_document(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split long documents into overlapping chunks for processing"""
        # Convert text to tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        start = 0
        
        # Create overlapping chunks
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            
            # Convert tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunk_text = ' '.join(chunk_text.split())  # Clean whitespace
            
            if chunk_text:
                chunks.append(chunk_text)
            
            # Move with overlap for continuity
            start += chunk_size - overlap
        
        return chunks

# ============================================================================
# Part 2: Semantic Understanding & Information Extraction (Methodology 3.b)
# ============================================================================

class SemanticUnderstanding:
    """Implements semantic understanding as per Methodology 3.b"""
    
    def __init__(self):
        print("\nInitializing Semantic Understanding Module...")
        
        try:
            # Initialize embedding model for semantic analysis
            print("  Loading sentence transformer...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("  ✓ Sentence transformer loaded")
        except Exception as e:
            print(f"  Could not load sentence transformer: {e}")
            self.embedding_model = None
        
        print("✓ Semantic module ready")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {
            'ORGANIZATIONS': [],
            'PERSONS': [],
            'DATES': [],
            'TECHNICAL_TERMS': [],
            'KEY_PHRASES': []
        }
        
        # Extract dates using regex patterns
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{4}-\d{2}-\d{2}',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
        ]
        
        all_dates = []
        for pattern in date_patterns:
            dates = re.findall(pattern, text, re.IGNORECASE)
            all_dates.extend(dates)
        
        entities['DATES'] = list(set(all_dates))
        
        # Extract organizations (Title Case patterns)
        org_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
        orgs = re.findall(org_pattern, text)
        
        # Filter out common words that aren't organizations
        common_words = ['The', 'This', 'That', 'There', 'However', 'Therefore']
        filtered_orgs = []
        for org in orgs:
            first_word = org.split()[0]
            if first_word not in common_words and len(org.split()) >= 2:
                filtered_orgs.append(org)
        
        entities['ORGANIZATIONS'] = list(set(filtered_orgs))
        
        # Extract technical terms and acronyms
        tech_terms = re.findall(r'\b(?:NLP|AI|ML|API|JSON|PDF|DOCX|LLM|BERT|GPT|Transformer)\b', text, re.IGNORECASE)
        entities['TECHNICAL_TERMS'] = list(set([term.upper() for term in tech_terms]))
        
        # Extract key phrases containing important keywords
        important_keywords = ['must', 'should', 'requires', 'important', 'critical', 
                             'deadline', 'risk', 'recommendation', 'decision']
        
        sentences = re.split(r'[.!?]+', text)
        key_phrases = []
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in important_keywords):
                if len(sentence) > 20:
                    key_phrases.append(sentence[:150] + "..." if len(sentence) > 150 else sentence)
        
        entities['KEY_PHRASES'] = key_phrases[:5]  # Limit to 5 key phrases
        
        return entities
    
    def extract_topics(self, chunks: List[str]) -> Dict[str, any]:
        """Identify main topics in document"""
        topics = {}
        
        if not chunks:
            return {"error": "No text provided"}
        
        # Combine all chunks for topic analysis
        all_text = ' '.join(chunks).lower()
        
        # Define common document themes to look for
        potential_topics = {
            'NLP_AND_AI': ['nlp', 'ai', 'machine learning', 'artificial intelligence', 'transformer'],
            'DOCUMENT_PROCESSING': ['document', 'text', 'extract', 'process', 'parse'],
            'SUMMARIZATION': ['summariz', 'summary', 'abstract', 'overview'],
            'INFORMATION_EXTRACTION': ['extract', 'entity', 'recognit', 'ner', 'relationship'],
            'ENTERPRISE_SOLUTION': ['enterprise', 'business', 'organization', 'company', 'platform']
        }
        
        # Count keyword occurrences for each topic
        for topic_name, keywords in potential_topics.items():
            count = 0
            for keyword in keywords:
                count += all_text.count(keyword)
            
            if count > 0:
                topics[topic_name] = {
                    'frequency': count,
                    'keywords': [k for k in keywords if k in all_text]
                }
        
        # Fallback: use most common words if no topics found
        if not topics:
            words = re.findall(r'\b[a-z]{4,}\b', all_text)
            word_counts = Counter(words)
            common_words = [word for word, count in word_counts.most_common(5)]
            
            for i, word in enumerate(common_words[:3]):
                topics[f"TOPIC_{i+1}"] = {
                    'frequency': word_counts[word],
                    'keywords': [word]
                }
        
        return topics
    
    def extract_actionable_insights(self, text: str) -> Dict[str, List[str]]:
        """Extract deadlines, risks, recommendations, and decisions"""
        insights = {
            'DEADLINES': [],
            'RISKS_AND_CONSTRAINTS': [],
            'RECOMMENDATIONS': [],
            'KEY_DECISIONS': []
        }
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            
            s_lower = sentence.lower()
            
            # Check for deadline patterns
            deadline_patterns = [
                r'(?:by|before|on|due\s+by|deadline|until)\s+[A-Za-z0-9\s,]+',
                r'\d{1,2}/\d{1,2}/\d{2,4}',
                r'(?:Q[1-4]\s+\d{4})'
            ]
            
            for pattern in deadline_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    insights['DEADLINES'].append(sentence[:100] + "..." if len(sentence) > 100 else sentence)
                    break
            
            # Check for risk indicators
            risk_words = ['risk', 'challenge', 'problem', 'limitation', 'constraint', 'issue', 'difficulty']
            if any(word in s_lower for word in risk_words):
                insights['RISKS_AND_CONSTRAINTS'].append(sentence[:120] + "..." if len(sentence) > 120 else sentence)
            
            # Check for recommendations
            recommendation_words = ['recommend', 'suggest', 'propose', 'advise', 'should', 'must', 'need to']
            if any(word in s_lower for word in recommendation_words):
                insights['RECOMMENDATIONS'].append(sentence[:120] + "..." if len(sentence) > 120 else sentence)
            
            # Check for decisions
            decision_words = ['decide', 'conclude', 'determine', 'choose', 'select', 'decision']
            if any(word in s_lower for word in decision_words):
                insights['KEY_DECISIONS'].append(sentence[:120] + "..." if len(sentence) > 120 else sentence)
        
        # Limit to 3 items per category for brevity
        for key in insights:
            insights[key] = insights[key][:3]
        
        return insights

# ============================================================================
# Main Execution (for testing)
# ============================================================================

def main():
    """Test the document processing pipeline"""
    
    pdf_file = "Team Alpha.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"\nError: File '{pdf_file}' not found!")
        return
    
    print(f"\nProcessing document: {pdf_file}")
    print("-" * 50)
    
    # Step 1: Document Ingestion
    print("\nStep 1: Document Ingestion & Segmentation")
    ingestion = DocumentIngestion()
    
    text = ingestion.extract_text_from_pdf(pdf_file)
    
    if not text:
        print("Failed to extract text from PDF")
        return
    
    print(f"✓ Extracted {len(text)} characters, {len(text.split())} words")
    
    sections = ingestion.segment_document(text)
    print(f"✓ Document segmented into {len(sections)} sections")
    
    chunks = ingestion.chunk_document(text, chunk_size=768, overlap=100)
    print(f"✓ Document chunked into {len(chunks)} chunks")
    
    # Step 2: Semantic Understanding
    print("\nStep 2: Semantic Understanding")
    semantic = SemanticUnderstanding()
    
    if chunks:
        sample_chunk = chunks[0]
        
        entities = semantic.extract_entities(sample_chunk)
        topics = semantic.extract_topics(chunks)
        insights = semantic.extract_actionable_insights(sample_chunk)
        
        print(f"✓ Extracted {sum(len(v) for v in entities.values())} entities")
        print(f"✓ Identified {len(topics)} topics")
        print(f"✓ Found {sum(len(v) for v in insights.values())} actionable insights")
    
    print("\nDocument processing complete!")

if __name__ == "__main__":
    main()