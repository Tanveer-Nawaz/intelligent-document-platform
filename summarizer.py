"""
Summarization and Insight Generation Module
Implements Methodology Section 3.c from project PDF
Lightweight extractive summarization without large model downloads
"""

from typing import List, Dict, Any
import json
import re
import os

class DocumentSummarizer:
    """
    Lightweight summarizer using extractive techniques
    Generates summaries without downloading large AI models
    """
    
    def __init__(self):
        print("\nInitializing Lightweight Summarization Module...")
        
        # Configuration for different summary types
        self.summary_types = {
            'executive': {'max_sentences': 3},
            'detailed': {'max_sentences': 7},
            'section_wise': {'max_sentences': 3}
        }
        
        print("✓ Lightweight summarizer ready")
    
    def generate_executive_summary(self, text: str) -> str:
        """Generate short executive summary for quick understanding"""
        return self._smart_extractive_summary(text, num_sentences=3, prefer_first_last=True)
    
    def generate_detailed_summary(self, text: str) -> str:
        """Generate detailed summary preserving technical depth"""
        return self._smart_extractive_summary(text, num_sentences=7, prefer_keywords=True)
    
    def generate_section_summaries(self, sections: Dict[str, str]) -> Dict[str, str]:
        """Generate summaries for each document section"""
        section_summaries = {}
        
        for section_name, section_content in sections.items():
            if len(section_content) > 100:
                # Get first few sentences from each section
                sentences = self._split_into_sentences(section_content)
                if len(sentences) > 3:
                    summary = ' '.join(sentences[:3])
                else:
                    summary = ' '.join(sentences)
                section_summaries[section_name] = summary
            else:
                section_summaries[section_name] = section_content
        
        return section_summaries
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences"""
        # Split by sentence endings (. ! ?)
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter short sentences
        cleaned = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) > 3:
                cleaned.append(sentence)
        
        return cleaned
    
    def _smart_extractive_summary(self, text: str, num_sentences: int = 5, 
                                 prefer_first_last: bool = False, 
                                 prefer_keywords: bool = False) -> str:
        """
        Extract key sentences to create a summary
        Scores sentences based on importance indicators
        """
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Score each sentence based on importance
        scored_sentences = []
        important_keywords = [
            'summary', 'conclusion', 'result', 'find', 'show', 'demonstrate',
            'important', 'key', 'main', 'primary', 'critical', 'essential',
            'recommend', 'suggest', 'propose', 'should', 'must', 'need',
            'risk', 'challenge', 'issue', 'problem', 'solution', 'approach'
        ]
        
        for i, sentence in enumerate(sentences):
            score = 0
            
            # First and last sentences are often important
            if prefer_first_last:
                if i < 3 or i > len(sentences) - 4:
                    score += 2
            
            # Sentences with important keywords get higher scores
            if prefer_keywords:
                sentence_lower = sentence.lower()
                for keyword in important_keywords:
                    if keyword in sentence_lower:
                        score += 3
            
            # Medium-length sentences are often good summaries
            word_count = len(sentence.split())
            if 10 <= word_count <= 30:
                score += 1
            
            # Questions can be important
            if '?' in sentence:
                score += 1
            
            scored_sentences.append((score, i, sentence))
        
        # Sort sentences by score (highest first)
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        # Take top sentences and reorder to original sequence
        top_scored = scored_sentences[:num_sentences]
        top_scored.sort(key=lambda x: x[1])
        
        # Extract sentence text
        top_sentences = [s[2] for s in top_scored]
        
        return ' '.join(top_sentences)
    
    def structure_insights(self, entities: Dict, topics: Dict, 
                          insights: Dict, sections: Dict) -> Dict[str, Any]:
        """Create structured JSON output from analysis results"""
        structured_output = {
            "document_metadata": {
                "total_sections": len(sections),
                "summary_types_generated": list(self.summary_types.keys())
            },
            "summaries": {
                "executive_summary": "",
                "detailed_summary": "",
                "section_summaries": {}
            },
            "actionable_insights": {
                "key_decisions": [],
                "deadlines_timelines": [],
                "risks_constraints": [],
                "recommendations_next_steps": []
            },
            "extracted_entities": entities,
            "identified_topics": topics,
            "document_structure": {
                "section_count": len(sections),
                "section_names": list(sections.keys())[:10]
            }
        }
        
        return structured_output

# ============================================================================
# Complete Pipeline with Summarization
# ============================================================================

def complete_analysis():
    """Run complete analysis pipeline with summarization"""
    
    print("=" * 60)
    print("Complete Document Analysis Pipeline")
    print("Includes all 3 methodology sections")
    print("=" * 60)
    
    # Check for existing analysis results
    if not os.path.exists("document_analysis_results.json"):
        print("Previous analysis results not found!")
        return
    
    # Load previous analysis
    print("\nLoading previous analysis results...")
    with open("document_analysis_results.json", "r", encoding="utf-8") as f:
        previous_results = json.load(f)
    
    print(f"✓ Loaded analysis of: {previous_results.get('document', 'Unknown')}")
    
    # Initialize summarizer
    summarizer = DocumentSummarizer()
    
    # Get document text for summarization
    text = ""
    if os.path.exists("document_preview.txt"):
        with open("document_preview.txt", "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split('\n')
            text_started = False
            for line in lines:
                if "FIRST 1000 CHARACTERS:" in line:
                    text_started = True
                    continue
                if text_started and line.strip():
                    text += line + " "
    
    # Generate summaries
    print("\nGenerating Summaries...")
    
    # 1. Executive Summary
    exec_summary = summarizer.generate_executive_summary(text[:2000])
    
    # 2. Detailed Summary
    detailed_summary = summarizer.generate_detailed_summary(text[:3000])
    
    # 3. Section-wise Summaries
    if 'sections' in previous_results:
        section_summaries = summarizer.generate_section_summaries(
            previous_results['sections']
        )
    
    # Create structured output
    structured_insights = summarizer.structure_insights(
        entities=previous_results.get('entities', {}),
        topics=previous_results.get('topics', {}),
        insights=previous_results.get('actionable_insights', {}),
        sections=previous_results.get('sections', {})
    )
    
    # Fill in the summaries
    structured_insights['summaries']['executive_summary'] = exec_summary
    structured_insights['summaries']['detailed_summary'] = detailed_summary
    structured_insights['summaries']['section_summaries'] = section_summaries
    
    # Save structured output
    output_file = "structured_document_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(structured_insights, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Structured insights saved to: {output_file}")
    print("\nAll 3 methodology sections implemented:")
    print("  ✓ 3.a - Document Ingestion & Segmentation")
    print("  ✓ 3.b - Semantic Understanding & Information Extraction")
    print("  ✓ 3.c - Summarization & Insight Generation")

if __name__ == "__main__":
    complete_analysis()