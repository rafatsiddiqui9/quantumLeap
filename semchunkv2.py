# Import statements and data structures
import tiktoken
from openai import OpenAI
import json
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import os
from datetime import datetime
from pprint import pprint
import re
from dataclasses import dataclass
from enum import Enum

# Core data structures
class SectionType(Enum):
    HEADER = "header"
    CONTENT = "content"
    QUOTE = "quote"
    ATTRIBUTION = "attribution"
    LIST = "list"
    FRONT_MATTER = "front_matter"
    TABLE_OF_CONTENTS = "table_of_contents"
    
@dataclass
class Section:
    text: str
    type: SectionType
    level: int = 0
    metadata: Dict = None

# Main class initialization
class SemanticChunker:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        """Initialize the semantic chunker with model configuration"""
        self.client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="dummy"
        )
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.max_tokens = 3000
        
        # Set up logging directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"/home/ubuntu/quantumLeap/data/preprocess/Step_2_Classic_Texts_and_Ethnographies/2.1_Public_Domain_Books/Project_Gutenberg/Psychology_Of_Unconscious_Mind/chunks_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logging file for processing summary
        self.log_file = os.path.join(self.log_dir, "processing_log.txt")
        
        # Initialize state variables
        self.missed_text = ""  # Store text not included in LLM output

    def log_message(self, message: str):
        """Write log message with timestamp and print to console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
        print(log_entry)
    
    def print_separator(self, message: str = ""):
        """Print a separator line with optional message"""
        print(f"\n{'='*100}")
        if message:
            print(f"{message}")
            print('='*100)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    # Continuing the SemanticChunker class...

    def is_chapter_heading(self, text: str) -> Tuple[bool, int]:
        """
        Enhanced chapter heading detection with level identification.
        Returns (is_heading, level).
        """
        text = text.strip()
        if not text:
            return False, 0
            
        # Chapter patterns
        chapter_patterns = [
            (r'^CHAPTER\s+[IVXL]+', 1),  # Main chapter headers
            (r'^[IVX]+\.\s*—\s*', 2),    # Sub-chapter headers
            (r'^\d+\.\s*—\s*', 2),       # Numbered sections
            (r'^\s*[A-Z][A-Z\s]+$', 1),  # ALL CAPS lines
            (r'^\s*[IVX]+\.\s+[A-Z]', 2) # Roman numeral sections
        ]
        
        for pattern, level in chapter_patterns:
            if re.match(pattern, text, re.I):
                return True, level
        
        # Check for centered text formatting
        line_length = len(text)
        leading_spaces = len(text) - len(text.lstrip())
        trailing_spaces = len(text) - len(text.rstrip())
        
        # Improved centered text detection
        is_centered = (abs(leading_spaces - trailing_spaces) <= 2 and 
                      leading_spaces > 5 and
                      text.strip())  # Must have content
        is_caps = text.isupper()
        reasonable_length = 10 < len(text.strip()) < 100
        
        if is_centered:
            if is_caps and reasonable_length:
                return True, 1  # Main header
            elif reasonable_length:
                return True, 2  # Sub header
                
        return False, 0
    
    def analyze_text_structure(self, text: str) -> List[Section]:
        """
        Enhanced text structure analysis with improved section detection.
        """
        sections = []
        lines = text.split('\n')
        current_section = []
        current_type = None
        current_level = 0
        
        def flush_section():
            nonlocal current_section, current_type, current_level
            if current_section:
                # Skip empty sections but preserve intentional spacing
                content = '\n'.join(current_section).strip()
                if content or current_type in {SectionType.HEADER, SectionType.FRONT_MATTER}:
                    sections.append(Section(
                        text='\n'.join(current_section),
                        type=current_type or SectionType.CONTENT,
                        level=current_level
                    ))
                current_section = []
                current_type = None
                current_level = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            
            # Detect centered headers
            if line.strip() and line.strip().isupper():
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces > 10:  # Likely centered
                    flush_section()
                    current_type = SectionType.HEADER
                    current_level = 1
                    current_section = [line]
                    # Include following blank lines
                    while i + 1 < len(lines) and not lines[i + 1].strip():
                        current_section.append(lines[i + 1])
                        i += 1
                    flush_section()
                    i += 1
                    continue
            
            # Detect Table of Contents
            if re.match(r'^\s*CONTENTS\s*$', line, re.I):
                flush_section()
                current_type = SectionType.TABLE_OF_CONTENTS
                current_section = [line]
                # Include following blank lines
                while i + 1 < len(lines) and not lines[i + 1].strip():
                    current_section.append(lines[i + 1])
                    i += 1
                i += 1
                continue
            
            # Detect Front Matter
            if re.match(r'^\s*(?:AUTHOR\'S\s+NOTE|PREFACE|INTRODUCTION)\s*$', line, re.I):
                flush_section()
                current_type = SectionType.FRONT_MATTER
                current_section = [line]
                # Include following blank lines
                while i + 1 < len(lines) and not lines[i + 1].strip():
                    current_section.append(lines[i + 1])
                    i += 1
                i += 1
                continue
            
            # Detect chapter headings
            is_heading, level = self.is_chapter_heading(line)
            if is_heading:
                flush_section()
                current_type = SectionType.HEADER
                current_level = level
                current_section = [line]
                # Include following blank lines
                while i + 1 < len(lines) and not lines[i + 1].strip():
                    current_section.append(lines[i + 1])
                    i += 1
                flush_section()
                i += 1
                continue
            
            # Detect quotes
            if ((line.startswith('"') and len(line) > 50) or 
                (line.startswith('_') and line.endswith('_'))):
                if current_type != SectionType.QUOTE:
                    flush_section()
                    current_type = SectionType.QUOTE
                
            # Detect attributions
            if re.match(r'^\s*(?:—|--)\s*[A-Z]', line):
                flush_section()
                current_type = SectionType.ATTRIBUTION
                current_section = [line]
                i += 1
                continue
                
            # Detect lists
            if re.match(r'^\s{4,}(?:[\w\-]+\.|\-|\*)\s', line):
                if current_type != SectionType.LIST:
                    flush_section()
                    current_type = SectionType.LIST
            
            current_section.append(line)
            i += 1
            
            # Handle section transitions
            if i < len(lines):
                next_line = lines[i]
                # Detect section breaks by multiple blank lines
                if (not line.strip() and not next_line.strip() and 
                    current_type not in {SectionType.HEADER, SectionType.FRONT_MATTER}):
                    flush_section()
        
        flush_section()  # Flush any remaining content
        
        # Filter and clean sections
        filtered_sections = []
        for section in sections:
            if section.text.strip() or section.type in {SectionType.HEADER, SectionType.FRONT_MATTER}:
                filtered_sections.append(section)
        
        return filtered_sections
    
    # Continuing the SemanticChunker class...

    def get_complete_paragraphs(self, text: str, max_tokens: int) -> Tuple[str, str]:
        """
        Enhanced version with improved content handling and structure preservation.
        """
        self.log_message(f"Starting get_complete_paragraphs with {len(text)} chars of text")
        
        # First, analyze the structure
        sections = self.analyze_text_structure(text)
        self.log_message(f"Found {len(sections)} sections")
        
        current_sections = []
        current_tokens = 0
        section_index = 0
        
        try:
            while section_index < len(sections):
                section = sections[section_index]
                section_tokens = self.count_tokens(section.text)
                
                self.log_message(f"Processing section {section_index + 1}: {section.type}, {section_tokens} tokens")
                
                # Handle oversized sections
                if section_tokens > max_tokens:
                    if not current_sections:  # If this is our first section
                        self.log_message(f"Warning: Section {section_index + 1} exceeds token limit")
                        # Try to split at paragraph boundary
                        paragraphs = section.text.split('\n\n')
                        current_text = ""
                        for para in paragraphs:
                            if self.count_tokens(current_text + para) > max_tokens:
                                break
                            current_text += para + '\n\n'
                        if current_text:
                            current_sections.append(Section(current_text.rstrip(), section.type, section.level))
                        remaining_text = section.text[len(current_text):]
                        if remaining_text:
                            self.missed_text = remaining_text
                        section_index += 1
                        continue
                    else:
                        break
                
                # If adding this section would exceed the limit
                if current_tokens + section_tokens > max_tokens:
                    if current_sections:  # Only break if we have content
                        break
                
                # Handle headers and their content together
                if section.type == SectionType.HEADER:
                    header_and_content = [section]
                    total_tokens = section_tokens
                    
                    # Look ahead for associated content
                    next_idx = section_index + 1
                    while (next_idx < len(sections) and 
                           sections[next_idx].type == SectionType.CONTENT and 
                           total_tokens + self.count_tokens(sections[next_idx].text) <= max_tokens):
                        header_and_content.append(sections[next_idx])
                        total_tokens += self.count_tokens(sections[next_idx].text)
                        next_idx += 1
                    
                    # Add header and its content
                    current_sections.extend(header_and_content)
                    current_tokens = total_tokens
                    section_index = next_idx
                    continue
                
                # Handle other section types
                current_sections.append(section)
                current_tokens += section_tokens
                section_index += 1
                
                self.log_message(f"After processing: tokens={current_tokens}/{max_tokens}, sections={len(current_sections)}")
            
            # Combine sections with proper spacing
            processed_sections = []
            for i, section in enumerate(current_sections):
                if i > 0:  # Add spacing before sections
                    if section.type == SectionType.HEADER or current_sections[i-1].type == SectionType.HEADER:
                        processed_sections.append("")  # Extra line before/after headers
                    processed_sections.append("")  # Standard section spacing
                
                processed_sections.append(section.text.rstrip())
                
                # Add extra spacing after headers
                if section.type == SectionType.HEADER:
                    processed_sections.append("")
            
            processed_text = "\n".join(processed_sections)
            
            # Prepare remaining text
            remaining_sections = sections[section_index:]
            remaining_text = ""
            if remaining_sections:
                remaining_parts = []
                for section in remaining_sections:
                    if remaining_parts:  # Add spacing between sections
                        remaining_parts.append("")
                    remaining_parts.append(section.text.rstrip())
                remaining_text = "\n".join(remaining_parts)
            
            self.log_message(f"Processed {len(current_sections)} sections, {len(remaining_sections)} remaining")
            return processed_text, remaining_text
            
        except Exception as e:
            self.log_message(f"Error in get_complete_paragraphs: {str(e)}")
            if current_sections:
                return "\n".join(s.text for s in current_sections), text
            return "", text
    
    def create_initial_chunks(self, text: str) -> List[str]:
        """
        Create initial chunks while preserving document structure.
        """
        chunks = []
        remaining_text = text
        chunk_number = 0
        
        while remaining_text.strip():
            chunk_number += 1
            self.log_message(f"\nProcessing chunk {chunk_number}")
            
            # Handle missed text from previous chunk
            if self.missed_text:
                self.log_message("Adding missed text from previous chunk")
                remaining_text = self.missed_text + '\n\n' + remaining_text
                self.missed_text = ""
            
            # Get complete paragraphs up to token limit
            chunk_text, remaining_text = self.get_complete_paragraphs(remaining_text, self.max_tokens)
            
            if chunk_text.strip():
                token_count = self.count_tokens(chunk_text)
                self.log_message(f"Created chunk {chunk_number} ({token_count} tokens)")
                chunks.append(chunk_text)
                
                # Log chunk preview
                preview_length = min(len(chunk_text), 500)
                preview = chunk_text[:preview_length] + ("..." if preview_length < len(chunk_text) else "")
                self.log_message(f"Chunk preview:\n{preview}")
            else:
                self.log_message("Warning: Empty chunk produced")
                if not remaining_text.strip():
                    break
            
            # Safety limit
            if len(chunks) >= 100:
                self.log_message("Warning: Maximum chunk limit reached")
                break
        
        # Save chunks to files
        self.log_message(f"Created {len(chunks)} initial chunks")
        os.makedirs(self.log_dir, exist_ok=True)
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(self.log_dir, f"chunk_{i+1:04d}.txt")
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
        
        return chunks
    
    # Continuing the SemanticChunker class...

    def get_semantic_sections(self, chunk: str) -> Tuple[List[str], Dict]:
        """
        Process chunks through LLM for semantic analysis with improved handling.
        """
        try:
            self.log_message(f"Sending request to LLM (input tokens: {self.count_tokens(chunk)})")
            
            # Enhanced prompt for better structure preservation
            system_prompt = """You are a text analysis expert. Your task is to:
            1. Maintain the original document structure exactly as provided
            2. Split the input into semantically coherent sections
            3. Preserve all formatting, spacing, and special characters
            4. Keep headers with their associated content
            5. Keep lists and quotes intact within their sections
            6. Mark sections using <START_SECTION> and <END_SECTION> tags
            7. Mark incomplete sections with <INCOMPLETE> tags
            8. Handle front matter, tables of contents, and chapter headings appropriately
            9. Preserve all original line breaks and paragraph spacing

            Critical Rules:
            - Do not modify any text content
            - Preserve all original formatting
            - Keep structural elements together (headers with content)
            - Maintain document hierarchy
            - Include ALL text - do not skip anything
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Split this text into sections, preserving ALL content and structure:\n\n{chunk}"}
                ],
                max_tokens=self.max_tokens,
                temperature=0.2,
                timeout=60  # 1-minute timeout
            )
            
            result = response.choices[0].message.content
            
            # Extract sections with improved parsing
            sections = []
            section_pattern = r'<START_SECTION>(.*?)<END_SECTION>'
            for match in re.finditer(section_pattern, result, re.DOTALL):
                section_text = match.group(1).strip()
                if section_text:  # Keep even short sections if they're structural
                    if len(section_text) > 50 or any(marker in section_text.upper() 
                        for marker in ['CHAPTER', 'CONTENTS', 'NOTE', 'INTRODUCTION']):
                        sections.append(section_text)
            
            # Handle incomplete sections
            incomplete_pattern = r'<INCOMPLETE>(.*?)</INCOMPLETE>'
            incomplete_match = re.search(incomplete_pattern, result, re.DOTALL)
            if incomplete_match:
                incomplete_text = incomplete_match.group(1).strip()
                if incomplete_text:
                    self.missed_text = incomplete_text
                    self.log_message(f"Found incomplete section ({self.count_tokens(incomplete_text)} tokens)")
            
            # Verify content preservation
            if not sections:
                self.log_message("Warning: No sections created by LLM")
                self.missed_text = chunk
            elif not incomplete_match:
                missed_text = self.verify_output_completeness(chunk, sections)
                if missed_text:
                    self.missed_text = missed_text
                    self.log_message(f"Found missed text ({self.count_tokens(missed_text)} tokens)")
            
            # Collect metrics
            metrics = {
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
                "finish_reason": response.choices[0].finish_reason,
                "sections_created": len(sections),
                "has_missed_text": bool(self.missed_text),
                "avg_section_length": sum(len(s) for s in sections) / len(sections) if sections else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            return sections, metrics
                
        except Exception as e:
            self.log_message(f"Error in LLM request: {str(e)}")
            return [], {}

    def verify_output_completeness(self, input_text: str, output_sections: List[str]) -> str:
        """
        Enhanced verification of content preservation with improved detection.
        """
        # Normalize texts for comparison
        input_normalized = ' '.join(input_text.split())
        output_normalized = ' '.join(' '.join(output_sections).split())
        
        # Quick full-text comparison
        if input_normalized == output_normalized:
            return ""
        
        # Find missing content using sliding window
        words = input_normalized.split()
        missing_sequences = set()  # Use set to avoid duplicates
        
        # Use multiple window sizes for better detection
        for window_size in [5, 10, 15]:  # Try different window sizes
            i = 0
            while i < len(words) - window_size:
                sequence = ' '.join(words[i:i+window_size])
                if sequence not in output_normalized:
                    # Find complete missing phrase
                    start = i
                    while start > 0 and ' '.join(words[start-1:i+window_size]) not in output_normalized:
                        start -= 1
                    end = i + window_size
                    while end < len(words) and ' '.join(words[i:end+1]) not in output_normalized:
                        end += 1
                    missing_sequences.add(' '.join(words[start:end]))
                    i = end
                else:
                    i += 1
        
        return '\n'.join(sorted(missing_sequences)) if missing_sequences else ""

    def validate_chunk(self, chunk: str, original_sections: List[Section]) -> bool:
        """
        Comprehensive chunk validation with detailed reporting.
        """
        # Normalize texts for comparison
        chunk_text = ' '.join(chunk.split())
        
        # Track missing content by section type
        missing_by_type = {}
        
        for section in original_sections:
            section_text = ' '.join(section.text.split())
            
            # For headers and front matter, require exact matches
            if section.type in [SectionType.HEADER, SectionType.FRONT_MATTER]:
                if section_text not in chunk_text:
                    missing_by_type.setdefault(section.type, []).append(section.text)
                continue
            
            # For other content, use sliding window detection
            words = section_text.split()
            window_size = 5
            missing_chunks = set()
            
            i = 0
            while i < len(words) - window_size:
                sequence = ' '.join(words[i:i+window_size])
                if sequence not in chunk_text:
                    # Find complete phrase
                    start = i
                    while start > 0 and ' '.join(words[start-1:i+window_size]) not in chunk_text:
                        start -= 1
                    end = i + window_size
                    while end < len(words) and ' '.join(words[i:end+1]) not in chunk_text:
                        end += 1
                    missing_chunks.add(' '.join(words[start:end]))
                    i = end
                else:
                    i += 1
            
            if missing_chunks:
                missing_by_type.setdefault(section.type, []).extend(missing_chunks)
        
        # Report missing content by type
        if missing_by_type:
            self.log_message("Missing content detected:")
            for section_type, missing_content in missing_by_type.items():
                self.log_message(f"\n{section_type.value}:")
                for content in missing_content:
                    self.log_message(f"  - {content[:100]}...")
            return False
        
        return True

    def save_chunk_log(self, chunk_num: int, original_chunk: str, sections: List[str], metrics: Dict):
        """
        Enhanced logging with more detailed analytics.
        """
        log_file = os.path.join(self.log_dir, f"chunk_{chunk_num:04d}.json")
        
        # Add detailed token analysis
        section_analytics = [{
            "length": len(section),
            "tokens": self.count_tokens(section),
            "lines": len(section.split('\n')),
            "paragraphs": len(section.split('\n\n')),
            "preview": section[:200]
        } for section in sections]
        
        log_data = {
            "chunk_number": chunk_num,
            "timestamp": datetime.now().isoformat(),
            "original_text": {
                "content": original_chunk,
                "length": len(original_chunk),
                "tokens": self.count_tokens(original_chunk)
            },
            "sections": {
                "count": len(sections),
                "analytics": section_analytics,
                "content": sections
            },
            "missed_text": {
                "content": self.missed_text,
                "length": len(self.missed_text) if self.missed_text else 0,
                "tokens": self.count_tokens(self.missed_text) if self.missed_text else 0
            },
            "metrics": metrics,
            "validation": {
                "all_content_preserved": not bool(self.missed_text),
                "total_output_tokens": sum(self.count_tokens(s) for s in sections)
            }
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
            
    def process_text(self, text: str, max_chunks: int = None) -> List[str]:
        """
        Main text processing pipeline with enhanced error handling and logging.
        """
        self.log_message("Starting text processing")
        
        try:
            # Create initial chunks
            initial_chunks = self.create_initial_chunks(text)
            
            if max_chunks:
                initial_chunks = initial_chunks[:max_chunks]
                self.log_message(f"Processing limited to first {max_chunks} chunks")
            
            # Process each chunk
            semantic_chunks = []
            for i, chunk in enumerate(initial_chunks):
                self.log_message(f"\nProcessing chunk {i+1}/{len(initial_chunks)}")
                
                # Detailed chunk analysis
                chunk_tokens = self.count_tokens(chunk)
                self.log_message(f"Chunk size: {len(chunk)} chars, {chunk_tokens} tokens")
                
                # Print input preview
                self.print_separator("INPUT CHUNK")
                print(f"Chunk {i+1}:")
                print("="*80)
                print(chunk[:1000] + "..." if len(chunk) > 1000 else chunk)
                print("="*80)
                
                # Get semantic sections
                sections, metrics = self.get_semantic_sections(chunk)
                
                # Process and validate sections
                self.print_separator("SEMANTIC SECTIONS")
                for j, section in enumerate(sections):
                    section_tokens = self.count_tokens(section)
                    print(f"\nSection {j+1} ({section_tokens} tokens):")
                    print("-"*40)
                    print(section[:500] + "..." if len(section) > 500 else section)
                    print("-"*40)
                
                # Print metrics
                self.print_separator("PROCESSING METRICS")
                pprint(metrics)
                
                if self.missed_text:
                    self.print_separator("MISSED CONTENT")
                    print(self.missed_text)
                
                semantic_chunks.extend(sections)
                
                # Save detailed processing log
                self.save_chunk_log(i+1, chunk, sections, metrics)
                
                time.sleep(1)  # Rate limiting
            
            self.log_message(f"Processing complete. Created {len(semantic_chunks)} semantic chunks")
            return semantic_chunks
            
        except Exception as e:
            self.log_message(f"Error in text processing: {str(e)}")
            raise

def main():
    """
    Main execution function with enhanced error handling and reporting.
    """
    try:
        # Initialize chunker
        chunker = SemanticChunker()
        
        # Read input file
        input_file = "/home/ubuntu/quantumLeap/data/input/Step_2_Classic_Texts_and_Ethnographies/2.1_Public_Domain_Books/Project_Gutenberg/psychology_of_unconscious.txt"
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunker.log_message(f"Starting processing of {input_file}")
        chunker.log_message(f"Input text: {len(text)} chars, {chunker.count_tokens(text)} tokens")
        
        # Process text with limit for testing
        semantic_chunks = chunker.process_text(text, max_chunks=3)
        
        # Save final chunks
        output_dir = os.path.join(chunker.log_dir, "semantic_chunks")
        os.makedirs(output_dir, exist_ok=True)
        
        for i, chunk in enumerate(semantic_chunks):
            output_file = os.path.join(output_dir, f"semantic_chunk_{i+1:04d}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
        
        chunker.log_message(f"Saved {len(semantic_chunks)} semantic chunks to {output_dir}")
        
        # Generate processing summary
        summary_file = os.path.join(chunker.log_dir, "processing_summary.json")
        summary = {
            "timestamp": datetime.now().isoformat(),
            "input_file": input_file,
            "input_stats": {
                "chars": len(text),
                "tokens": chunker.count_tokens(text)
            },
            "output_stats": {
                "total_chunks": len(semantic_chunks),
                "total_tokens": sum(chunker.count_tokens(c) for c in semantic_chunks),
                "avg_chunk_size": sum(len(c) for c in semantic_chunks) / len(semantic_chunks)
            },
            "output_directory": output_dir
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

def test_structure_analysis():
    """
    Enhanced test function with detailed validation and reporting.
    """
    chunker = SemanticChunker()
    
    test_text = """
                             AUTHOR'S NOTE

My task in this work has been to investigate an individual phantasy
system, and in the doing of it problems of such magnitude have been
uncovered, that my endeavor to grasp them in their entirety has
necessarily meant only a superficial orientation toward those paths, the
opening and exploration of which may possibly crown the work of future
investigators with success.

                                CONTENTS

        INTRODUCTION                                                     3
        
        Relation of the Incest Phantasy to the Oedipus Legend—Moral
        revulsion over such a discovery

 I.—    CONCERNING THE TWO KINDS OF THINKING                             8
"""
    
    try:
        print("\nTesting structural analysis...")
        sections = chunker.analyze_text_structure(test_text)
        
        print("\nIdentified sections:")
        for i, section in enumerate(sections, 1):
            print(f"\nSection {i}:")
            print("="*80)
            print(f"Type: {section.type}")
            print(f"Level: {section.level}")
            print(f"Length: {len(section.text)} chars, {chunker.count_tokens(section.text)} tokens")
            print("-"*40)
            print(section.text)
            print("="*80)
        
        print("\nTesting chunking with structure preservation...")
        chunks = chunker.create_initial_chunks(test_text)
        
        print("\nResulting chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i}:")
            print("="*80)
            print(chunk)
            print("="*80)
            
        print("\nValidating chunk content...")
        for i, chunk in enumerate(chunks, 1):
            print(f"\nValidating chunk {i}:")
            is_valid = chunker.validate_chunk(chunk, sections)
            print(f"Chunk {i} validation: {'PASSED' if is_valid else 'FAILED'}")
            
        # Generate test summary
        test_summary = {
            "timestamp": datetime.now().isoformat(),
            "sections_identified": len(sections),
            "chunks_created": len(chunks),
            "section_types": {str(s.type): sum(1 for sec in sections if sec.type == s.type) for s in sections},
            "validation_results": [chunker.validate_chunk(c, sections) for c in chunks]
        }
        
        print("\nTest Summary:")
        pprint(test_summary)
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    if os.environ.get("SEMANTIC_CHUNKER_TEST"):
        test_structure_analysis()
    else:
        main()