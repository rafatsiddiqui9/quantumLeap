import tiktoken
import openai
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

# Add these new data structures after imports
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


class SemanticChunker:
    def __init__(self, model_name: str = "gpt-3.5-turbo", max_tokens: int = 3000):
        """Initialize the semantic chunker with model configuration"""
        # Initialize OpenAI client with proper configuration
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        
        openai.api_key = openai_api_key
        
        self.model_name = model_name
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as e:
            print(f"Error initializing tiktoken: {e}")
            self.encoding = tiktoken.get_encoding("gpt-3.5-turbo")
        
        self.max_tokens = max_tokens
        
        # Set up logging directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"./logs/chunks_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logging file for processing summary
        self.log_file = os.path.join(self.log_dir, "processing_log.txt")
        
        # Initialize state variables
        self.missed_text = ""  # Store text not included in LLM output
        
        self.log_message("SemanticChunker initialized.")

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
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            self.log_message(f"Error counting tokens: {e}")
            return 0
    
    def find_chapter_breaks(self, text: str) -> List[int]:
        """Find indices where chapters begin (centered headings)"""
        lines = text.split('\n')
        chapter_breaks = []
        
        for i, line in enumerate(lines):
            if self.is_chapter_heading(line)[0]:
                chapter_breaks.append(i)
        
        return chapter_breaks
    
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
        ]
        
        for pattern, level in chapter_patterns:
            if re.match(pattern, text, re.I):
                return True, level
        
        # Check for centered text formatting
        line_length = len(text)
        leading_spaces = len(text) - len(text.lstrip())
        trailing_spaces = len(text) - len(text.rstrip())
        
        is_centered = abs(leading_spaces - trailing_spaces) <= 2 and leading_spaces > 5
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
        Analyze text structure and break it into typed sections.
        """
        sections = []
        lines = text.split('\n')
        current_section = []
        current_type = None
        current_level = 0
        
        def flush_section():
            nonlocal current_section, current_type, current_level
            if current_section:
                sections.append(Section(
                    text='\n'.join(current_section),
                    type=current_type or SectionType.CONTENT,
                    level=current_level
                ))
                current_section = []
                current_type = None
                current_level = 0
        
        in_toc = False
        in_front_matter = False
        
        for i, line in enumerate(lines):
            # Detect Table of Contents
            if re.match(r'^\s*CONTENTS\s*$', line, re.I):
                flush_section()
                in_toc = True
                current_type = SectionType.TABLE_OF_CONTENTS
                current_section.append(line)
                continue
                
            # Detect Front Matter
            if re.match(r'^\s*AUTHOR\'S\s+NOTE\s*$', line, re.I):
                flush_section()
                in_front_matter = True
                current_type = SectionType.FRONT_MATTER
                current_section.append(line)
                continue
                
            # Check for section transitions
            is_heading, level = self.is_chapter_heading(line)
            if is_heading:
                flush_section()
                current_section = [line]
                current_type = SectionType.HEADER
                current_level = level
                continue
                
            # Detect quotes
            if line.startswith('"') and len(line) > 50:
                flush_section()
                current_type = SectionType.QUOTE
                current_section.append(line)
                continue
                
            # Detect attributions
            if re.match(r'^\s*—\s*[A-Z]', line):
                flush_section()
                current_type = SectionType.ATTRIBUTION
                current_section.append(line)
                continue
                
            # Detect lists
            if re.match(r'^\s{4,}(?:[\w\-]+\.|\-|\*)\s', line):
                if current_type != SectionType.LIST:
                    flush_section()
                    current_type = SectionType.LIST
                current_section.append(line)
                continue
                
            # Accumulate content
            current_section.append(line)
            
            # Handle section transitions
            if in_toc and not line.strip() and i < len(lines)-1 and lines[i+1].strip():
                in_toc = False
                flush_section()
                
            if in_front_matter and not line.strip() and i < len(lines)-1 and lines[i+1].strip():
                in_front_matter = False
                flush_section()
        
        flush_section()  # Flush any remaining content
        self.log_message(f"Analyzed text structure into {len(sections)} sections.")
        return sections
    
    
    def get_complete_paragraphs(self, text: str, max_tokens: int) -> Tuple[str, str]:
        """
        Enhanced version that respects document structure when splitting text.
        """
        # First, analyze the structure
        sections = self.analyze_text_structure(text)
        
        current_sections = []
        current_tokens = 0
        last_processed_index = -1  # Track the last processed section
        
        for i, section in enumerate(sections):
            section_tokens = self.count_tokens(section.text)
            
            # Always keep headers with their content
            if section.type == SectionType.HEADER:
                if current_sections and (current_tokens + section_tokens) > max_tokens:
                    break
                current_sections.append(section)
                current_tokens += section_tokens
                continue
                
            # Don't split lists or quotes
            if section.type in [SectionType.LIST, SectionType.QUOTE]:
                if current_tokens + section_tokens > max_tokens:
                    break
                current_sections.append(section)
                current_tokens += section_tokens
                continue
                
            # For regular content, split at paragraph boundaries if needed
            if section.type == SectionType.CONTENT:
                paragraphs = section.text.split('\n\n')
                for para in paragraphs:
                    para_tokens = self.count_tokens(para)
                    if current_tokens + para_tokens > max_tokens:
                        break
                    current_sections.append(Section(para, SectionType.CONTENT))
                    current_tokens += para_tokens
                if current_tokens >= max_tokens:
                    break
        
        if current_sections:
            last_processed_index = len(current_sections)
        
        processed_text = '\n\n'.join(s.text for s in current_sections)
        remaining_sections = sections[len(current_sections):]
        remaining_text = '\n\n'.join(s.text for s in remaining_sections)
        
        self.log_message(f"Processed up to section {len(current_sections)}")
        self.log_message(f"Remaining text token count: {self.count_tokens(remaining_text)}")
        
        return processed_text, remaining_text
    
    
    def verify_output_completeness(self, input_text: str, output_sections: List[str]) -> str:
        """Verify all input text is present in output sections and return missing text"""
        # Normalize texts for comparison
        input_normalized = ' '.join(input_text.split())
        output_normalized = ' '.join(' '.join(output_sections).split())
        
        # Find missing content
        words = input_normalized.split()
        window_size = 5  # Look for sequences of 5 words
        
        missing_sequences = []
        i = 0
        while i < len(words) - window_size:
            sequence = ' '.join(words[i:i+window_size])
            if sequence not in output_normalized:
                # Find complete missing phrase
                start = i
                while start > 0 and ' '.join(words[start-1:i+window_size]) not in output_normalized:
                    start -= 1
                end = i + window_size
                while end < len(words) and ' '.join(words[start:end+1]) not in output_normalized:
                    end += 1
                missing_sequences.append(' '.join(words[start:end]))
                i = end
            else:
                i += 1
        
        return '\n'.join(missing_sequences) if missing_sequences else ""
    
    
    def create_initial_chunks(self, text: str) -> List[str]:
        """
        Create initial chunks while preserving document structure.
        """
        chunks = []
        remaining_text = text
        
        self.log_message("Entering create_initial_chunks")
        
        while remaining_text:
            # Add any missed text from previous chunk
            if self.missed_text:
                self.log_message("Adding missed text from previous chunk")
                remaining_text = self.missed_text + '\n\n' + remaining_text
                self.missed_text = ""
            
            # Get complete paragraphs up to token limit
            chunk_text, remaining_text = self.get_complete_paragraphs(remaining_text, self.max_tokens)
            if chunk_text:
                chunks.append(chunk_text)
                self.log_message(f"Created chunk {len(chunks)} with {self.count_tokens(chunk_text)} tokens.")
            else:
                self.log_message("No chunk created in this iteration. Breaking to prevent infinite loop.")
                break  # Prevent infinite loop if no chunk is created
            
            # Safety break to prevent infinite loops in edge cases
            if len(chunks) > 10000:
                self.log_message("Reached maximum number of chunks (10000). Breaking to prevent infinite loop.")
                break
        
        self.log_message(f"Created {len(chunks)} initial chunks.")
        
        # Save the chunks
        os.makedirs(self.log_dir, exist_ok=True)
        for i, chunk in enumerate(chunks):
            with open(os.path.join(self.log_dir, f"chunk_{i+1:04d}.txt"), 'w', encoding='utf-8') as f:
                f.write(chunk)
                
        return chunks
    
    
    def get_semantic_sections(self, chunk: str) -> Tuple[List[str], Dict]:
        """Update the system prompt for better structural preservation."""
        try:
            self.log_message(f"Sending request to LLM (input tokens: {self.count_tokens(chunk)})")
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a text analysis expert. Your task is to:
                        1. Maintain the original document structure (headers, lists, quotes)
                        2. Split the input text into coherent semantic sections
                        3. Each section must respect structural boundaries
                        4. Use <START_SECTION> and <END_SECTION> to mark sections
                        5. Include ALL text from the input - do not skip any content
                        6. Preserve ALL formatting, indentation, and special characters
                        7. If there's a header, keep it with its content
                        8. Keep lists and quotes intact within their sections
                        9. If a section would be incomplete, mark it with <INCOMPLETE> tags
                        
                        Rules:
                        - Preserve ALL text exactly as provided
                        - Maintain original formatting and spacing
                        - Don't add any commentary
                        - Don't modify the text
                        - Keep structural elements together
                        - Respect document hierarchy"""
                    },
                    {
                        "role": "user",
                        "content": f"Split this text into coherent sections, preserving ALL content and structure:\n\n{chunk}"
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0.2
            )
            
            result = response.choices[0].message.content
            
            # Extract sections
            sections = []
            section_pattern = r'<START_SECTION>(.*?)<END_SECTION>'
            for match in re.finditer(section_pattern, result, re.DOTALL):
                section_text = match.group(1).strip()
                if section_text and len(section_text) > 50:  # Ignore empty or very short sections
                    sections.append(section_text)
            
            # Check for incomplete section
            incomplete_pattern = r'<INCOMPLETE>(.*?)</INCOMPLETE>'
            incomplete_match = re.search(incomplete_pattern, result, re.DOTALL)
            if incomplete_match:
                incomplete_text = incomplete_match.group(1).strip()
                if incomplete_text:
                    self.missed_text = incomplete_text
                    self.log_message(f"Found incomplete section ({self.count_tokens(incomplete_text)} tokens)")
            
            # Verify all content is included
            if not incomplete_match:  # Only check if no explicit incomplete section
                missed_text = self.verify_output_completeness(chunk, sections)
                if missed_text:
                    self.missed_text = missed_text
                    self.log_message(f"Found missed text ({self.count_tokens(missed_text)} tokens)")
            
            metrics = {
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
                "finish_reason": response.choices[0].finish_reason,
                "sections_created": len(sections),
                "has_missed_text": bool(self.missed_text)
            }
            
            return sections, metrics
                
        except openai.error.OpenAIError as e:
            self.log_message(f"OpenAI API error: {e}")
            return [], {}
        except Exception as e:
            self.log_message(f"Unexpected error in LLM request: {str(e)}")
            return [], {}
    
    
    def process_text(self, text: str, max_chunks: int = None) -> List[str]:
        """Process entire text into semantic sections"""
        self.log_message("Starting text processing")
        
        # Create initial chunks
        initial_chunks = self.create_initial_chunks(text)
        
        if max_chunks:
            initial_chunks = initial_chunks[:max_chunks]
            self.log_message(f"Processing limited to first {max_chunks} chunks")
        
        # Process each chunk
        semantic_chunks = []
        for i, chunk in enumerate(initial_chunks):
            self.log_message(f"Processing chunk {i+1}/{len(initial_chunks)}")
            
            # Get semantic sections
            sections, metrics = self.get_semantic_sections(chunk)
            
            # Print processing details
            self.print_separator("INPUT CHUNK")
            print(f"Chunk {i+1} (Tokens: {self.count_tokens(chunk)})")
            print(chunk[:1000] + "..." if len(chunk) > 1000 else chunk)
            
            self.print_separator("SEMANTIC SECTIONS")
            for j, section in enumerate(sections):
                print(f"\nSection {j+1} (Tokens: {self.count_tokens(section)})")
                print(section[:500] + "..." if len(section) > 500 else section)
            
            self.print_separator("METRICS")
            pprint(metrics)
            
            if self.missed_text:
                self.print_separator("MISSED TEXT")
                print(self.missed_text)
            
            semantic_chunks.extend(sections)
            
            # Save intermediate results
            self.save_chunk_log(i+1, chunk, sections, metrics)
            
            time.sleep(1)  # Rate limiting
        
        self.log_message(f"Processing complete. Total semantic chunks created: {len(semantic_chunks)}")
        return semantic_chunks

    
    def save_chunk_log(self, chunk_num: int, original_chunk: str, sections: List[str], metrics: Dict):
        """Save intermediate processing results"""
        log_file = os.path.join(self.log_dir, f"chunk_{chunk_num:04d}.json")
        log_data = {
            "chunk_number": chunk_num,
            "original_text": original_chunk,
            "semantic_sections": sections,
            "missed_text": self.missed_text,
            "metrics": metrics,
            "token_counts": {
                "input": self.count_tokens(original_chunk),
                "sections": [self.count_tokens(s) for s in sections],
                "missed": self.count_tokens(self.missed_text) if self.missed_text else 0
            }
        }
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            self.log_message(f"Saved chunk log to {log_file}")
        except Exception as e:
            self.log_message(f"Error saving chunk log: {e}")

def main():
    try:
        # Initialize chunker
        chunker = SemanticChunker()
        
        # Read input file
        input_file = "/home/ubuntu/quantumLeap/data/input/Step_2_Classic_Texts_and_Ethnographies/2.1_Public_Domain_Books/Project_Gutenberg/psychology_of_unconscious.txt"
        if not os.path.exists(input_file):
            chunker.log_message(f"Input file not found: {input_file}")
            return
        
        chunker.log_message(f"Reading input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunker.log_message(f"Input file read successfully. Total tokens: {chunker.count_tokens(text)}")
        
        # Process text (limit to first 3 chunks for testing)
        semantic_chunks = chunker.process_text(text, max_chunks=3)
        
        # Save final chunks
        output_dir = os.path.join(chunker.log_dir, "semantic_chunks")
        os.makedirs(output_dir, exist_ok=True)
        
        for i, chunk in enumerate(semantic_chunks):
            output_file = os.path.join(output_dir, f"semantic_chunk_{i+1:04d}.txt")
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(chunk)
                chunker.log_message(f"Saved semantic chunk {i+1} to {output_file}")
            except Exception as e:
                chunker.log_message(f"Error saving semantic chunk {i+1}: {e}")
        
        chunker.log_message(f"Saved {len(semantic_chunks)} semantic chunks to {output_dir}")
    
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")

if __name__ == "__main__":
    main()
