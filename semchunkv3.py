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
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        """Initialize the semantic chunker with model configuration"""
        self.client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="dummy"
        )
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.max_tokens = 3000
        self.processed_sections = set()
        
        # Set up logging directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"/home/ubuntu/quantumLeap/data/preprocess/Step_2_Classic_Texts_and_Ethnographies/2.1_Public_Domain_Books/Project_Gutenberg/Psychology_Of_Unconscious_Mind/chunks/chunks_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logging file for processing summary
        self.log_file = os.path.join(self.log_dir, "processing_log.txt")
        
        # Initialize state variables
        self.missed_text = ""  # Store text not included in LLM output
        self.section_hashes = set()
        self.redundant_sections = []
        self.unprocessed_sections = []
        
    def is_redundant(self, section: str) -> bool:
        """
        Enhanced redundancy detection using content hashing
        """
        # Normalize text for comparison
        normalized = ' '.join(section.split())
        content_hash = hash(normalized)
        
        if content_hash in self.section_hashes:
            return True
        
        # Check for near-duplicates (>90% similarity)
        for existing_section in self.processed_sections:
            if self.calculate_similarity(normalized, ' '.join(existing_section.split())) > 0.9:
                return True
        
        self.section_hashes.add(content_hash)
        self.processed_sections.add(section)
        return False
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity ratio between two texts
        """
        words1 = set(text1.split())
        words2 = set(text2.split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)
    
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
    
    def find_chapter_breaks(self, text: str) -> List[int]:
        """Find indices where chapters begin (centered headings)"""
        lines = text.split('\n')
        chapter_breaks = []
        
        for i, line in enumerate(lines):
            if self.is_chapter_heading(line):
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
            nonlocal current_section, current_type
            if current_section:
                sections.append(Section(
                    text='\n'.join(current_section),
                    type=current_type or SectionType.CONTENT,
                    level=current_level
                ))
                current_section = []
                current_type = None
        
        in_toc = False
        in_front_matter = False
        
        for i, line in enumerate(lines):
            # Detect Table of Contents
            if re.match(r'^\s*CONTENTS\s*$', line, re.I):
                flush_section()
                in_toc = True
                current_type = SectionType.TABLE_OF_CONTENTS
                continue
                
            # Detect Front Matter
            if re.match(r'^\s*AUTHOR\'S\s+NOTE\s*$', line, re.I):
                flush_section()
                in_front_matter = True
                current_type = SectionType.FRONT_MATTER
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
                
            # Detect attributions
            if re.match(r'^\s*—\s*[A-Z]', line):
                flush_section()
                current_type = SectionType.ATTRIBUTION
                
            # Detect lists
            if re.match(r'^\s{4,}(?:[\w\-]+\.|\-|\*)\s', line):
                if current_type != SectionType.LIST:
                    flush_section()
                    current_type = SectionType.LIST
                    
            current_section.append(line)
            
            # Handle section transitions
            if in_toc and not line.strip() and i < len(lines)-1 and lines[i+1].strip():
                in_toc = False
                flush_section()
                
            if in_front_matter and not line.strip() and i < len(lines)-1 and lines[i+1].strip():
                in_front_matter = False
                flush_section()
        
        flush_section()  # Flush any remaining content
        return sections
    
    
    def get_complete_paragraphs(self, text: str, max_tokens: int) -> Tuple[str, str]:
        """
        Enhanced version that respects document structure with optimization.
        """
        # Add debug logging
        self.log_message(f"Starting get_complete_paragraphs with {len(text)} chars of text")
        
        # First, analyze the structure
        sections = self.analyze_text_structure(text)
        self.log_message(f"Found {len(sections)} sections")
        
        current_sections = []
        current_tokens = 0
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
            nonlocal current_section, current_type
            if current_section:
                sections.append(Section(
                    text='\n'.join(current_section),
                    type=current_type or SectionType.CONTENT,
                    level=current_level
                ))
                current_section = []
                current_type = None
        
        in_toc = False
        in_front_matter = False
        
        for i, line in enumerate(lines):
            # Detect Table of Contents
            if re.match(r'^\s*CONTENTS\s*$', line, re.I):
                flush_section()
                in_toc = True
                current_type = SectionType.TABLE_OF_CONTENTS
                continue
                
            # Detect Front Matter
            if re.match(r'^\s*AUTHOR\'S\s+NOTE\s*$', line, re.I):
                flush_section()
                in_front_matter = True
                current_type = SectionType.FRONT_MATTER
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
                
            # Detect attributions
            if re.match(r'^\s*—\s*[A-Z]', line):
                flush_section()
                current_type = SectionType.ATTRIBUTION
                
            # Detect lists
            if re.match(r'^\s{4,}(?:[\w\-]+\.|\-|\*)\s', line):
                if current_type != SectionType.LIST:
                    flush_section()
                    current_type = SectionType.LIST
                    
            current_section.append(line)
            
            # Handle section transitions
            if in_toc and not line.strip() and i < len(lines)-1 and lines[i+1].strip():
                in_toc = False
                flush_section()
                
            if in_front_matter and not line.strip() and i < len(lines)-1 and lines[i+1].strip():
                in_front_matter = False
                flush_section()
        
        flush_section()  # Flush any remaining content
        return sections
    
    
    def get_complete_paragraphs(self, text: str, max_tokens: int) -> Tuple[str, str]:
        """
        Enhanced version that respects document structure with optimization.
        """
        # Add debug logging
        self.log_message(f"Starting get_complete_paragraphs with {len(text)} chars of text")
        
        # First, analyze the structure
        sections = self.analyze_text_structure(text)
        self.log_message(f"Found {len(sections)} sections")
        
        current_sections = []
        current_tokens = 0
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
            nonlocal current_section, current_type
            if current_section:
                sections.append(Section(
                    text='\n'.join(current_section),
                    type=current_type or SectionType.CONTENT,
                    level=current_level
                ))
                current_section = []
                current_type = None
        
        in_toc = False
        in_front_matter = False
        
        for i, line in enumerate(lines):
            # Detect Table of Contents
            if re.match(r'^\s*CONTENTS\s*$', line, re.I):
                flush_section()
                in_toc = True
                current_type = SectionType.TABLE_OF_CONTENTS
                continue
                
            # Detect Front Matter
            if re.match(r'^\s*AUTHOR\'S\s+NOTE\s*$', line, re.I):
                flush_section()
                in_front_matter = True
                current_type = SectionType.FRONT_MATTER
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
                
            # Detect attributions
            if re.match(r'^\s*—\s*[A-Z]', line):
                flush_section()
                current_type = SectionType.ATTRIBUTION
                
            # Detect lists
            if re.match(r'^\s{4,}(?:[\w\-]+\.|\-|\*)\s', line):
                if current_type != SectionType.LIST:
                    flush_section()
                    current_type = SectionType.LIST
                    
            current_section.append(line)
            
            # Handle section transitions
            if in_toc and not line.strip() and i < len(lines)-1 and lines[i+1].strip():
                in_toc = False
                flush_section()
                
            if in_front_matter and not line.strip() and i < len(lines)-1 and lines[i+1].strip():
                in_front_matter = False
                flush_section()
        
        flush_section()  # Flush any remaining content
        return sections
    
    
    def get_complete_paragraphs(self, text: str, max_tokens: int) -> Tuple[str, str]:
        """
        Enhanced version that respects document structure with optimization.
        """
        # Add debug logging
        self.log_message(f"Starting get_complete_paragraphs with {len(text)} chars of text")
        
        # First, analyze the structure
        sections = self.analyze_text_structure(text)
        self.log_message(f"Found {len(sections)} sections")
        
        current_sections = []
        current_tokens = 0
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
            nonlocal current_section, current_type
            if current_section:
                sections.append(Section(
                    text='\n'.join(current_section),
                    type=current_type or SectionType.CONTENT,
                    level=current_level
                ))
                current_section = []
                current_type = None
        
        in_toc = False
        in_front_matter = False
        
        for i, line in enumerate(lines):
            # Detect Table of Contents
            if re.match(r'^\s*CONTENTS\s*$', line, re.I):
                flush_section()
                in_toc = True
                current_type = SectionType.TABLE_OF_CONTENTS
                continue
                
            # Detect Front Matter
            if re.match(r'^\s*AUTHOR\'S\s+NOTE\s*$', line, re.I):
                flush_section()
                in_front_matter = True
                current_type = SectionType.FRONT_MATTER
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
                
            # Detect attributions
            if re.match(r'^\s*—\s*[A-Z]', line):
                flush_section()
                current_type = SectionType.ATTRIBUTION
                
            # Detect lists
            if re.match(r'^\s{4,}(?:[\w\-]+\.|\-|\*)\s', line):
                if current_type != SectionType.LIST:
                    flush_section()
                    current_type = SectionType.LIST
                    
            current_section.append(line)
            
            # Handle section transitions
            if in_toc and not line.strip() and i < len(lines)-1 and lines[i+1].strip():
                in_toc = False
                flush_section()
                
            if in_front_matter and not line.strip() and i < len(lines)-1 and lines[i+1].strip():
                in_front_matter = False
                flush_section()
        
        flush_section()  # Flush any remaining content
        return sections
    
    def analyze_text_structure(self, text: str) -> List[Section]:
        """
        Enhanced text structure analysis with better header and spacing detection.
        """
        sections = []
        lines = text.split('\n')
        current_section = []
        current_type = None
        current_level = 0
        
        def flush_section():
            nonlocal current_section, current_type
            if current_section:
                # Skip empty sections
                content = '\n'.join(current_section).strip()
                if content:  # Only create section if there's actual content
                    sections.append(Section(
                        text='\n'.join(current_section),
                        type=current_type or SectionType.CONTENT,
                        level=current_level
                    ))
                current_section = []
                current_type = None
        
        in_toc = False
        in_front_matter = False
        
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
                    if not next_line.strip():  # Include following blank line
                        current_section.append(next_line)
                        i += 1
                    flush_section()
                    i += 1
                    continue
            
            # Detect Table of Contents
            if re.match(r'^\s*CONTENTS\s*$', line, re.I):
                flush_section()
                in_toc = True
                current_type = SectionType.TABLE_OF_CONTENTS
                current_section = [line]
                if not next_line.strip():  # Include following blank line
                    current_section.append(next_line)
                    i += 1
                i += 1
                continue
            
            # Detect Author's Note
            if re.match(r'^\s*AUTHOR\'S\s+NOTE\s*$', line, re.I):
                flush_section()
                in_front_matter = True
                current_type = SectionType.FRONT_MATTER
                current_section = [line]
                if not next_line.strip():  # Include following blank line
                    current_section.append(next_line)
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
                if not next_line.strip():  # Include following blank line
                    current_section.append(next_line)
                    i += 1
                flush_section()
                i += 1
                continue
            
            # Handle section content
            if in_toc:
                if not line.strip() and not next_line.strip():
                    in_toc = False
                    flush_section()
                else:
                    current_section.append(line)
            elif in_front_matter:
                if not line.strip() and not next_line.strip():
                    in_front_matter = False
                    flush_section()
                else:
                    current_section.append(line)
            else:
                current_section.append(line)
            
            i += 1
        
        flush_section()  # Flush any remaining content
        
        # Filter out empty sections and preserve correct spacing
        filtered_sections = []
        for section in sections:
            if section.text.strip():
                filtered_sections.append(section)
        
        return filtered_sections

    def get_complete_paragraphs(self, text: str, max_tokens: int) -> Tuple[str, str]:
        """
        Enhanced version with corrected content processing logic.
        """
        self.log_message(f"Starting get_complete_paragraphs with {len(text)} chars of text")
        
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
                
                # If this section would exceed our token limit
                if current_tokens + section_tokens > max_tokens:
                    if current_sections:  # Only break if we have content
                        break
                
                # Always include header with its following content
                if section.type == SectionType.HEADER:
                    # Add the header
                    current_sections.append(section)
                    current_tokens += section_tokens
                    
                    # Look ahead for content
                    next_index = section_index + 1
                    if next_index < len(sections) and sections[next_index].type == SectionType.CONTENT:
                        next_section = sections[next_index]
                        next_tokens = self.count_tokens(next_section.text)
                        if current_tokens + next_tokens <= max_tokens:
                            current_sections.append(next_section)
                            current_tokens += next_tokens
                            section_index += 1  # Skip the content section in next iteration
                    
                # Handle content sections not attached to headers
                elif section.type == SectionType.CONTENT:
                    current_sections.append(section)
                    current_tokens += section_tokens
                
                # Handle other section types (TABLE_OF_CONTENTS, etc.)
                else:
                    current_sections.append(section)
                    current_tokens += section_tokens
                
                section_index += 1
                self.log_message(f"After processing: current_tokens={current_tokens}, max_tokens={max_tokens}, sections_processed={len(current_sections)}")
            
            # Combine sections with proper spacing
            processed_sections = []
            for i, section in enumerate(current_sections):
                # Add extra newline before sections (except the first one)
                if i > 0:
                    processed_sections.append("")
                
                # Add the section text
                processed_sections.append(section.text.rstrip())
                
                # Add extra newline after headers
                if section.type == SectionType.HEADER:
                    processed_sections.append("")
            
            processed_text = "\n".join(processed_sections)
            
            # Prepare remaining sections
            remaining_sections = []
            if section_index < len(sections):
                for section in sections[section_index:]:
                    if remaining_sections:
                        remaining_sections.append("")
                    remaining_sections.append(section.text.rstrip())
            
            remaining_text = "\n".join(remaining_sections) if remaining_sections else ""
            
            self.log_message(f"Completed processing: {len(current_sections)} sections included, {len(sections) - section_index} remaining")
            self.log_message(f"Processed text preview: {processed_text[:200]}...")
            
            return processed_text, remaining_text
            
        except Exception as e:
            self.log_message(f"Error in get_complete_paragraphs: {str(e)}")
            if current_sections:
                return "\n".join([s.text for s in current_sections]), text
            return "", text

    def process_text(self, text: str, max_chunks: int = None) -> List[str]:
        """Process entire text into semantic sections with enhanced logging"""
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
            print("Content preview:")
            print(chunk[:1000] + "..." if len(chunk) > 1000 else chunk)
            
            self.print_separator("SEMANTIC SECTIONS")
            for j, section in enumerate(sections):
                print(f"\nSection {j+1} (Tokens: {self.count_tokens(section)})")
                print("Content preview:")
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
                while end < len(words) and ' '.join(words[i:end+1]) not in output_normalized:
                    end += 1
                missing_sequences.append(' '.join(words[start:end]))
                i = end
            else:
                i += 1
        
        return '\n'.join(missing_sequences) if missing_sequences else ""
    
    def create_initial_chunks(self, text: str) -> List[str]:
        """
        Create initial chunks with enhanced logging and ensure that no significant portions are dropped.
        """
        chunks = []
        remaining_text = text
        chunk_number = 0

        while remaining_text.strip():
            chunk_number += 1
            self.log_message(f"\nProcessing chunk {chunk_number}")

            # Add any missed text from previous chunk
            if self.missed_text:
                self.log_message("Adding missed text from previous chunk")
                remaining_text = self.missed_text + '\n\n' + remaining_text
                self.missed_text = ""

            # Get complete paragraphs up to token limit
            chunk_text, remaining_text = self.get_complete_paragraphs(remaining_text, self.max_tokens)

            if chunk_text.strip():
                self.log_message(f"Created chunk {chunk_number} with {self.count_tokens(chunk_text)} tokens")
                chunks.append(chunk_text)

                # Debug output
                preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                self.log_message(f"Chunk {chunk_number} preview:\n{preview}")
            else:
                self.log_message("Warning: Empty chunk produced")
                if remaining_text.strip():
                    # If there's remaining text but the chunk is empty, save the unprocessed text
                    self.log_unprocessed_text(remaining_text)
                    break  # Exit the loop as we cannot create further chunks
                else:
                    break

            if len(chunks) >= 100:  # Safety limit
                self.log_message("Warning: Maximum chunk limit reached")
                break

        self.log_message(f"Created {len(chunks)} initial chunks")

        # Save the chunks
        os.makedirs(self.log_dir, exist_ok=True)
        for i, chunk in enumerate(chunks):
            with open(os.path.join(self.log_dir, f"chunk_{i+1:04d}.txt"), 'w', encoding='utf-8') as f:
                f.write(chunk)

        return chunks
    
    def get_semantic_sections(self, chunk: str) -> Tuple[List[str], Dict]:
        """Update the system prompt for better structural preservation and implement 
        redundancy detection."""
        try:
            self.log_message(f"Sending request to LLM (input tokens: {self.count_tokens(chunk)})")
            
            # Initialize metrics dictionary first
            metrics = {
                "completion_tokens": 0,
                "prompt_tokens": 0, 
                "total_tokens": 0,
                "finish_reason": None,
                "sections_created": 0,
                "has_missed_text": False,
                "redundant_sections": 0,
                "unprocessed_sections": 0,
                "total_processed_sections": 0
            }
            
            response = self.client.chat.completions.create(
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
            
            processed_sections = []
            for section in sections:
                if self.is_redundant(section):
                    self.redundant_sections.append(section)
                    self.log_redundant_content(section)
                else:
                    processed_sections.append(section)
            
            # Verify all content is accounted for
            all_processed_content = ' '.join(processed_sections)
            unprocessed = self.verify_output_completeness(chunk, processed_sections)
            
            if unprocessed:
                self.unprocessed_sections.append(unprocessed)
                self.log_unprocessed_text(unprocessed)
                
            # If we have missed text but no explicit incomplete sections
            if not incomplete_match and unprocessed:
                self.missed_text = unprocessed
            
            # Now update metrics safely
            metrics.update({
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
                "finish_reason": response.choices[0].finish_reason,
                "sections_created": len(sections),
                "has_missed_text": bool(self.missed_text),
                "redundant_sections": len(self.redundant_sections),
                "unprocessed_sections": len(self.unprocessed_sections),
                "total_processed_sections": len(processed_sections)
            })

            return sections, metrics

        except Exception as e:
            self.log_message(f"Error in semantic sections: {str(e)}")
            return [], {}
    def validate_semantic_sections(self, chunk: str, sections: List[str]) -> bool:
        """Validate that all content from chunk is preserved in sections"""
        # Normalize for comparison
        chunk_text = ' '.join(chunk.split())
        sections_text = ' '.join(' '.join(sections).split())
        
        # Quick length check first
        if len(sections_text) < (len(chunk_text) * 0.9):  # Allow 10% variation for formatting
            self.log_message("WARNING: Significant content loss detected")
            return False
            
        # Check for key phrases preservation
        key_phrases = self.extract_key_phrases(chunk)
        for phrase in key_phrases:
            if phrase not in sections_text:
                self.log_message(f"Missing key phrase: {phrase}")
                return False
                
        return True

    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract important phrases for validation"""
        # Split into sentences and get important ones (e.g. first/last of paragraphs)
        sentences = text.split('.')
        if len(sentences) < 3:
            return sentences
            
        return [
            sentences[0],  # First sentence
            sentences[-1],  # Last sentence
            sentences[len(sentences)//2]  # Middle sentence
        ]

    def save_semantic_sections(self, sections: List[str], chunk_num: int):
        """Save semantic sections with validation"""
        output_dir = os.path.join(self.log_dir, "semantic_chunks")
        os.makedirs(output_dir, exist_ok=True)
        
        for i, section in enumerate(sections):
            # Validate section isn't empty or corrupted
            if not section.strip():
                continue
                
            output_file = os.path.join(output_dir, f"semantic_chunk_{chunk_num}_{i+1:04d}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(section)
                
        self.log_message(f"Saved {len(sections)} valid sections from chunk {chunk_num}")
        
    def save_final_report(self):
        """
        Generate comprehensive final report
        """
        report = {
            "processing_summary": {
                "total_chunks": len(self.chunk_boundaries),
                "total_sections": len(self.processed_sections),
                "redundant_sections": len(self.redundant_sections),
                "unprocessed_sections": len(self.unprocessed_sections)
            },
            "content_tracking": {
                "missed_text_instances": [
                    {"text": text, "tokens": self.count_tokens(text)}
                    for text in self.unprocessed_sections
                ],
                "redundant_content": [
                    {"text": text, "tokens": self.count_tokens(text)}
                    for text in self.redundant_sections
                ]
            }
        }
        
        report_file = os.path.join(self.log_dir, "final_processing_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

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
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

    def validate_chunk(self, chunk: str, original_sections: List[Section]) -> bool:
        """Validate that chunk contains all expected content"""
        # Normalize texts for comparison
        chunk_text = ' '.join(chunk.split())
        original_text = ' '.join(' '.join(s.text for s in original_sections).split())
        
        # Check if all content is present
        missing_content = []
        words = original_text.split()
        window_size = 5
        
        i = 0
        while i < len(words) - window_size:
            sequence = ' '.join(words[i:i+window_size])
            if sequence not in chunk_text:
                missing_content.append(sequence)
            i += 1
        
        if missing_content:
            self.log_message("Missing content detected:")
            for mc in missing_content:
                self.log_message(f"  - {mc}")
            return False
        
        return True
    
    def log_unprocessed_text(self, text: str):
        """
        Log and save unprocessed or redundant text in a separate file.
        """
        unprocessed_file = os.path.join(self.log_dir, "unprocessed_texts.txt")
        with open(unprocessed_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n--- Unprocessed Text ---\n")
            f.write(text)
        self.log_message(f"Unprocessed text saved to {unprocessed_file}")
        
    def is_redundant(self, section: str) -> bool:
        """
        Determine if a section is redundant.
        Implement your redundancy detection logic here.
        """
        # Placeholder logic for redundancy detection
        # For example, check if the section is identical to previous sections
        if section in self.processed_sections:
            return True
        else:
            self.processed_sections.add(section)
            return False

    def log_redundant_content(self, content: str):
        """
        Log and save redundant content in a separate file.
        """
        redundant_file = os.path.join(self.log_dir, "redundant_texts.txt")
        with open(redundant_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n--- Redundant Text ---\n")
            f.write(content)
        self.log_message(f"Redundant content saved to {redundant_file}")

        
    
def main():
    # Initialize chunker
    chunker = SemanticChunker()
    
    # Read input file
    input_file = "/home/ubuntu/quantumLeap/data/input/Step_2_Classic_Texts_and_Ethnographies/2.1_Public_Domain_Books/Project_Gutenberg/psychology_of_unconscious.txt"
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Process text (limit to first 5 chunks for testing)
    semantic_chunks = chunker.process_text(text, max_chunks=10)
    
    # Save final chunks
    output_dir = os.path.join(chunker.log_dir, "semantic_chunks")
    os.makedirs(output_dir, exist_ok=True)
    
    for i, chunk in enumerate(semantic_chunks):
        output_file = os.path.join(output_dir, f"semantic_chunk_{i+1:04d}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(chunk)
    
    chunker.log_message(f"Saved {len(semantic_chunks)} semantic chunks to {output_dir}")

def test_structure_analysis():
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
        print("Testing structural analysis...")
        sections = chunker.analyze_text_structure(test_text)
        
        print("\nIdentified sections:")
        for i, section in enumerate(sections, 1):
            print(f"\nSection {i}:")
            print(f"Type: {section.type}")
            print(f"Level: {section.level}")
            print(f"Content preview: {section.text[:100]}...")
        
        print("\nTesting chunking with structure preservation...")
        chunks = chunker.create_initial_chunks(test_text)
        
        print("\nResulting chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i}:")
            print(chunk[:200])
            print("...")
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
    
    # Add validation
    print("\nValidating chunk content...")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nValidating chunk {i}:")
        is_valid = chunker.validate_chunk(chunk, sections)
        print(f"Chunk {i} validation: {'PASSED' if is_valid else 'FAILED'}")
        
if __name__ == "__main__":
    # test_structure_analysis()
    main()  # Comment out for testing