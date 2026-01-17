"""
Document Chunking Module
========================
Implements intelligent chunking strategies for long documents.
Supports fixed-size, paragraph-aware, and semantic chunking.
"""

import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from loguru import logger


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED = "fixed"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"


@dataclass
class Chunk:
    """Represents a single chunk of text."""
    chunk_id: str
    content: str
    doc_id: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate chunk_id if not provided."""
        if not self.chunk_id:
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.chunk_id = f"{self.doc_id}_chunk_{self.chunk_index}_{content_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata
        }
    
    @property
    def token_estimate(self) -> int:
        """Estimate number of tokens (rough approximation)."""
        # Rough estimate: 1 token ≈ 4 characters for English
        # Adjust for Arabic/other scripts
        return len(self.content) // 4


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    chunk_size: int = 512  # Target chunk size in characters
    chunk_overlap: int = 50  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size
    max_chunk_size: int = 1000  # Maximum chunk size
    respect_sentence_boundaries: bool = True


class DocumentChunker:
    """
    Chunks documents using various strategies.
    
    Strategies:
    - FIXED: Fixed-size chunks with overlap
    - PARAGRAPH: Split on paragraph boundaries
    - SENTENCE: Split on sentence boundaries
    - SEMANTIC: Smart splitting based on content structure
    """
    
    # Sentence boundary patterns (multilingual)
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?؟。])\s+')
    
    # Paragraph boundary
    PARAGRAPH_BREAK = re.compile(r'\n\s*\n')
    
    # Section markers
    SECTION_MARKERS = re.compile(r'^(?:#{1,6}\s|Chapter|Section|Part|\d+\.|[IVX]+\.)', re.MULTILINE)
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        logger.info(f"Initialized DocumentChunker with strategy: {self.config.strategy.value}")
    
    def chunk_document(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk a document using the configured strategy.
        
        Args:
            text: Document text
            doc_id: Document identifier
            metadata: Additional metadata to include in chunks
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        
        # Choose chunking method based on strategy
        if self.config.strategy == ChunkingStrategy.FIXED:
            chunks = self._chunk_fixed(text, doc_id)
        elif self.config.strategy == ChunkingStrategy.PARAGRAPH:
            chunks = self._chunk_paragraph(text, doc_id)
        elif self.config.strategy == ChunkingStrategy.SENTENCE:
            chunks = self._chunk_sentence(text, doc_id)
        elif self.config.strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._chunk_semantic(text, doc_id)
        else:
            chunks = self._chunk_fixed(text, doc_id)
        
        # Add metadata to all chunks
        for chunk in chunks:
            chunk.metadata.update(metadata)
            chunk.metadata["chunk_strategy"] = self.config.strategy.value
            chunk.metadata["total_chunks"] = len(chunks)
        
        logger.debug(f"Created {len(chunks)} chunks for document {doc_id}")
        return chunks
    
    def _chunk_fixed(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Fixed-size chunking with overlap.
        """
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Adjust end to respect sentence boundaries if configured
            if self.config.respect_sentence_boundaries and end < len(text):
                # Look for sentence boundary near the end
                search_start = max(end - 100, start)
                search_end = min(end + 100, len(text))
                search_text = text[search_start:search_end]
                
                matches = list(self.SENTENCE_ENDINGS.finditer(search_text))
                if matches:
                    # Use the closest match to target end
                    best_match = min(matches, key=lambda m: abs((search_start + m.end()) - end))
                    end = search_start + best_match.end()
            
            # Ensure we don't exceed text length
            end = min(end, len(text))
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(Chunk(
                    chunk_id="",
                    content=chunk_text,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end
                ))
                chunk_index += 1
            
            # Move start with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _chunk_paragraph(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Paragraph-based chunking.
        """
        chunks = []
        paragraphs = self.PARAGRAPH_BREAK.split(text)
        
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        position = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                position += 2  # Account for paragraph break
                continue
            
            # Check if adding this paragraph exceeds max chunk size
            potential_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            if len(potential_chunk) > self.config.max_chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append(Chunk(
                    chunk_id="",
                    content=current_chunk.strip(),
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=position
                ))
                chunk_index += 1
                current_chunk = para
                current_start = position
            else:
                current_chunk = potential_chunk
            
            position += len(para) + 2  # +2 for paragraph break
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
            chunks.append(Chunk(
                chunk_id="",
                content=current_chunk.strip(),
                doc_id=doc_id,
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=len(text)
            ))
        
        return chunks
    
    def _chunk_sentence(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Sentence-based chunking.
        """
        chunks = []
        sentences = self.SENTENCE_ENDINGS.split(text)
        
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        position = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > self.config.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(Chunk(
                    chunk_id="",
                    content=current_chunk.strip(),
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=position
                ))
                chunk_index += 1
                current_chunk = sentence
                current_start = position
            else:
                current_chunk = potential_chunk
            
            position += len(sentence) + 1
        
        # Last chunk
        if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
            chunks.append(Chunk(
                chunk_id="",
                content=current_chunk.strip(),
                doc_id=doc_id,
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=len(text)
            ))
        
        return chunks
    
    def _chunk_semantic(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Semantic chunking - smart splitting based on content structure.
        
        This approach:
        1. First tries to split on section markers (headers, etc.)
        2. Then falls back to paragraph boundaries
        3. Finally uses sentence boundaries for fine-tuning
        """
        chunks = []
        
        # First, try to identify sections
        sections = self._identify_sections(text)
        
        if len(sections) > 1:
            # Process each section
            for section_start, section_end, section_text in sections:
                section_chunks = self._chunk_paragraph(section_text, doc_id)
                
                # Adjust positions relative to original text
                for chunk in section_chunks:
                    chunk.start_char += section_start
                    chunk.end_char += section_start
                    chunk.chunk_index = len(chunks)
                    chunks.append(chunk)
        else:
            # No clear sections, use paragraph chunking
            chunks = self._chunk_paragraph(text, doc_id)
        
        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)
        
        # Split large chunks
        chunks = self._split_large_chunks(chunks, doc_id)
        
        # Reindex
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
        
        return chunks
    
    def _identify_sections(self, text: str) -> List[tuple]:
        """
        Identify major sections in the text.
        
        Returns:
            List of (start, end, text) tuples
        """
        sections = []
        
        # Find section markers
        matches = list(self.SECTION_MARKERS.finditer(text))
        
        if not matches:
            return [(0, len(text), text)]
        
        # Create sections from markers
        for i, match in enumerate(matches):
            start = match.start()
            
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)
            
            section_text = text[start:end].strip()
            if section_text:
                sections.append((start, end, section_text))
        
        # Add text before first section if any
        if matches and matches[0].start() > 0:
            intro_text = text[:matches[0].start()].strip()
            if intro_text:
                sections.insert(0, (0, matches[0].start(), intro_text))
        
        return sections if sections else [(0, len(text), text)]
    
    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge chunks that are too small."""
        if not chunks:
            return chunks
        
        merged = []
        current = None
        
        for chunk in chunks:
            if current is None:
                current = chunk
            elif len(current.content) + len(chunk.content) < self.config.min_chunk_size * 2:
                # Merge with current
                current.content = current.content + "\n\n" + chunk.content
                current.end_char = chunk.end_char
            else:
                if len(current.content) >= self.config.min_chunk_size:
                    merged.append(current)
                current = chunk
        
        if current and len(current.content) >= self.config.min_chunk_size:
            merged.append(current)
        
        return merged
    
    def _split_large_chunks(self, chunks: List[Chunk], doc_id: str) -> List[Chunk]:
        """Split chunks that are too large."""
        result = []
        
        for chunk in chunks:
            if len(chunk.content) > self.config.max_chunk_size:
                # Use fixed chunking for large chunks
                sub_chunks = self._chunk_fixed(chunk.content, doc_id)
                
                # Adjust positions
                for sub in sub_chunks:
                    sub.start_char += chunk.start_char
                    sub.end_char += chunk.start_char
                    result.append(sub)
            else:
                result.append(chunk)
        
        return result


def chunk_text(
    text: str,
    doc_id: str,
    strategy: str = "semantic",
    **kwargs
) -> List[Chunk]:
    """
    Convenience function for quick text chunking.
    
    Args:
        text: Text to chunk
        doc_id: Document identifier
        strategy: Chunking strategy name
        **kwargs: Additional config options
        
    Returns:
        List of Chunk objects
    """
    config = ChunkingConfig(
        strategy=ChunkingStrategy(strategy),
        **kwargs
    )
    chunker = DocumentChunker(config)
    return chunker.chunk_document(text, doc_id)


if __name__ == "__main__":
    # Test the module
    test_text = """
# Introduction to Tunisian History

Tunisia has a rich history spanning thousands of years. From the ancient Carthaginian civilization to the modern republic, the country has witnessed numerous transformations.

## Ancient Period

Carthage was founded by Phoenician colonists in the 9th century BC. It became a major power in the Mediterranean, rivaling Rome itself. The Punic Wars between Rome and Carthage shaped the ancient world.

The city of Carthage was known for its maritime trade and powerful navy. Hannibal Barca, one of history's greatest military commanders, led Carthaginian forces against Rome.

## Islamic Period

In the 7th century AD, Arab armies brought Islam to Tunisia. The city of Kairouan became an important center of Islamic learning and culture. The Great Mosque of Kairouan remains one of the most important Islamic monuments.

## French Protectorate

France established a protectorate over Tunisia in 1881. This period saw significant economic and social changes, as well as the growth of nationalist movements seeking independence.

## Independence and Modern Era

Tunisia gained independence in 1956 under the leadership of Habib Bourguiba. The country has since undergone significant political and social transformations, including the 2011 revolution that sparked the Arab Spring.
    """
    
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(test_text, "test_doc_001")
    
    print(f"Created {len(chunks)} chunks:\n")
    
    for chunk in chunks:
        print(f"Chunk {chunk.chunk_index}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Length: {len(chunk.content)} chars")
        print(f"  Position: {chunk.start_char}-{chunk.end_char}")
        print(f"  Preview: {chunk.content[:100]}...")
        print()
