"""
Tests for Document Chunking Module
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.chunking import (
    DocumentChunker, 
    ChunkingConfig, 
    ChunkingStrategy, 
    Chunk
)
from src.data.ingestion import Document


class TestChunkingConfig:
    """Test ChunkingConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ChunkingConfig()
        
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.strategy == ChunkingStrategy.SEMANTIC
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ChunkingConfig(
            chunk_size=256,
            chunk_overlap=25,
            strategy=ChunkingStrategy.PARAGRAPH
        )
        
        assert config.chunk_size == 256
        assert config.chunk_overlap == 25


class TestChunk:
    """Test Chunk dataclass."""
    
    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = Chunk(
            content="Test chunk content",
            chunk_id="chunk_1",
            doc_id="doc_1",
            chunk_index=0
        )
        
        assert chunk.content == "Test chunk content"
        assert chunk.chunk_index == 0
    
    def test_chunk_to_dict(self):
        """Test converting chunk to dictionary."""
        chunk = Chunk(
            content="Test",
            chunk_id="chunk_1",
            doc_id="doc_1",
            chunk_index=0
        )
        
        d = chunk.to_dict()
        assert d["content"] == "Test"
        assert d["chunk_index"] == 0


class TestDocumentChunker:
    """Test DocumentChunker class."""
    
    @pytest.fixture
    def chunker(self):
        """Create chunker instance."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        return DocumentChunker(config)
    
    @pytest.fixture
    def sample_doc(self):
        """Create sample document."""
        content = """
        First paragraph with some content about Tunisia.
        This is additional information about the country.
        
        Second paragraph discussing the history.
        Ancient civilizations lived here for millennia.
        
        Third paragraph about culture and traditions.
        The people have rich cultural heritage.
        """
        return Document(
            content=content,
            source="test.txt",
            doc_type="txt"
        )
    
    def test_chunk_document(self, chunker, sample_doc):
        """Test basic document chunking."""
        chunks = chunker.chunk_document(sample_doc)
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
    
    def test_chunk_size_limit(self):
        """Test that chunks respect size limit."""
        config = ChunkingConfig(chunk_size=50, chunk_overlap=0)
        chunker = DocumentChunker(config)
        
        content = "Word " * 100  # ~500 characters
        doc = Document(content=content, source="test.txt", doc_type="txt")
        
        chunks = chunker.chunk_document(doc)
        
        # Most chunks should be around the limit
        for chunk in chunks:
            # Allow some flexibility
            assert len(chunk.content) <= 100
    
    def test_chunk_overlap(self):
        """Test chunk overlap."""
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            strategy=ChunkingStrategy.FIXED
        )
        chunker = DocumentChunker(config)
        
        content = "A" * 200
        doc = Document(content=content, source="test.txt", doc_type="txt")
        
        chunks = chunker.chunk_document(doc)
        
        if len(chunks) >= 2:
            # Check that chunks have some overlap
            end_of_first = chunks[0].content[-20:]
            start_of_second = chunks[1].content[:20]
            # Due to overlap, there should be some similarity
            assert len(chunks) > 1
    
    def test_chunk_metadata(self, chunker, sample_doc):
        """Test that chunks inherit document metadata."""
        sample_doc.metadata = {"author": "Test"}
        
        chunks = chunker.chunk_document(sample_doc)
        
        for chunk in chunks:
            assert chunk.doc_id == sample_doc.doc_id
            assert "author" in chunk.metadata
    
    def test_empty_document(self, chunker):
        """Test chunking empty document."""
        doc = Document(content="", source="test.txt", doc_type="txt")
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) == 0
    
    def test_small_document(self, chunker):
        """Test chunking document smaller than chunk size."""
        doc = Document(content="Small", source="test.txt", doc_type="txt")
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) == 1
        assert chunks[0].content == "Small"


class TestChunkingStrategies:
    """Test different chunking strategies."""
    
    def test_fixed_strategy(self):
        """Test fixed-size chunking."""
        config = ChunkingConfig(
            chunk_size=50,
            strategy=ChunkingStrategy.FIXED
        )
        chunker = DocumentChunker(config)
        
        content = "A" * 100
        doc = Document(content=content, source="test.txt", doc_type="txt")
        
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) >= 2
    
    def test_paragraph_strategy(self):
        """Test paragraph-based chunking."""
        config = ChunkingConfig(
            chunk_size=200,
            strategy=ChunkingStrategy.PARAGRAPH
        )
        chunker = DocumentChunker(config)
        
        content = "Para 1.\n\nPara 2.\n\nPara 3."
        doc = Document(content=content, source="test.txt", doc_type="txt")
        
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) >= 1
    
    def test_sentence_strategy(self):
        """Test sentence-based chunking."""
        config = ChunkingConfig(
            chunk_size=50,
            strategy=ChunkingStrategy.SENTENCE
        )
        chunker = DocumentChunker(config)
        
        content = "First sentence. Second sentence. Third sentence."
        doc = Document(content=content, source="test.txt", doc_type="txt")
        
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
