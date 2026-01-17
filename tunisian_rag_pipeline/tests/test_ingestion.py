"""
Tests for Data Ingestion Module
"""

import pytest
import tempfile
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingestion import DataIngestion, Document


class TestDocument:
    """Test Document dataclass."""
    
    def test_document_creation(self):
        """Test basic document creation."""
        doc = Document(
            content="Test content",
            source="test.txt",
            doc_type="txt"
        )
        
        assert doc.content == "Test content"
        assert doc.source == "test.txt"
        assert doc.doc_type == "txt"
        assert doc.doc_id is not None
        assert doc.metadata == {}
    
    def test_document_with_metadata(self):
        """Test document with metadata."""
        doc = Document(
            content="Test",
            source="test.txt",
            doc_type="txt",
            metadata={"author": "Test Author"}
        )
        
        assert doc.metadata["author"] == "Test Author"
    
    def test_document_to_dict(self):
        """Test converting document to dictionary."""
        doc = Document(
            content="Test",
            source="test.txt",
            doc_type="txt"
        )
        
        d = doc.to_dict()
        assert d["content"] == "Test"
        assert d["source"] == "test.txt"


class TestDataIngestion:
    """Test DataIngestion class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test TXT file
            txt_path = Path(tmpdir) / "test.txt"
            txt_path.write_text("This is test content.", encoding="utf-8")
            
            # Create test JSON file
            json_path = Path(tmpdir) / "test.json"
            json_data = {
                "content": "JSON content",
                "title": "Test Title"
            }
            json_path.write_text(json.dumps(json_data), encoding="utf-8")
            
            yield tmpdir
    
    def test_initialization(self, temp_data_dir):
        """Test DataIngestion initialization."""
        ingestion = DataIngestion(temp_data_dir)
        assert ingestion.data_dir == Path(temp_data_dir)
    
    def test_load_txt_file(self, temp_data_dir):
        """Test loading TXT file."""
        ingestion = DataIngestion(temp_data_dir)
        txt_path = Path(temp_data_dir) / "test.txt"
        
        doc = ingestion.load_text_file(str(txt_path))
        
        assert doc is not None
        assert "test content" in doc.content
        assert doc.doc_type == "txt"
    
    def test_load_json_file(self, temp_data_dir):
        """Test loading JSON file."""
        ingestion = DataIngestion(temp_data_dir)
        json_path = Path(temp_data_dir) / "test.json"
        
        doc = ingestion.load_json_file(str(json_path))
        
        assert doc is not None
        assert "JSON content" in doc.content
    
    def test_load_all(self, temp_data_dir):
        """Test loading all files."""
        ingestion = DataIngestion(temp_data_dir)
        docs = ingestion.load_all()
        
        assert len(docs) >= 2  # At least txt and json
    
    def test_empty_directory(self):
        """Test with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestion = DataIngestion(tmpdir)
            docs = ingestion.load_all()
            assert len(docs) == 0
    
    def test_nonexistent_file(self, temp_data_dir):
        """Test loading nonexistent file."""
        ingestion = DataIngestion(temp_data_dir)
        doc = ingestion.load_text_file("nonexistent.txt")
        assert doc is None


class TestDocumentContent:
    """Test document content handling."""
    
    def test_unicode_content(self):
        """Test handling Unicode content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with Arabic and French text
            path = Path(tmpdir) / "unicode.txt"
            content = "تونس - La Tunisie - Tunisia"
            path.write_text(content, encoding="utf-8")
            
            ingestion = DataIngestion(tmpdir)
            doc = ingestion.load_text_file(str(path))
            
            assert "تونس" in doc.content
            assert "Tunisie" in doc.content
    
    def test_large_file(self):
        """Test handling large files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "large.txt"
            content = "Test content. " * 10000  # ~130KB
            path.write_text(content, encoding="utf-8")
            
            ingestion = DataIngestion(tmpdir)
            doc = ingestion.load_text_file(str(path))
            
            assert len(doc.content) > 100000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
