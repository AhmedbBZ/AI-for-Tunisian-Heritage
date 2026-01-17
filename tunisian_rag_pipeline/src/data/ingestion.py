"""
Data Ingestion Module
=====================
Handles loading data from various file formats (TXT, JSON, PDF, CSV, etc.)
Extracts text content and metadata from the Tunisian heritage dataset.
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

from loguru import logger

# Optional imports with fallbacks
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger.warning("PyPDF2 not installed. PDF support disabled.")


@dataclass
class Document:
    """Represents a single document with content and metadata."""
    doc_id: str
    content: str
    source_file: str
    file_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    language: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Generate doc_id if not provided."""
        if not self.doc_id:
            # Create hash from content for unique ID
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]
            self.doc_id = f"{Path(self.source_file).stem}_{content_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "source_file": self.source_file,
            "file_type": self.file_type,
            "metadata": self.metadata,
            "language": self.language,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create document from dictionary."""
        return cls(**data)


class DataIngestion:
    """
    Handles ingestion of various file formats into Document objects.
    
    Supported formats:
    - .txt: Plain text files
    - .json: JSON files (with text content)
    - .pdf: PDF documents
    - .csv: CSV files
    - .md: Markdown files
    """
    
    SUPPORTED_FORMATS = {'.txt', '.json', '.pdf', '.csv', '.md'}
    
    def __init__(
        self,
        data_dir: str,
        encoding: str = 'utf-8',
        max_file_size_mb: int = 100
    ):
        """
        Initialize the data ingestion module.
        
        Args:
            data_dir: Path to the data directory
            encoding: Text encoding to use
            max_file_size_mb: Maximum file size to process in MB
        """
        self.data_dir = Path(data_dir)
        self.encoding = encoding
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        logger.info(f"Initialized DataIngestion with data_dir: {self.data_dir}")
    
    def discover_files(self, recursive: bool = True) -> List[Path]:
        """
        Discover all supported files in the data directory.
        
        Args:
            recursive: Whether to search subdirectories
            
        Returns:
            List of file paths
        """
        files = []
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in self.data_dir.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                # Check file size
                if file_path.stat().st_size <= self.max_file_size_bytes:
                    files.append(file_path)
                else:
                    logger.warning(f"Skipping large file: {file_path}")
        
        logger.info(f"Discovered {len(files)} files")
        return files
    
    def load_text_file(self, file_path: Path) -> Optional[Document]:
        """Load a plain text file."""
        try:
            with open(file_path, 'r', encoding=self.encoding, errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Empty file: {file_path}")
                return None
            
            # Try to load associated metadata
            metadata = self._load_metadata(file_path)
            
            return Document(
                doc_id="",
                content=content,
                source_file=str(file_path),
                file_type="text",
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return None
    
    def load_json_file(self, file_path: Path) -> Optional[Document]:
        """Load a JSON file."""
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, str):
                content = data
                metadata = {}
            elif isinstance(data, dict):
                # Try to find text content
                content = data.get('content') or data.get('text') or data.get('body') or str(data)
                metadata = {k: v for k, v in data.items() if k not in ['content', 'text', 'body']}
            elif isinstance(data, list):
                # Concatenate list items
                content = "\n\n".join(str(item) for item in data)
                metadata = {"is_list": True, "item_count": len(data)}
            else:
                content = str(data)
                metadata = {}
            
            if not content.strip():
                return None
            
            return Document(
                doc_id="",
                content=content,
                source_file=str(file_path),
                file_type="json",
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return None
    
    def load_pdf_file(self, file_path: Path) -> Optional[Document]:
        """Load a PDF file."""
        if not PDF_SUPPORT:
            logger.warning(f"Skipping PDF (PyPDF2 not installed): {file_path}")
            return None
        
        try:
            content_parts = []
            metadata = {"page_count": 0}
            
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                metadata["page_count"] = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        content_parts.append(f"[Page {page_num + 1}]\n{text}")
            
            content = "\n\n".join(content_parts)
            
            if not content.strip():
                logger.warning(f"Could not extract text from PDF: {file_path}")
                return None
            
            return Document(
                doc_id="",
                content=content,
                source_file=str(file_path),
                file_type="pdf",
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}")
            return None
    
    def load_csv_file(self, file_path: Path) -> Optional[Document]:
        """Load a CSV file."""
        try:
            content_parts = []
            
            with open(file_path, 'r', encoding=self.encoding, errors='ignore') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    return None
                
                # Convert rows to readable text
                for i, row in enumerate(rows):
                    row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                    content_parts.append(row_text)
            
            content = "\n".join(content_parts)
            
            return Document(
                doc_id="",
                content=content,
                source_file=str(file_path),
                file_type="csv",
                metadata={"row_count": len(rows), "columns": list(rows[0].keys()) if rows else []}
            )
        
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            return None
    
    def load_markdown_file(self, file_path: Path) -> Optional[Document]:
        """Load a Markdown file."""
        # Markdown is essentially text, use text loader
        doc = self.load_text_file(file_path)
        if doc:
            doc.file_type = "markdown"
        return doc
    
    def _load_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Try to load associated metadata file.
        Looks for a JSON file with the same name in a 'metadata' folder.
        """
        metadata = {}
        
        # Check for metadata file
        metadata_dir = file_path.parent.parent / "metadata"
        metadata_file = metadata_dir / f"{file_path.stem}.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding=self.encoding) as f:
                    metadata = json.load(f)
                logger.debug(f"Loaded metadata from {metadata_file}")
            except Exception as e:
                logger.warning(f"Could not load metadata file {metadata_file}: {e}")
        
        return metadata
    
    def load_file(self, file_path: Path) -> Optional[Document]:
        """
        Load a single file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document object or None if loading fails
        """
        suffix = file_path.suffix.lower()
        
        loaders = {
            '.txt': self.load_text_file,
            '.json': self.load_json_file,
            '.pdf': self.load_pdf_file,
            '.csv': self.load_csv_file,
            '.md': self.load_markdown_file
        }
        
        loader = loaders.get(suffix)
        if loader:
            return loader(file_path)
        else:
            logger.warning(f"Unsupported file format: {suffix}")
            return None
    
    def load_all(self, recursive: bool = True) -> Generator[Document, None, None]:
        """
        Load all supported files from the data directory.
        
        Args:
            recursive: Whether to search subdirectories
            
        Yields:
            Document objects
        """
        files = self.discover_files(recursive=recursive)
        
        for file_path in files:
            doc = self.load_file(file_path)
            if doc:
                yield doc
    
    def load_all_as_list(self, recursive: bool = True) -> List[Document]:
        """
        Load all documents as a list.
        
        Args:
            recursive: Whether to search subdirectories
            
        Returns:
            List of Document objects
        """
        return list(self.load_all(recursive=recursive))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the data directory."""
        stats = {
            "total_files": 0,
            "files_by_type": {},
            "total_size_mb": 0
        }
        
        for file_path in self.discover_files():
            stats["total_files"] += 1
            
            suffix = file_path.suffix.lower()
            stats["files_by_type"][suffix] = stats["files_by_type"].get(suffix, 0) + 1
            stats["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)
        
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        
        return stats


# Convenience function for quick loading
def load_documents(data_dir: str, **kwargs) -> List[Document]:
    """
    Convenience function to load all documents from a directory.
    
    Args:
        data_dir: Path to data directory
        **kwargs: Additional arguments for DataIngestion
        
    Returns:
        List of Document objects
    """
    ingestion = DataIngestion(data_dir, **kwargs)
    return ingestion.load_all_as_list()


if __name__ == "__main__":
    # Test the module
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "../tunisian_heritage_data"
    
    print(f"Testing DataIngestion with: {data_dir}")
    
    try:
        ingestion = DataIngestion(data_dir)
        
        # Get statistics
        stats = ingestion.get_statistics()
        print(f"\nData Statistics:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Files by type: {stats['files_by_type']}")
        print(f"  Total size: {stats['total_size_mb']} MB")
        
        # Load documents
        docs = ingestion.load_all_as_list()
        print(f"\nLoaded {len(docs)} documents")
        
        for doc in docs[:3]:  # Show first 3
            print(f"\n  - {doc.doc_id}")
            print(f"    Source: {doc.source_file}")
            print(f"    Type: {doc.file_type}")
            print(f"    Content length: {len(doc.content)} chars")
            print(f"    Preview: {doc.content[:100]}...")
    
    except Exception as e:
        print(f"Error: {e}")
