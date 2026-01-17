"""
Vector Store Module
===================
Manages vector database operations using ChromaDB.
Supports persistent storage, metadata filtering, and efficient retrieval.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from loguru import logger

# Lazy imports
chromadb = None
Settings = None


def _import_chromadb():
    """Lazy import chromadb."""
    global chromadb, Settings
    if chromadb is None:
        import chromadb as _chromadb
        from chromadb.config import Settings as _Settings
        chromadb = _chromadb
        Settings = _Settings
    return chromadb, Settings


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    persist_directory: str = "./vector_db"
    collection_name: str = "tunisian_heritage"
    distance_metric: str = "cosine"  # cosine, l2, ip
    embedding_dim: int = 384


@dataclass
class SearchResult:
    """Represents a search result."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata
        }


class VectorStore:
    """
    Vector database wrapper using ChromaDB.
    
    Features:
    - Persistent storage
    - Metadata filtering
    - Batch operations
    - Collection management
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize the vector store.
        
        Args:
            config: Vector store configuration
        """
        self.config = config or VectorStoreConfig()
        self.client = None
        self.collection = None
        
        # Ensure persist directory exists
        os.makedirs(self.config.persist_directory, exist_ok=True)
        
        logger.info(f"Initialized VectorStore at: {self.config.persist_directory}")
    
    def _connect(self) -> None:
        """Connect to ChromaDB."""
        if self.client is not None:
            return
        
        chroma, Settings = _import_chromadb()
        
        logger.info("Connecting to ChromaDB...")
        
        self.client = chroma.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        logger.info("Connected to ChromaDB")
    
    def _get_or_create_collection(self) -> Any:
        """Get or create the collection."""
        if self.collection is not None:
            return self.collection
        
        self._connect()
        
        # Map distance metric
        distance_map = {
            "cosine": "cosine",
            "l2": "l2",
            "ip": "ip"
        }
        
        metadata = {"hnsw:space": distance_map.get(self.config.distance_metric, "cosine")}
        
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata=metadata
        )
        
        logger.info(f"Using collection: {self.config.collection_name}")
        return self.collection
    
    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            ids: Unique identifiers for each document
            embeddings: Embedding vectors
            documents: Text content
            metadatas: Optional metadata dictionaries
        """
        collection = self._get_or_create_collection()
        
        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in ids]
        
        # Clean metadata (ChromaDB only supports certain types)
        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif isinstance(v, list):
                    clean_meta[k] = json.dumps(v)
                elif v is not None:
                    clean_meta[k] = str(v)
            clean_metadatas.append(clean_meta)
        
        # Add in batches
        batch_size = 500
        total = len(ids)
        
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            
            collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=clean_metadatas[i:end]
            )
            
            logger.debug(f"Added batch {i//batch_size + 1}: {end}/{total} documents")
        
        logger.info(f"Added {total} documents to vector store")
    
    def add_chunks(
        self,
        chunks_with_embeddings: List[Dict[str, Any]]
    ) -> None:
        """
        Add chunk objects with embeddings to the store.
        
        Args:
            chunks_with_embeddings: List of dicts with chunk data and embeddings
        """
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks_with_embeddings:
            chunk_id = chunk.get("chunk_id", chunk.get("id", ""))
            content = chunk.get("content", "")
            embedding = chunk.get("embedding", [])
            
            # Build metadata
            meta = {
                "doc_id": chunk.get("doc_id", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "source_file": chunk.get("source_file", ""),
                "file_type": chunk.get("file_type", ""),
                "language": chunk.get("language", ""),
            }
            
            # Add custom metadata
            if "metadata" in chunk:
                for k, v in chunk["metadata"].items():
                    if k not in meta:
                        meta[k] = v
            
            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(content)
            metadatas.append(meta)
        
        self.add_documents(ids, embeddings, documents, metadatas)
    
    def search(
        self,
        query_embedding: Union[List[float], np.ndarray],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            where: Metadata filter (ChromaDB format)
            where_document: Document content filter
            
        Returns:
            List of SearchResult objects
        """
        collection = self._get_or_create_collection()
        
        # Convert numpy array if needed
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        # Build query kwargs
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        
        if where:
            query_kwargs["where"] = where
        
        if where_document:
            query_kwargs["where_document"] = where_document
        
        # Execute query
        results = collection.query(**query_kwargs)
        
        # Convert to SearchResult objects
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # Convert distance to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0
                
                # For cosine distance, similarity = 1 - distance
                if self.config.distance_metric == "cosine":
                    score = 1 - distance
                else:
                    score = 1 / (1 + distance)  # Convert L2 to similarity-like
                
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    score=score,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {}
                ))
        
        return search_results
    
    def search_with_filter(
        self,
        query_embedding: Union[List[float], np.ndarray],
        top_k: int = 5,
        language: Optional[str] = None,
        doc_type: Optional[str] = None,
        source_file: Optional[str] = None,
        custom_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search with common filter options.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            language: Filter by language (ar, en, fr)
            doc_type: Filter by document type
            source_file: Filter by source file
            custom_filter: Additional ChromaDB filter
            
        Returns:
            List of SearchResult objects
        """
        where_conditions = []
        
        if language:
            where_conditions.append({"language": language})
        
        if doc_type:
            where_conditions.append({"file_type": doc_type})
        
        if source_file:
            where_conditions.append({"source_file": {"$contains": source_file}})
        
        if custom_filter:
            where_conditions.append(custom_filter)
        
        # Combine conditions
        where = None
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}
        
        return self.search(query_embedding, top_k=top_k, where=where)
    
    def get_document(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID.
        
        Args:
            chunk_id: Document/chunk ID
            
        Returns:
            Document data or None
        """
        collection = self._get_or_create_collection()
        
        result = collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas", "embeddings"]
        )
        
        if result["ids"]:
            return {
                "chunk_id": result["ids"][0],
                "content": result["documents"][0] if result["documents"] else "",
                "metadata": result["metadatas"][0] if result["metadatas"] else {},
                "embedding": result["embeddings"][0] if result["embeddings"] else []
            }
        
        return None
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents by ID.
        
        Args:
            ids: List of document IDs to delete
        """
        collection = self._get_or_create_collection()
        collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents")
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self._connect()
        
        try:
            self.client.delete_collection(self.config.collection_name)
            self.collection = None
            logger.info(f"Cleared collection: {self.config.collection_name}")
        except Exception as e:
            logger.warning(f"Error clearing collection: {e}")
        
        # Recreate empty collection
        self._get_or_create_collection()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        collection = self._get_or_create_collection()
        
        count = collection.count()
        
        return {
            "collection_name": self.config.collection_name,
            "document_count": count,
            "persist_directory": self.config.persist_directory,
            "distance_metric": self.config.distance_metric
        }
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        self._connect()
        collections = self.client.list_collections()
        return [c.name for c in collections]


def create_vector_store(
    persist_directory: str = "./vector_db",
    collection_name: str = "tunisian_heritage",
    **kwargs
) -> VectorStore:
    """
    Factory function to create a vector store.
    
    Args:
        persist_directory: Storage directory
        collection_name: Collection name
        **kwargs: Additional config options
        
    Returns:
        VectorStore instance
    """
    config = VectorStoreConfig(
        persist_directory=persist_directory,
        collection_name=collection_name,
        **kwargs
    )
    return VectorStore(config)


if __name__ == "__main__":
    # Test the module
    import tempfile
    
    print("Testing VectorStore:\n")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create vector store
        store = create_vector_store(
            persist_directory=tmpdir,
            collection_name="test_collection"
        )
        
        # Test data
        test_docs = [
            {"id": "doc1", "content": "The Tunisian Revolution began in 2010.", "language": "en"},
            {"id": "doc2", "content": "La révolution tunisienne a commencé en 2010.", "language": "fr"},
            {"id": "doc3", "content": "Carthage was an ancient Phoenician city.", "language": "en"},
        ]
        
        # Create fake embeddings (normally would use EmbeddingGenerator)
        embedding_dim = 384
        embeddings = [np.random.randn(embedding_dim).tolist() for _ in test_docs]
        
        # Add documents
        print("Adding documents...")
        store.add_documents(
            ids=[d["id"] for d in test_docs],
            embeddings=embeddings,
            documents=[d["content"] for d in test_docs],
            metadatas=[{"language": d["language"]} for d in test_docs]
        )
        
        # Get stats
        stats = store.get_collection_stats()
        print(f"Collection stats: {stats}")
        
        # Search
        print("\nSearching...")
        query_embedding = np.random.randn(embedding_dim).tolist()
        results = store.search(query_embedding, top_k=2)
        
        print(f"Found {len(results)} results:")
        for r in results:
            print(f"  - {r.chunk_id}: {r.content[:50]}... (score: {r.score:.4f})")
        
        # Search with filter
        print("\nSearching with language filter (en)...")
        results = store.search_with_filter(query_embedding, top_k=2, language="en")
        print(f"Found {len(results)} English results")
        
        print("\nTest completed successfully!")
