#!/usr/bin/env python3
"""
Build Vector Database Script
============================
Script to build or rebuild the vector database from source documents.

Usage:
    python build_vector_db.py --data-dir ../tunisian_heritage_data --output-dir ./vector_db
    python build_vector_db.py --rebuild  # Rebuild from scratch
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.data.ingestion import DataIngestion
from src.data.preprocessing import TextPreprocessor, PreprocessingConfig
from src.data.chunking import DocumentChunker, ChunkingConfig, ChunkingStrategy
from src.embeddings.embedder import EmbeddingGenerator, EmbeddingConfig
from src.retrieval.vector_store import VectorStore, VectorStoreConfig
from src.utils.helpers import (
    ensure_directory, 
    timed_block, 
    format_bytes, 
    format_duration,
    Timer,
    setup_logging
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build vector database for Tunisian Heritage RAG pipeline"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../tunisian_heritage_data",
        help="Path to data directory containing source documents"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./vector_db",
        help="Path to output vector database directory"
    )
    
    parser.add_argument(
        "--collection",
        type=str,
        default="tunisian_heritage",
        help="Name of the vector database collection"
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model to use"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Size of text chunks"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks"
    )
    
    parser.add_argument(
        "--chunk-strategy",
        type=str,
        default="semantic",
        choices=["fixed", "paragraph", "sentence", "semantic"],
        help="Chunking strategy to use"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding generation"
    )
    
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the database from scratch (delete existing)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main function to build vector database."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)
    
    total_timer = Timer()
    
    logger.info("=" * 60)
    logger.info("Tunisian Heritage RAG - Vector Database Builder")
    logger.info("=" * 60)
    
    # Resolve paths
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Check data directory exists
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Handle rebuild
    if args.rebuild and output_dir.exists():
        logger.warning("Rebuild flag set - removing existing database")
        import shutil
        shutil.rmtree(output_dir)
    
    ensure_directory(output_dir)
    
    # Step 1: Load documents
    logger.info("\n📁 Step 1: Loading documents...")
    with timed_block("Document loading"):
        ingestion = DataIngestion(str(data_dir))
        documents = list(ingestion.load_all())  # Convert generator to list
        logger.info(f"Loaded {len(documents)} documents")
        
        # Calculate total size
        total_chars = sum(len(doc.content) for doc in documents)
        logger.info(f"Total content: {total_chars:,} characters")
    
    if not documents:
        logger.error("No documents loaded!")
        sys.exit(1)
    
    # Step 2: Preprocess documents
    logger.info("\n🧹 Step 2: Preprocessing documents...")
    with timed_block("Preprocessing"):
        preprocessor = TextPreprocessor()
        processed_docs = []
        
        for doc in documents:
            # Process the document text
            cleaned_text, language = preprocessor.process_document(doc.content)
            
            # Update document with processed content
            doc.content = cleaned_text
            if language:
                doc.metadata['language'] = language
            doc.metadata['preprocessed'] = True
            processed_docs.append(doc)
        
        logger.info(f"Preprocessed {len(processed_docs)} documents")
    
    # Step 3: Chunk documents
    logger.info("\n✂️ Step 3: Chunking documents...")
    with timed_block("Chunking"):
        chunk_config = ChunkingConfig(
            strategy=ChunkingStrategy(args.chunk_strategy),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        chunker = DocumentChunker(chunk_config)
        
        all_chunks = []
        for doc in processed_docs:
            chunks = chunker.chunk_document(doc.content, doc.doc_id, doc.metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks")
        
        # Stats
        avg_chunk_size = sum(len(c.content) for c in all_chunks) / len(all_chunks) if all_chunks else 0
        logger.info(f"Average chunk size: {avg_chunk_size:.0f} characters")
    
    # Step 4: Initialize embedding model
    logger.info("\n🔤 Step 4: Initializing embedding model...")
    with timed_block("Embedding model initialization"):
        embed_config = EmbeddingConfig(
            model_name=args.embedding_model,
            batch_size=args.batch_size
        )
        embedder = EmbeddingGenerator(embed_config)
        logger.info(f"Embedding model: {args.embedding_model}")
        logger.info(f"Embedding dimension: {embedder.dimension}")
    
    # Step 5: Initialize vector store
    logger.info("\n🗄️ Step 5: Initializing vector store...")
    with timed_block("Vector store initialization"):
        store_config = VectorStoreConfig(
            persist_directory=str(output_dir),
            collection_name=args.collection
        )
        vector_store = VectorStore(store_config)
        logger.info(f"Collection: {args.collection}")
    
    # Step 6: Generate embeddings and add to store
    logger.info("\n📊 Step 6: Generating embeddings and building index...")
    with timed_block("Embedding generation and indexing"):
        # Process in batches
        batch_size = args.batch_size
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            # Generate embeddings for this batch
            chunks_with_embeddings = embedder.embed_chunks(batch)
            
            # Add to vector store
            vector_store.add_chunks(chunks_with_embeddings)
        
        logger.info(f"Added {len(all_chunks)} chunks to vector store")
    
    # Step 7: Verify and get stats
    logger.info("\n📈 Step 7: Verifying database...")
    stats = vector_store.get_collection_stats()
    logger.info(f"Collection stats: {stats}")
    
    # Save build info
    build_info = {
        "build_time": datetime.now().isoformat(),
        "data_directory": str(data_dir),
        "num_documents": len(documents),
        "num_chunks": len(all_chunks),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "chunk_strategy": args.chunk_strategy,
        "embedding_model": args.embedding_model,
        "collection_name": args.collection,
        "total_build_time_seconds": total_timer.elapsed
    }
    
    import json
    build_info_path = output_dir / "build_info.json"
    with open(build_info_path, 'w') as f:
        json.dump(build_info, f, indent=2)
    logger.info(f"Build info saved to: {build_info_path}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("BUILD COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"📄 Documents processed: {len(documents)}")
    logger.info(f"✂️  Chunks created: {len(all_chunks)}")
    logger.info(f"🗄️  Vector DB location: {output_dir}")
    logger.info(f"⏱️  Total time: {format_duration(total_timer.elapsed)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
