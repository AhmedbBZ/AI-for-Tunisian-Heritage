#!/usr/bin/env python3
"""
Query Interface Script
======================
Interactive CLI for querying the Tunisian Heritage RAG pipeline.

Usage:
    python query.py                           # Interactive mode
    python query.py "What is Tunisian couscous?"  # Single query
    python query.py --json                    # Output as JSON
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query the Tunisian Heritage RAG pipeline"
    )
    
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Query to process (if not provided, enters interactive mode)"
    )
    
    parser.add_argument(
        "--vector-db",
        type=str,
        default="./vector_db",
        help="Path to vector database directory"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "fr", "ar", "auto"],
        default="auto",
        help="Query language (auto for automatic detection)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation (only show retrieved documents)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def print_header():
    """Print CLI header."""
    print("\n" + "=" * 60)
    print("🇹🇳 Tunisian Heritage RAG System")
    print("=" * 60)
    print("Ask questions about Tunisian history, culture, and heritage.")
    print("Type 'quit' or 'exit' to quit, 'help' for commands.")
    print("=" * 60 + "\n")


def print_help():
    """Print help information."""
    print("""
Available commands:
  help          - Show this help message
  quit, exit    - Exit the program
  stats         - Show pipeline statistics
  clear         - Clear screen
  
Tips:
  - Ask questions in English, French, or Arabic
  - Be specific for better results
  - Use quotes for multi-word queries on command line
  
Examples:
  What caused the Tunisian revolution?
  ما هي الثورة التونسية؟
  Parlez-moi de la culture tunisienne
""")


def format_response_text(response) -> str:
    """Format response for text output."""
    lines = [
        "",
        "━" * 60,
        "📝 ANSWER:",
        "━" * 60,
        response.answer,
        "",
        "━" * 60,
        "📚 SOURCES:",
        "━" * 60,
    ]
    
    for i, source in enumerate(response.sources, 1):
        score = source.get('score', 0)
        source_info = source.get('source_info', 'Unknown')
        content_preview = source.get('content', '')[:150] + "..."
        
        lines.extend([
            f"[{i}] {source_info} (score: {score:.3f})",
            f"    {content_preview}",
            ""
        ])
    
    lines.extend([
        "━" * 60,
        f"🎯 Confidence: {response.confidence:.2f}",
        f"🏷️  Intent: {response.intent}",
        f"🌍 Language: {response.language}",
        f"⏱️  Time: {response.processing_time:.2f}s",
        "━" * 60,
        ""
    ])
    
    return "\n".join(lines)


def run_query(pipeline, query: str, args) -> dict:
    """Run a single query and return results."""
    language = None if args.language == "auto" else args.language
    
    response = pipeline.query(
        question=query,
        language=language,
        top_k=args.top_k
    )
    
    return response


def interactive_mode(pipeline, args):
    """Run interactive query mode."""
    print_header()
    
    while True:
        try:
            # Get user input
            user_input = input("🔍 Question: ").strip()
            
            # Handle special commands
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            if user_input.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_header()
                continue
            
            if user_input.lower() == 'stats':
                stats = pipeline.get_stats()
                print(f"\n📊 Pipeline Stats:")
                print(json.dumps(stats, indent=2))
                continue
            
            # Process query
            print("\n⏳ Processing...")
            response = run_query(pipeline, user_input, args)
            
            if args.json:
                print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
            else:
                print(format_response_text(response))
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            logger.error(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


def single_query_mode(pipeline, query: str, args):
    """Run a single query and exit."""
    response = run_query(pipeline, query, args)
    
    if args.json:
        print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_response_text(response))


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    
    # Check vector DB exists
    vector_db_path = Path(args.vector_db)
    if not vector_db_path.exists():
        print(f"\n❌ Error: Vector database not found at {vector_db_path}")
        print("Run 'python build_vector_db.py' first to build the database.")
        sys.exit(1)
    
    # Import and initialize pipeline
    try:
        from src.pipeline.rag_pipeline import RAGPipeline, RAGConfig
        
        config = RAGConfig(
            vector_db_dir=str(vector_db_path),
            top_k=args.top_k
        )
        
        print("🔄 Initializing RAG pipeline...")
        pipeline = RAGPipeline(config)
        pipeline.initialize()
        print("✅ Pipeline ready!\n")
    
    except Exception as e:
        print(f"\n❌ Error initializing pipeline: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Run appropriate mode
    if args.query:
        single_query_mode(pipeline, args.query, args)
    else:
        interactive_mode(pipeline, args)


if __name__ == "__main__":
    main()
