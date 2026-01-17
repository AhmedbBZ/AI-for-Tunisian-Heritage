#!/usr/bin/env python3
"""
Simple Chat Interface for Tunisian Heritage RAG
================================================
Just run: python chat.py
"""

import sys
import os
from pathlib import Path

# Suppress verbose logging
os.environ.setdefault('LOGURU_LEVEL', 'ERROR')

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Quiet loguru
from loguru import logger
logger.remove()

from src.pipeline.rag_pipeline import RAGPipeline, RAGConfig


def main():
    print("\n" + "=" * 50)
    print("🇹🇳 Tunisian Heritage Chat")
    print("=" * 50)
    print("Loading pipeline...")
    
    # Initialize pipeline with optimized settings
    config = RAGConfig(
        vector_db_dir='./vector_db',
        llm_provider='lmstudio',
        llm_model='saka-14b-i1',
        top_k=5,
        min_score=0.1,  # Lower threshold for better recall
        use_mmr=False,  # Disable for speed
        max_new_tokens=512
    )
    
    pipeline = RAGPipeline(config)
    pipeline.initialize()
    
    print("Ready! Type your question (or 'quit' to exit)\n")
    
    while True:
        try:
            # Get user input
            question = input("You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            # Get response
            print("\nThinking...")
            response = pipeline.query(question)
            
            # Print answer
            print(f"\nAssistant: {response.answer}")
            print(f"\n[Confidence: {response.confidence:.0%} | Sources: {len(response.sources)}]\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
